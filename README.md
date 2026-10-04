# MapPilot

这是一个面向SLAM 定位与建图的SLAM 定位与建图库。长期目标是提供位姿表示与李群运算、扫描匹配、因子图后端、闭环检测与位姿图优化、多传感器融合、重定位和地图表示，把定位建图沉淀为可复用库。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m mappilot.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `MAPPILOT_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 位姿表示

`mappilot.geometry.Pose3` 提供右手坐标系下的 SE(3) 刚体位姿（局部坐标 → 父坐标），平移按 `(x, y, z)`、四元数按 `(w, x, y, z)` 排列。支持从平移加四元数或 4×4 齐次矩阵创建，提供复合、求逆、单点与点序列变换，以及六维切向量 `(wx, wy, wz, vx, vy, vz)` 下的 `exp`/`log` 映射（左扰动约定）。该模块为纯 Python 实现，无需启动 HTTP 服务：

```bash
PYTHONPATH=src python3 -c "from mappilot.geometry import Pose3; print(Pose3.exp((0, 0, 1, 1, 2, 3)).log())"
```

## 扫描匹配

`mappilot.registration.point_to_point_icp` 用点到点 ICP 估计把源点云变换到目标点云的 `Pose3`。每轮先按当前位姿变换源点，再为每个源点选择最近目标点（目标点可重复选择，等距时取索引最小者），可选 `max_correspondence_distance` 距离门限过滤配对，然后用中心化互协方差的 SVD（Kabsch）求解使平方距离和最小的刚体变换，行列式强制为 +1，旋转不含镜像。相邻两轮 RMSE 绝对差不超过 `tolerance` 即收敛；用尽 `max_iterations` 不抛异常，返回最后一轮结果且 `converged=False`。

```python
from mappilot.registration import point_to_point_icp

result = point_to_point_icp(
    source_points,          # 至少三个三维点
    target_points,
    initial_pose=None,              # 默认单位位姿
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=None,  # None 表示不限制配对距离
)
result.pose            # 估计的 Pose3
result.converged       # 是否在迭代上限内收敛
result.iterations      # 实际完成的位姿更新次数
result.rmse            # 最终配对的欧氏距离均方根
result.correspondences # (源点索引, 目标点索引)，按源点索引升序
```

`ICPResult` 不可变；`correspondences` 与 `rmse` 都基于返回位姿重新计算，二者始终一致。两个点云都必须至少包含三个有限实数的三维点；有效配对少于三个或配对点共线等无法唯一约束三维刚体位姿时抛出 `ValueError`，点云不可迭代、点维度错误、坐标非实数或为布尔值、`initial_pose` 不是 `Pose3`、参数类型错误时抛出 `TypeError`。该模块仅依赖标准库，导入时不会启动 HTTP 服务。

`mappilot.registration.point_to_plane_icp` 在同样的最近点配对循环上引入目标表面法向约束：额外接收与目标点一一对应的法向量（计算前按自身长度归一化，正比例缩放不改变结果），每轮求解使变换后源点到对应目标切平面有符号投影误差平方和最小的六自由度位姿增量（线性化后经 `Pose3.exp` 左乘更新）。`initial_pose`、`max_iterations`、`tolerance`、`max_correspondence_distance` 语义与点到点 ICP 一致。

```python
from mappilot.registration import point_to_plane_icp

result = point_to_plane_icp(
    source_points,          # 至少六个三维点
    target_points,          # 至少六个三维点
    target_normals,         # 与目标点一一对应的非零有限法向量
    initial_pose=None,
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=None,
)
result.pose            # 估计的 Pose3
result.converged       # 是否在迭代上限内收敛
result.iterations      # 实际完成的位姿更新次数
result.rmse            # 最终配对在目标切平面上的有符号投影误差均方根
result.correspondences # (源点索引, 目标点索引)，按源点索引升序
```

`PointToPlaneICPResult` 同样不可变，`correspondences` 与 `rmse` 基于返回位姿重新计算。源点、目标点少于六个，法向量数量与目标点不符或长度为零，有效配对少于六个，以及法向约束不能唯一确定六自由度位姿增量（如法向全部平行的平面）时抛出 `ValueError`；类型错误的异常语义与点到点 ICP 相同。

## 位姿图优化

`mappilot.pose_graph.optimize_pose_graph` 以节点初值和相对位姿约束生成全局一致轨迹。节点为有序且 ID 唯一的 `(id, pose)` 序列，`id` 为非布尔整数，`pose` 为节点的 `Pose3` 初值；约束为 `(from_id, to_id, measurement, information)` 序列，其中 `measurement` 是从 `from` 节点到 `to` 节点的测量相对位姿，`information` 是按 `Pose3.log` 六个分量排列的对称正定 6×6 信息矩阵（不对称判断绝对容差 1e-12）。每条约束的残差与总误差为

```
e = log( z^{-1} · T_i^{-1} · T_j )
E = Σ eᵀ Ω e
```

优化采用左扰动约定（`T ← exp(dx)·T`）下的高斯-牛顿迭代，残差雅可比为 `J_to = J_l(e)^{-1}·Ad_{z^{-1}T_i^{-1}}`、`J_from = −J_to`，固定节点的行列从法方程中消去后整体求解并同时更新所有自由节点。`fixed_node_ids` 不得为空，且每个（无向图）连通分量至少有一个固定节点；固定节点的位姿逐值不变。

```python
from mappilot.pose_graph import optimize_pose_graph

result = optimize_pose_graph(
    nodes,             # (整数 ID, Pose3) 有序序列，ID 唯一
    constraints,       # (起点 ID, 终点 ID, measurement Pose3, 6x6 information)
    fixed_node_ids,    # 非空；每个连通分量至少一个
    max_iterations=50, # 高斯-牛顿更新次数上限，正整数
    tolerance=1e-6,    # 相邻两次更新总误差绝对差收敛阈值，正有限实数
)
result.nodes         # (ID, Pose3) 元组，按输入顺序排列
result.converged     # 是否收敛
result.iterations    # 实际完成的高斯-牛顿更新次数
result.initial_error # 从初始位姿重新计算的总误差
result.final_error   # 从返回位姿重新计算的总误差
```

`PoseGraphResult` 不可变。初始总误差不大于 `tolerance` 时直接返回 `converged=True`、`iterations=0` 且位姿不变；否则相邻两次更新的总误差绝对差不大于 `tolerance` 时收敛，用尽 `max_iterations` 不抛异常，返回最后结果并令 `converged=False`。`initial_error` 与 `final_error` 均从对应位姿重新计算，与 `nodes` 始终一致。

输入不可迭代、节点 ID 不是非布尔整数、位姿或测量不是 `Pose3`、约束或信息矩阵形状错误、矩阵元素不是非布尔实数，以及 `max_iterations` 或 `tolerance` 类型错误时抛出 `TypeError`；节点为空或 ID 重复、约束引用未知节点、固定节点未知、固定集合为空、存在未固定的连通分量、矩阵含非有限值、以 1e-12 绝对容差判断不对称或不是正定矩阵，以及 `max_iterations` 非正或 `tolerance` 非正、非有限时抛出 `ValueError`。该模块仅依赖标准库与 `mappilot.geometry`，导入时不会启动 HTTP 服务。

## 闭环检测

`mappilot.loop_closure.detect_loop_closures` 在按时间排列的关键帧序列中发现并几何验证闭环。每个关键帧为 `(id, pose, points)`：唯一非布尔整数 ID、世界系 `Pose3` 初值、该帧局部坐标系中至少三个有限实数三维点。仅检查输入位置差不小于 `min_separation` 的早帧-晚帧对；位姿原点距离超过 `translation_threshold` 或相对旋转最小测地角超过 `rotation_threshold` 的配对在几何验证前被跳过。其余配对以两个位姿初值推导早帧点云到晚帧点云的初始变换 `T_late^{-1}·T_early`，再按 `point_to_point_icp` 的既有规则细化；只有 ICP 收敛、最终有效配对数不少于 `min_correspondences` 且 RMSE 不大于 `max_rmse` 时才接受。配对不足、几何退化或迭代耗尽只拒绝该候选，不影响其他配对。

```python
from mappilot.loop_closure import detect_loop_closures

closures = detect_loop_closures(
    keyframes,            # (整数 ID, Pose3, 局部点云) 按时间排列
    min_separation=10,    # 最小帧序间隔，正整数
    translation_threshold=10.0,   # 平移预筛阈值，非负有限实数
    rotation_threshold=1.5707963267948966,  # 旋转预筛阈值，[0, pi]
    min_correspondences=3,        # 最少配对数，不小于三
    max_rmse=0.5,                 # 最大验收 RMSE，非负有限实数
    max_iterations=50,            # 以下三项沿用 point_to_point_icp 语义
    tolerance=1e-6,
    max_correspondence_distance=None,
)
closure.early_id         # 早帧 ID
closure.late_id          # 晚帧 ID
closure.relative_pose    # 相对 Pose3，可直接作为 optimize_pose_graph 边的 measurement
closure.rmse             # 最终 RMSE
closure.correspondences  # (早帧点索引, 晚帧点索引)，按早帧索引升序
```

返回值及其元素均不可变，结果按早帧、晚帧的输入位置升序排列；空输入返回空元组，生成器只被消费一次，输入对象不被修改。`relative_pose` 采用 `T_early.inverse().compose(T_late)` 的位姿图约定，等于 ICP 所得早帧到晚帧扫描变换之逆。关键帧或点云不可迭代、记录形状错误、ID 类型错误、位姿不是 `Pose3`、坐标维度错误或坐标不是非布尔实数时抛出 `TypeError`；ID 重复、点云少于三个点、坐标含非有限值或控制参数越界时抛出 `ValueError`。该模块仅依赖标准库与 `mappilot.geometry`、`mappilot.registration`，导入时不会启动 HTTP 服务。

## 地图表示

`mappilot.mapping.build_occupancy_grid` 把按时间排列的激光帧构建为确定性的二维占据栅格。每帧为 `(id, pose, points)`：唯一非布尔整数 ID、传感器世界系 `Pose3`、该帧局部坐标系中至少一个有限实数三维点。构图先用每帧位姿把点变换到世界系，再投影到 x-y 平面（忽略 z），然后以传感器格为起点、命中格为终点执行标准二维 Bresenham 遍历：终点之前的格记为空闲，终点格记为占据；同一格冲突时占据优先于空闲，未被射线覆盖的格保持未知。

栅格边界是包含所有传感器原点及有效射线终点的最小分辨率对齐矩形，四边增加 `padding` 后向外对齐；`origin` 为左下角外边界，世界坐标 `p` 到格索引使用 `floor((p - origin) / resolution)`，因此恰好落在网格线上的终点由该线之后的格覆盖。`max_range` 为 `None` 时保留全部命中；设置后水平距离严格超过该值的射线截断到量程边界，截断终点（及其射线）只记空闲、不产生占据；水平距离为零的有效点直接把所在格记为占据。

```python
from mappilot.mapping import build_occupancy_grid

grid = build_occupancy_grid(
    scans,              # (整数 ID, Pose3, 局部三维点) 按时间排列，每帧至少一个点
    resolution=0.05,    # 正有限实数
    padding=0.0,        # 非负有限实数，四边外扩后向外对齐
    max_range=None,     # None 或正有限实数；超出的射线截断为空闲
)
grid.resolution  # 浮点分辨率
grid.origin      # 左下角外边界 (x, y)
grid.width       # 列数
grid.height      # 行数
grid.data        # 一维元组，按 y 递增、每行 x 递增；-1/0/100 = 未知/空闲/占据
```

`OccupancyGrid2D` 不可变，格 `(ix, iy)` 的索引为 `iy * width + ix`。输入帧或点的顺序变化不改变相同观测集合的结果；`scans` 及点云生成器只被消费一次且输入不被修改，空 `scans` 或空点云会被拒绝。`scans` 或点云不可迭代、帧形状错误、ID 不是非布尔整数、位姿不是 `Pose3`、点维度错误、坐标或参数不是非布尔实数时抛出 `TypeError`；重复 ID、坐标非有限、空帧或空点云、`resolution` 非正或非有限、`padding` 为负或非有限、`max_range` 非正或非有限时抛出 `ValueError`。该模块仅依赖标准库与 `mappilot.geometry`，导入时不会启动 HTTP 服务。

## 重定位

`mappilot.localization.correlative_scan_match` 在有粗略先验时（重定位与丢失恢复）对二维占据栅格做确定性的相关扫描匹配：在初值周围按规则网格穷举候选位姿并逐一打分，返回最优者。候选位姿把 `dx`、`dy` 直接加到初值世界平移（z 保持不变），并在初始旋转左侧复合绕世界 z 轴的 `dyaw`；每个偏移都是绝对值不超过对应窗口的整数步长倍数，且始终包含零，因此初值本身总是候选。

每个候选按 `Pose3` 语义变换扫描点，再按 `floor((p - origin) / resolution)` 投影到栅格（z 不参与评分）：占据格记 +1，空闲格记 -1，未知格与地图外点不计入已知点数；已知点数达到 `min_known_points` 的候选以这些值的平均数为得分。最高分者胜出，同分时依次选择已知点更多、`dx²+dy²+dyaw²` 更小、`dx`、`dy`、`dyaw` 升序者。

```python
from mappilot.localization import correlative_scan_match

result = correlative_scan_match(
    grid,                # OccupancyGrid2D
    scan_points,         # 局部坐标系三维点，至少一个
    initial_pose,        # 粗略先验 Pose3
    x_window=1.0, x_step=0.05,        # 非负有限窗口，正有限步长
    y_window=1.0, y_step=0.05,
    yaw_window=0.5, yaw_step=0.1,     # 偏航单位为弧度
    min_known_points=10, # 候选可评分的最少已知点数，非布尔整数，至少为 1
    min_score=0.5,       # 验收阈值，[-1, 1]
)
result.pose                 # 最优候选位姿（无合格候选时为原初值）
result.matched              # 最佳得分达到 min_score 才为 True
result.score                # 最佳候选得分；无合格候选时为 None
result.known_points         # 最佳候选已知点数；无合格候选时为 0
result.evaluated_candidates # 实际搜索的候选总数
```

`CorrelativeScanMatchResult` 不可变。最佳候选低于阈值时仍返回该位姿与得分（`matched=False`）；没有候选达到 `min_known_points` 时返回原初值、`matched=False`、`score=None`、`known_points=0`。扫描生成器只被消费一次，输入不被修改，相同输入逐值一致。扫描不可迭代、点维度错误、坐标或实数参数类型错误、`min_known_points` 不是非布尔整数、地图或初值类型错误时抛出 `TypeError`；扫描为空、坐标非有限、窗口为负或非有限、步长非正或非有限、`min_known_points` 小于一、`min_score` 不在 `[-1, 1]` 时抛出 `ValueError`。该模块仅依赖标准库与 `mappilot.geometry`、`mappilot.mapping`，导入时不会启动 HTTP 服务。

## 运动畸变补偿

`mappilot.motion_compensation.deskew_point_cloud` 把一帧扫描期间机体连续运动造成的点云畸变统一补偿到指定参考时刻。扫描点按原顺序给出，每条记录为相对 `scan_time` 的秒偏移与一个传感器系三维点；轨迹为严格递增的绝对秒时间戳与机体系到世界系 `Pose3` 样本；`sensor_to_body` 是传感器系到机体系的 `Pose3`；`reference_time` 缺省等于 `scan_time`。

每个点按其绝对时刻 `scan_time + offset` 在两侧轨迹样本之间插值机体位姿：平移逐分量线性插值，旋转沿单位四元数最短弧插值；恰好命中样本时直接使用该 `Pose3`。点先经采样时刻的 `sensor_to_body` 与机体世界位姿变到世界系，再经参考时刻传感器世界位姿的逆变换拉回参考时刻传感器系：

```
p_ref = T_sensor_world(reference_time)^{-1} · T_body_world(t) · T_sensor_body · p
```

```python
from mappilot.motion_compensation import deskew_point_cloud

points = deskew_point_cloud(
    scan_points,             # (相对 scan_time 的秒偏移, 三维点) 按原顺序排列
    scan_time,               # 扫描原点绝对时间戳（秒）
    trajectory,              # (绝对秒时间戳, 机体到世界 Pose3)，严格递增，至少两条
    sensor_to_body,          # 传感器系到机体系 Pose3
    reference_time=None,     # None 时等于 scan_time
    max_interpolation_gap=None,  # None 或正有限实数
)
```

返回不可变三维点元组组成的元组，顺序与数量和输入完全一致；静止轨迹下逐值保留点。所有点时刻及 `reference_time` 必须落在轨迹时间戳闭区间内；需要跨越的相邻样本间隔超过 `max_interpolation_gap` 时抛出 `ValueError`，恰好命中样本不受此限制。输入生成器只被消费一次，调用方对象不被修改。不可迭代输入、记录形状或点维度错误、时间与坐标不是非布尔实数、轨迹位姿或外参不是 `Pose3` 时抛出 `TypeError`；空点云、轨迹样本少于两条、时间戳重复或逆序、任一数值非有限、`max_interpolation_gap` 非正时抛出 `ValueError`。该模块仅依赖标准库与 `mappilot.geometry`，纯 Python 实现，导入时不会启动 HTTP 服务。

`mappilot.inertial.preintegrate_imu` 把一段按时间排列的 IMU 样本压缩为可供视觉或激光后端复用的单个相对运动增量。每个样本为 `(timestamp, acceleration, angular_velocity)`：`timestamp` 为绝对秒时间戳，后两项是采样时刻机体系中的三轴比力与三轴角速度。时间戳必须严格递增且至少两条；偏置在全段恒定，区间 `[t[k], t[k+1]]` 采用左端样本的偏置校正测量。

积分从单位旋转、零速度、零位置开始。对每个区间长度 `dt`，先用区间开始处的增量旋转把校正比力转到起始机体系，再按

```
Δp' = Δp + Δv·dt + 0.5·a·dt²
Δv' = Δv + a·dt
ΔR' = ΔR · Pose3.exp((ωx·dt, ωy·dt, ωz·dt, 0, 0, 0))
```

更新（先位置、速度，再右复合旋转增量）。

```python
from mappilot.inertial import preintegrate_imu

result = preintegrate_imu(
    samples,                          # (绝对秒时间戳, 三轴比力, 三轴角速度)，严格递增，至少两条
    accelerometer_bias=(0, 0, 0),     # 三轴加速度计偏置，缺省为零
    gyroscope_bias=(0, 0, 0),         # 三轴陀螺偏置，缺省为零
    max_interval=None,                # None 不限制间隔，或正有限实数
)
result.delta_pose       # 平移为 Δp、旋转为累计 ΔR 的 Pose3
result.delta_velocity   # 起始机体系中的速度增量
result.duration         # 末首时间戳之差
result.intervals        # 样本数减一
```

`PreintegratedImu` 与其所有字段均不可变；零测量产生单位 `delta_pose` 与零速度，不均匀采样按相同规则逐区间处理。相邻间隔严格大于 `max_interval` 时抛出 `ValueError`，恰好等于限制被接受。样本生成器只被消费一次，输入不被修改，相同输入逐值一致。样本或向量不可迭代、记录或向量维度错误、测量值与偏置不是非布尔实数、`max_interval` 既非 `None` 也非实数时抛出 `TypeError`；样本不足两条、任一数值非有限、时间戳重复或逆序、`max_interval` 非正或非有限，或有限输入在累计中产生非有限结果时抛出 `ValueError`，且不返回部分结果。该模块仅依赖标准库与 `mappilot.geometry`，纯 Python 实现，沿用 Pose3 的右手坐标系约定，导入时不会启动 HTTP 服务。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

多传感器融合等能力尚未实现，留待后续任务从已冻结事实出发独立设计并验证。
