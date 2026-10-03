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

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

闭环检测等能力尚未实现，留待后续任务从已冻结事实出发独立设计并验证。
