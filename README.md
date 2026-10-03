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

`mappilot.registration.point_to_plane_icp` 用点到面 ICP 估计同一刚体位姿，额外接收与目标点一一对应的法向量 `target_normals`。配对规则、初始位姿、迭代上限、收敛阈值与距离门限语义与点到点版本一致；不同之处在于每轮求解的是使变换后源点到对应目标切平面有符号投影误差平方和最小的六自由度切空间增量（左扰动，经 `Pose3.exp` 施加）。法向量在使用前按自身长度归一化，正比例缩放不改变结果。

```python
from mappilot.registration import point_to_plane_icp

result = point_to_plane_icp(
    source_points,          # 至少六个三维点
    target_points,          # 至少六个三维点
    target_normals,         # 每个目标点一个有限非零三维法向量
    initial_pose=None,
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=None,
)
result.pose            # 估计的 Pose3
result.converged       # 是否在迭代上限内收敛
result.iterations      # 实际完成的位姿更新次数
result.rmse            # 最终配对的有符号投影误差均方根
result.correspondences # (源点索引, 目标点索引)，按源点索引升序
```

`PointToPlaneICPResult` 同样不可变，`correspondences` 与 `rmse` 基于返回位姿重新计算。源点与目标点都至少包含六个点；法向量数量必须与目标点一致且每个法向量有限、非零；有效配对少于六个，或法向约束无法唯一确定六自由度位姿增量（如法向量全部平行的平面片）时抛出 `ValueError`；类型错误语义与点到点版本相同。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

位姿图优化、闭环检测等能力尚未实现，留待后续任务从已冻结事实出发独立设计并验证。
