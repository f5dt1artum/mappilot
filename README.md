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

`mappilot.registration.point_to_point_icp` 用点到点 ICP 估计把源点云对齐到目标点云的 `Pose3`。每轮先用当前位姿变换源点，再为每个源点选择最近目标点（目标点可被重复选择，等距时取索引最小者），可选地丢弃距离超过 `max_correspondence_distance` 的配对，最后以 Horn 闭式单位四元数解更新使平方距离和最小的刚体位姿（旋转行列式恒为 +1，不含镜像）。相邻两轮 RMSE 绝对差不超过 `tolerance` 时收敛；用尽 `max_iterations` 则返回最后结果且 `converged` 为 `False`，不抛出异常。

```python
from mappilot.registration import point_to_point_icp

result = point_to_point_icp(
    source_points,          # 至少三个三维点
    target_points,          # 至少三个三维点
    initial_pose=None,      # 默认单位位姿
    max_iterations=50,      # 正整数
    tolerance=1e-6,         # 有限正数
    max_correspondence_distance=None,  # 给出时须为有限正数
)
result.pose            # 估计的 Pose3
result.converged       # 是否在预算内收敛
result.iterations      # 实际完成的位姿更新次数
result.rmse            # 最终配对的欧氏距离均方根（按最终位姿重算）
result.correspondences # (源点索引, 目标点索引)，按源点索引升序
```

返回的 `ICPResult` 为不可变命名元组。非可迭代点云、点维度错误、坐标非实数或为布尔值、`initial_pose` 不是 `Pose3`、参数类型错误抛出 `TypeError`；坐标含 NaN/无穷、点数不足、距离门限后有效配对少于三个、配对共线等无法唯一约束三维刚体位姿，或数值参数范围不合法时抛出 `ValueError`。该模块仅依赖标准库，导入时不启动 HTTP 服务。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

位姿图优化、闭环检测等能力尚未实现，留待后续任务从已冻结事实出发独立设计并验证。
