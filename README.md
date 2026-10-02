# MapPilot

这是一个面向SLAM 定位与建图的SLAM 定位与建图库。长期目标是提供位姿表示与李群运算、扫描匹配、因子图后端、闭环检测与位姿图优化、多传感器融合、重定位和地图表示，把定位建图沉淀为可复用库。

当前基线提供进程健康检查，并新增了位姿数学的第一块带测试覆盖的能力：`mappilot.geometry.Pose3`。后续能力（扫描匹配、位姿图优化等）必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 位姿表示

`Pose3` 是右手坐标系下的 SE(3) 刚体位姿，把局部坐标变换到父坐标：`p_parent = R @ p_local + t`。平移为 `(x, y, z)`，四元数为 `(w, x, y, z)`，无需启动 HTTP 服务即可使用：

```python
from mappilot.geometry import Pose3  # 也可直接 from mappilot import Pose3

pose = Pose3((1.0, 2.0, 3.0), (0.0, 1.0, 0.0, 0.0))   # 平移 + 四元数
pose = Pose3.from_matrix(matrix44)                     # 4x4 齐次矩阵
pose.quaternion      # 规范化、w 为正的 (w, x, y, z)（w 为零时首个非零分量为正）
pose.to_matrix()     # 4x4 齐次矩阵（不可变元组）

pose.compose(other)  # 先应用 other 再应用 pose；pose.inverse() 为其逆
pose.transform((x, y, z))
pose.transform_points([(x, y, z), ...])               # 空序列返回空元组
Pose3.exp((wx, wy, wz, vx, vy, vz))                   # 左扰动约定
pose.log()                                            # 与 exp 互为逆映射
```

校验规则：标量必须是实数（拒绝布尔值），维度/形状/元素类型不符抛 `TypeError`；NaN/无穷、零长度四元数、末行不是 `(0, 0, 0, 1)`、旋转块非正交（绝对误差 1e-9）或行列式非正抛 `ValueError`。合法的非单位四元数会自动归一化，反射矩阵不会被静默修正。

## 启动

```bash
PYTHONPATH=src python3 -m mappilot.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `MAPPILOT_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

除健康检查外，当前基线实现了 `mappilot.geometry.Pose3` 的位姿表示与 SE(3) 运算；扫描匹配与位姿图优化等能力留待后续任务，从已冻结事实出发独立设计并验证。
