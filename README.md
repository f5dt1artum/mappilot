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

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

扫描匹配与位姿图优化等能力尚未实现，留待后续任务从已冻结事实出发独立设计并验证。
