# 停止距离与到达容差：离线诊断与修复

日期：2026-11-05。结论：**旧 0.20 m 停止 / 0.10 m 到达组合存在可复现的提前停止条件；最小配置修复和参数接线已完成本地离线验证。最终按用户指定使用 0.5 m 停止 / 1.0 m 到达。** 未启动 ROS、bridge 或实际运动，未更新车端镜像。

## 原因与最小对照

`localPlanner.cpp` 发布路径时按 `relativeGoalDis` 裁剪离散路径；`pathFollower.cpp` 在前视范围内推进跟踪点。短路径中跟踪点可成为路径末端。线速度分支要求 `dis > stopDisThre`，否则减速；Action 状态提供器则独立比较机器人到目标的距离与 `xy_goal_tolerance_m`。

因此，两者中间存在一个区域：follower 已要求停止，但 Action 尚不认为到达。原生加减速步进还可能产生微小速度振荡，不能断言所有案例都会严格零速。另一种解释是车本来没有运动授权；此前 bridge-off 实测确实属于这一情形，所以不能反过来将旧实测的 `no_progress` 归因于参数冲突。

最小辨别试验固定其他条件，只改变停止距离。测试从生产 C++ 文件抽取完整速度选择、转向限幅、减速及加速度步进代码，经 g++ 编译运行；不把控制公式重新翻译为 Python。将合成的理想直线运动反馈交给项目实际 `RosNavigationStatusProvider`，使用原有速度阈值 0.08 m/s、稳定时间 0.2 s、无进展 12 s 和进展阈值 0.05 m。

环境假设：100 Hz 理想运动积分、1 cm 末端离散、0.75 m 局部路径上限、前向零航向误差、无障碍、不含硬件延迟。路径几何为合成输入，不是完整 localPlanner 或 ROS/DDS 回放，也不模拟底盘动力学及 mux。

| 初始目标距离 | 旧停止 0.20 / 到达 0.10 | 诊断停止 0.05 / 到达 0.10 |
| --- | --- | --- |
| 0.15 m | 12.00 s blocked；余距 0.150 m | 3.00 s reached；余距 0.0971 m |
| 0.34 m | 26.14 s blocked；余距 0.1804 m | 8.55 s reached；余距 0.0970 m |
| 0.45 m | 17.70 s blocked；余距 0.1881 m | 10.47 s reached；余距 0.0970 m |

这组单因素对照确认了冲突机制。0.05 m 仅是诊断对照，不是最终交付默认值，也未部署到车上。

## 最终配置与验证

| 启动变量 | 默认值 | 原生/Action 参数 |
| --- | --- | --- |
| `NAV_FOLLOWER_STOP_DISTANCE_M` | 0.5 m | `stopDisThre` |
| `NAV_GOAL_TOLERANCE_M` | 1.0 m | runtime MotionGoal + MotionServer `xy_goal_tolerance_m` |
| `NAV_FOLLOWER_LOOKAHEAD_M` | 1.0 m | `lookAheadDis` |
| `NAV_FOLLOWER_SLOWDOWN_DISTANCE_M` | 1.0 m | `slowDwnDisThre` |

不修改原生控制算法，不放宽碰撞检测，不强制产生目标或 REACHED。调整前视距离是为了保持 `lookahead > stop`，防止跟踪点选择本身落在停止范围。主机与容器入口均检查有限正数及 `stop < tolerance`、`stop < lookahead`，错误配置在启动组件前退出。三项新增变量已进入 Docker 白名单和 lower launch 的 pathFollower 参数。

| 初始目标距离 | 最终 0.5 / 1.0 m 组合结果 | 状态判定时速度 |
| --- | --- | --- |
| 1.2 m | 6.82 s reached；余距 0.5242 m | 0.0765 m/s |
| 1.6 m | 10.40 s reached；余距 0.5241 m | 0.0765 m/s |
| 2.0 m | 13.98 s reached；余距 0.5241 m | 0.0765 m/s |

这里 reached 满足现有“速度 ≤ 0.08 m/s 并稳定 0.2 s”合同，不代表速度严格为零。正式 Action 终止时仍撤销原生目标并 hold；该终止行为由既有回归覆盖，真实制动距离未测。固定位置不动且目标在容差外的负对照仍为 blocked，没有伪造进展。

**距离定义保留原实现：** 到达容差以 standoff 接近点为参考，不是对象距离。standoff=0.8 m 不变；接近点已在 1.0 m 范围内时 resolver 不派发 Action。例如旧约 0.34 m 接近目标会被过滤。测试同时覆盖这一行为，以及容差外候选携带 1.0 m Action 容差。仿真源码存在 `success_distance=1.0`，也存在停止比例等其他条件；本次按用户指定设置数值，不宣称两种距离指标完全相同。

## 复现与交付边界

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/ubuntu/anaconda3/envs/conceptgraph/bin/python -m pytest \
  tests/test_navigation_stop_distance.py tests/test_real_navigation.py \
  tests/test_sysnav_ros_adapters.py tests/test_shadow_approach.py -q -s
bash scripts/run_real_robot_navigation.sh --check
```

69 项通过，包含原生停止机制对照、最终参数案例、非法数值/不兼容组合拒绝、主机/容器 check、launch 参数解析、Docker 白名单及既有到达/终止/隔离回归。原始输出保留于 `logs/diagnostics/navigation-stop-distance/pytest-requested-defaults.txt`，配置和源码哈希见同目录 `receipt.json`。

本次是本地源码修复，不是新的真机验收。旧镜像 `896fbd15b040...` 及既有 bridge-off 验收记录保持原样。后续需要把修改后的脚本、`real_robot/navigation_parameters.py` 和 lower launch 纳入新镜像，重建 ROS 安装目录，再验证实际参数加载。不要只在旧镜像外设置新增环境变量便认为修复生效。

参数语义及迁移细节见 [参数调节说明](real_robot_parameter_tuning.md)，启动见 [正式入口](real_robot_navigation_entry.md)。
