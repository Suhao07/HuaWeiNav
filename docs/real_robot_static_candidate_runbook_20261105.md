# 2026-11-05 静止候选运行与回滚

> 已升级到 [正式实物模式一键入口](real_robot_navigation_entry.md)，实物模式已按 bridge-off 规划与速度输出口径完成。调参见 [实物参数调节说明](real_robot_parameter_tuning.md)。本文保留作此前 shadow/静止验收记录。

> 历史阶段曾切换为 `huawei-vln-shadow-acceptance`，bridge 关闭的真实规划/速度验收通过。当时入口见 [shadow 部署说明](real_robot_shadow_deployment_20261105.md)，现以文首正式入口为准；下文保留作旧候选回滚说明。

此记录固定感知/定位/对应判定代码候选，不是运动或完整任务发布。

## 历史候选部署

- Orin 根目录：`/home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5`。
- 当前镜像：`huawei-vln-realworld:orin-r36.5-yoloworld-correspondence-candidate`。
  ID `sha256:3d97aec88b898338ed74a4e4620819433053e6852c22caae2ef9972d892b9679`。
- 父镜像 ID `sha256:016e990ad48182ca54cb6d8ab4f802a2bb5be505e9d8c316912a6725e3320745`。
  8 个文件差异及哈希在部署根 `correspondence-candidate-build/manifest.json`。
- 原源码包 SHA256 `71f340c5601b26b10e30db724f86f39e02ddcc67ecdbba78f4511b3be772ec41`；
  后续文件差异必须连同各 candidate-build manifest 使用，不能只用原源码包描述当前版本。
- 宿主 Docker wrapper SHA256 `0a30f4c17771507cb239ddd86db810c5dd4d8b35fc32d7e85860bf650aaae987`。
  它包含世界系/感知配置传递修复，不能回退到漏传版本。模型资产仍以 `asset_manifest.json` 为准。
- 容器 `huawei-vln-static-correspondence-candidate`，tmux `strive_static_correspondence`；
  日志 `logs/correspondence-static-launch.log`。LIO/相机会话独立运行。
- `STRIVE_DRY_RUN=true`、`ALLOW_REAL_MOTION=false`、lower/AgileX/waypoint adapter 关闭；
  `LLM_OFFLINE=1`、final verifier=false、`correspondence_package` 默认空。

保存地图仍为原部署 `realworld_eb188e7/output/lio_maps/tethered-20261105T1503/lio_map.pcd`，
SHA256 `e5223d558032b77ab7066227584dae53a5566c33249d62cfe16a220217e21e5f`，使用同目录 alignment。
仅在该保存地图定位已按实际车位恢复且点云/位姿核对通过时使用；移动后重启不能沿用旧初值。

## 静止启动、导出与回滚

确认没有另一套检测/运行时容器后，在 Orin 执行：

```bash
cd /home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5
bash correspondence-candidate-build/start_static.sh
```

脚本仍运行真实检测/融合及规则目标解析，控制开关断言必须保留。Docker wrapper 使用
`--rm`，停止前导出所需运行数据：

```bash
cd /home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5
mkdir -p logs/correspondence-static-export
docker cp huawei-vln-static-correspondence-candidate:/workspace/STRIVE/output/runtime/correspondence-static-20261105/. logs/correspondence-static-export/
docker stop -t 8 huawei-vln-static-correspondence-candidate
```

如需回到上一版静止服务，在新容器停止后使用：

```bash
bash evidence-candidate-build/start_static.sh
```

上一版保留图像/框/位姿配对修复，但没有新实例对应入口。切换前旧运行数据已归档到
`logs/pre-correspondence-switch-runtime`。不停止 LIO/相机，不改选源，不删除 PCD。

## 当前证据与边界

候选在 ARM 断网临时容器完成哈希/导入/安装路径/真实包离线检查及 19 项受控测试。
静止上线后 45 秒窗口收到 21 条带 ID 椅子检测、22 条融合对象消息；四个运动话题发布者为 0。
115 条决策快照有 84 次 go_to_object，目标 sysnav_object:2、视点 0，未发布 waypoint。
视点是当前观测位置；上述结果不证明目标路线可行或已经行驶。

源证据及结果见本地 `logs/diagnostics/correspondence-candidate-build/validation.json`。
当前精确绑定仍 matched=false。运行目录 `prior_map/correspondence_session.json` 给出当前
进程身份；重启后必须重新采集并准备与该会话一致的包，不能给旧包改 session 字段冒充新证据。

用户已授权并完成一次冻结证据的真实 Doubao 判定，结果为 fragments，未绑定；该次授权已使用。车端仍保持 LLM_OFFLINE=1。后续离线掩码核查确认历史混合和模型解释问题并存；SDK 修复与 masked-RGB 输入修复仅在本地，未进入上述车端镜像。详见 `archive/real_robot_correspondence_mask_diagnosis_20261105.md`。底盘语义、受限控制合同、低速及异常停车、真实行驶→新鲜视觉核验→STOP 仍待分别完成，自主运动仍未授权。
