# STRIVE HM3D Baseline 框架文档

本文档面向代码维护和二次开发，说明本项目 HM3D ObjectNav baseline 的主要模块、输入输出、数据流、核心模型和关键公式。

## 1. 总体目标

STRIVE HM3D baseline 解决的是 Habitat HM3D ObjectNav 任务：

```text
给定目标类别，例如 toilet，智能体在未知室内环境中通过 RGB-D 观测、目标检测、分割、三维建图、前沿探索和 LLM 决策，导航到目标附近并执行 stop。
```

当前 Docker 路径复用 CogNav_ObjNav：

- Habitat/HM3D 数据。
- CogNav LLMClient。
- 本地或 CogNav 权重目录中的视觉模型权重。

## 2. 主要入口

### 2.1 benchmark 入口

文件：

```text
objnav_benchmark_with_process_obs.py
```

职责：

- 解析命令行参数。
- 构建 Habitat config。
- 创建 Habitat 环境。
- 创建 mapper 和 agent。
- 逐 episode reset。
- 调用 LLM 获取目标相关类别。
- 执行导航循环。
- 保存视频、点云、metrics。

关键输入：

```text
--eval_episodes: 运行到哪个 episode index。
--start_episode: 从哪个 episode index 开始。
--save_dir: 保存到 logs/<save_dir>。
--vlm: LLM/VLM 后端，当前推荐 cognav。
```

关键输出：

```text
logs/<save_dir>/metrics.csv
logs/<save_dir>/episode-<i>/fps.mp4
logs/<save_dir>/episode-<i>/depth.mp4
logs/<save_dir>/episode-<i>/metrics.mp4
logs/<save_dir>/episode-<i>/pcd/
```

### 2.2 Docker 入口

文件：

```text
docker/run_hm3d_baseline.sh
```

职责：

- 查找或下载 SAM/GroundingDINO 权重。
- 校验 CogNav HM3D 数据。
- 挂载 STRIVE 仓库、CogNav 仓库、HuggingFace cache、权重文件。
- 透传 LLM、AMap、Habitat 和权重环境变量。
- 在容器内启动 `objnav_benchmark_with_process_obs.py`。

## 3. 模块结构

```text
docker/
  Dockerfile                         # 在 CogNav 基础镜像上补 STRIVE 依赖
  build.sh                           # 构建 strive-hm3d:local
  run_hm3d_baseline.sh               # 运行 benchmark
  preflight.py / preflight.sh        # 运行前检查

config_utils.py                      # Habitat/HM3D/MP3D config 适配
constants.py                         # LLM key 和模型名基础常量
objnav_benchmark_with_process_obs.py # benchmark 主循环
objnav_agent_with_process_obs.py     # 导航 agent、规划、轨迹保存
mapper_with_process_obs.py           # 语义建图、节点/房间/前沿管理
mapping_utils/transform.py           # 相机内参、坐标变换
cv_utils/
  gpt_utils.py                       # LLM 问答工具函数
  sam.py                             # GroundingDINO + SAM 目标感知封装
llm_utils/
  cognav_llm_adapter.py              # CogNav LLMClient 适配 OpenAI parse 接口
```

## 4. 端到端数据流

### 4.1 初始化阶段

```text
docker/run_hm3d_baseline.sh
  -> 挂载 /workspace/STRIVE
  -> 挂载 /workspace/CogNav_ObjNav
  -> 设置 HM3D_DATA_PATH、SAM_CHECKPOINT、GROUNDING_DINO_CHECKPOINT
  -> python objnav_benchmark_with_process_obs.py
```

### 4.2 每个 episode 的数据流

```text
Habitat episode
  -> env.reset()
  -> observation = {rgb, depth, semantic/goal info}
  -> HM3D_Objnav_Mapper 初始化地图
  -> HM3D_Objnav_Agent 初始化当前位置、轨迹、目标
```

### 4.3 每个导航周期的数据流

```text
RGB-D observation
  -> GroundingDINO 根据文本类别产生 2D boxes
  -> SAM 根据 boxes 产生 masks
  -> depth + camera intrinsic 反投影为局部 3D 点
  -> agent pose 将局部点变换到世界/地图坐标
  -> 将检测类别按 target_list 归一到任务目标，例如 tv_monitor -> tv
  -> mapper 更新点云、障碍、对象实例、节点、房间、前沿
  -> LLM 根据目标、房间、对象、距离选择候选房间/目标
  -> planner 计算下一动作
  -> env.step(action)
```

### 4.4 结束阶段

```text
found_goal 或 episode_over 或 step >= 500
  -> save_trajectory()
  -> 保存 rgb/depth/topdown mp4
  -> 保存点云 debug 文件
  -> 写入 metrics.csv
```

`found_goal` 不是“图像里出现了目标”这么简单。流程会先把 2D 检测结果变成 3D 对象，再用 `target_list` 做目标别名匹配，随后寻找可达停止点并做可视性检查。举例：任务目标是 `tv` 时，视觉模型可能输出 `tv_monitor`，需要归一成 `tv` 后才能触发目标确认。

## 5. 核心输入输出

### 5.1 Habitat observation

输入来源：

```text
habitat.Env.reset()
habitat.Env.step(action)
```

常用字段：

```text
obs["rgb"]   : H x W x 3, uint8
obs["depth"] : H x W x 1, float32
```

Agent 内部使用：

```text
self.obs
self.position
self.rotation
self.metrics
```

### 5.2 视觉感知输出

GroundingDINO 输入：

```text
image: RGB panorama 或当前视角
text prompts: target list / object classes
```

GroundingDINO 输出：

```text
boxes: N x 4, 每个框为 [x1, y1, x2, y2]
scores: N
labels: N
```

SAM 输入：

```text
image
boxes
```

SAM 输出：

```text
masks: N x H x W
```

### 5.3 Mapper 输出

Mapper 维护：

```text
节点 nodes: 探索候选点和已访问点
房间 rooms: 由节点和空间聚类得到的 room-level 表示
对象 objects: 由检测、分割和点云融合得到的对象实例
前沿 frontiers: 未探索区域边界
点云 pcd: 障碍、对象、可通行区域等三维信息
```

关键输出给 agent：

```text
candidate nodes
candidate rooms
object candidates
final waypoint
```

### 5.4 LLM 输出

通过 `llm_utils/cognav_llm_adapter.py` 适配后，LLM 输出被解析为 Pydantic 结构体。

典型输出：

```text
similar object list
object_found flag
candidate room id
reason / explanation
```

若启用：

```bash
STRIVE_LLM_FALLBACK=1
```

当 LLM 返回空内容或非 JSON 时，会返回保守默认结构，只用于 smoke 测试。

## 6. 核心几何和模型公式

### 6.1 相机内参

给定图像宽高 `W, H` 和水平视场角 `hfov`：

```text
fx = W / (2 * tan(hfov / 2))
fy = fx
cx = W / 2
cy = H / 2
```

相机内参矩阵：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

代码位置：

```text
mapping_utils/transform.py::habitat_camera_intrinsic
```

### 6.2 RGB-D 反投影

对像素点 `(u, v)` 和深度 `z`：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = z
```

齐次形式：

```text
p_camera = z * K^-1 * [u, v, 1]^T
```

### 6.3 相机坐标到世界坐标

给定 agent 位姿旋转 `R` 和平移 `t`：

```text
p_world = R * p_camera + t
```

对于点云集合：

```text
P_world = R * P_camera + t
```

这些点进入 mapper 后被聚合到对象点云、障碍点云和可通行区域中。

### 6.4 节点距离和路径选择

节点间欧氏距离：

```text
d(p_i, p_j) = ||p_i - p_j||_2
```

候选前沿/房间的代价通常由以下因素共同决定：

```text
cost = travel_distance + semantic_penalty - target_likelihood_bonus
```

当前代码中该代价不是单一显式函数，而是由 mapper 计算候选、agent 计算路径距离、LLM 综合语义和距离后选择。

### 6.5 成功判定

Habitat ObjectNav 的成功通常由 stop 动作和目标距离共同决定：

```text
success = 1 if action == STOP and distance_to_goal <= success_distance else 0
```

当前配置中：

```text
success_distance = 1.0
max_episode_steps = 500
```

## 7. LLM 决策接口

STRIVE 原始代码使用 OpenAI 风格：

```python
client.beta.chat.completions.parse(...)
```

本项目通过适配器把 CogNav LLMClient 包装成相同接口：

```text
llm_utils/cognav_llm_adapter.py
```

适配流程：

```text
Pydantic response_format
  -> 生成 JSON schema prompt
  -> 调用 CogNav LLMClient.chat_completion()
  -> 提取 JSON object
  -> response_format.model_validate()
  -> 返回 OpenAI-compatible parsed response
```

这样 `cv_utils/gpt_utils.py`、`mapper_with_process_obs.py` 等调用方无需直接依赖 CogNav LLMClient 的具体实现。

## 8. 规划和动作

动作来自 Habitat ObjectNav action space，常见为：

```text
0: STOP
1: MOVE_FORWARD
2: TURN_LEFT
3: TURN_RIGHT
```

Agent 的规划逻辑：

```text
当前位姿 + mapper 目标 waypoint
  -> pathfinder / PID planner
  -> next action
  -> env.step(action)
```

核心代码：

```text
objnav_agent_with_process_obs.py::make_plan_mod_no_relocate
objnav_agent_with_process_obs.py::step_mod
```

## 9. 日志和可视化

每个 episode 输出：

```text
fps.mp4      # RGB 第一视角
depth.mp4    # 深度图
metrics.mp4  # top-down map + 指标
```

`save_trajectory()` 会把不同阶段产生的 frame 统一 resize 到首帧尺寸，避免 `imageio` 写 mp4 时因为尺寸不同失败。

## 10. 可替换模块

### 10.1 替换 LLM

入口：

```text
llm_utils/cognav_llm_adapter.py::get_client_and_model
cv_utils/gpt_utils.py
```

推荐保持返回接口兼容：

```python
client.beta.chat.completions.parse(...)
```

### 10.2 替换检测/分割模型

入口：

```text
cv_utils/sam.py
config_utils.py 中的 GROUNDING_DINO_* 和 SAM_CHECKPOINT
```

替换模型时需要保证输出仍能转换为：

```text
boxes
masks
labels
scores
```

### 10.3 替换数据集

入口：

```text
config_utils.py::hm3d_config
config_utils.py::mp3d_config
```

需要对齐：

```text
scene_datasets
episode json.gz
success_distance
sensor config
measurements
```

## 11. 当前已验证路径

已验证：

```text
Docker image: strive-hm3d:local
Data root: CogNav_ObjNav/data
LLM client: CogNav LLMClient + Ark provider
Vision: MMDetection GroundingDINO Swin-L + SAM ViT-H
Benchmark: HM3D ObjectNav val episode 0
```

真实 LLM 测试结果：

```text
success: 1.0
Found Goal: True
Episode Steps: 128
输出目录: logs/hm3d_cognav_real_llm_smoke/
```
