# STRIVE HM3D Docker Benchmark 指令文档

本文档记录从零创建本地 Docker 镜像、复用 CogNav_ObjNav 数据和 LLM client、配置权重、运行 HM3D ObjectNav baseline benchmark 的完整流程。

## 1. 前置条件

默认项目路径：

```bash
cd "/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
```

依赖的 CogNav 仓库路径：

```bash
export COGNAV_ROOT=/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav
```

本方案复用 CogNav 仓库中的：

- Habitat/HM3D 数据：`$COGNAV_ROOT/data`
- CogNav LLM client：`$COGNAV_ROOT/utils/llm_client.py`
- 权重目录：`$COGNAV_ROOT/model/pretrained_model`

本机需要已经可用：

- NVIDIA GPU 和 Docker GPU runtime
- 基础镜像 `cognav-vln:1.0`
- HM3D 场景数据：`$COGNAV_ROOT/data/scene_datasets/hm3d_v0.2`
- ObjectNav episode 数据：`$COGNAV_ROOT/data/objectgoal_hm3d/val/val.json.gz` 或 `$COGNAV_ROOT/data/objectnav_hm3d_v2/val/val.json.gz`

## 2. 从零构建 STRIVE 镜像

默认构建镜像名为 `strive-hm3d:local`：

```bash
bash docker/build.sh
```

如果需要指定基础镜像或输出镜像名：

```bash
COGNAV_BASE_IMAGE=cognav-vln:1.0 IMAGE_TAG=strive-hm3d:local bash docker/build.sh
```

Dockerfile 会在 CogNav 基础环境上补齐 STRIVE HM3D baseline 需要的依赖：

- `mmengine`
- `mmcv`
- `mmdet`
- `segment-anything`
- `pathfinding`
- `bresenham`
- `supervision`

构建完成后检查镜像：

```bash
docker images strive-hm3d:local
```

## 3. 权重配置

启动脚本会优先搜索本机权重，搜索不到时可按需下载。

### 3.1 SAM

默认搜索：

```text
$COGNAV_ROOT/model/pretrained_model/sam_vit_h_4b8939.pth
/home/ubuntu/WorkSpace/research/code/CoRL2025/SG-Nav/segment_anything/sam_vit_h_4b8939.pth
/home/ubuntu/WorkSpace/research/code/CoRL2025/AKGVP/data/models/sam_vit_h_4b8939.pth
```

手动指定：

```bash
SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth bash docker/run_hm3d_baseline.sh
```

### 3.2 GroundingDINO

STRIVE 当前 mmdet 配置使用 Swin-L 权重：

```text
grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

默认搜索：

```text
$COGNAV_ROOT/model/pretrained_model/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
./grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

如果本机没有，可以允许脚本下载到 CogNav 权重目录：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

或：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/run_hm3d_baseline.sh
```

手动指定：

```bash
GROUNDING_DINO_CHECKPOINT=/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth \
  bash docker/run_hm3d_baseline.sh
```

## 4. Preflight 检查

运行：

```bash
bash docker/preflight.sh
```

检查内容包括：

- CogNav 数据路径是否存在
- SAM 和 GroundingDINO 权重是否存在
- Habitat/Habitat-Sim 是否可 import
- PyTorch CUDA 是否可用
- mmdet DetInferencer 是否可用
- SAM builder 是否可用
- CogNav LLMClient 是否可 import
- HM3D config 和相机内参是否能生成

成功时会输出：

```text
preflight OK
```

## 5. 运行 HM3D Benchmark

### 5.1 离线 smoke 测试

不调用真实 LLM，只验证 Habitat、检测、分割、建图、规划、保存结果的主链路：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 \
  bash docker/run_hm3d_baseline.sh \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir hm3d_cognav_offline_smoke \
  --vlm cognav
```

说明：

- `LLM_OFFLINE=1`：CogNav LLMClient 走离线模式。
- `STRIVE_LLM_FALLBACK=1`：LLM JSON 为空或不可解析时，使用保守结构化默认值继续跑通管线。
- 该模式只适合工程 smoke，不代表真实策略效果。

### 5.2 真实 Ark LLM 测试

不要把 key 写入仓库文件，建议只在当前 shell 中导出：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY="<your-ark-api-key>"
export LLM_MODEL=doubao-seed-2-0-lite-260428
export LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

如果后续流程需要地图服务，也可传入：

```bash
export MAP_PROVIDER=Amap
export AMAP_KEY="<your-amap-key>"
```

运行 1 个 episode：

```bash
bash docker/run_hm3d_baseline.sh \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir hm3d_cognav_real_llm_smoke \
  --vlm cognav
```

运行多个 episode：

```bash
bash docker/run_hm3d_baseline.sh \
  --eval_episodes 10 \
  --start_episode 0 \
  --save_dir hm3d_cognav_real_llm_eval10 \
  --vlm cognav
```

从指定 episode 继续：

```bash
bash docker/run_hm3d_baseline.sh \
  --eval_episodes 20 \
  --start_episode 10 \
  --save_dir hm3d_cognav_real_llm_eval20 \
  --vlm cognav
```

## 6. 输出结果

输出目录在：

```text
logs/<save_dir>/
```

例如：

```text
logs/hm3d_cognav_real_llm_smoke/
```

关键文件：

```text
logs/<save_dir>/metrics.csv
logs/<save_dir>/episode-0/fps.mp4
logs/<save_dir>/episode-0/depth.mp4
logs/<save_dir>/episode-0/metrics.mp4
```

说明：

- `metrics.csv`：每个 episode 的成功率、SPL、距离目标、步数等指标。
- `fps.mp4`：第一视角 RGB 过程，可视化主结果。
- `depth.mp4`：深度图过程。
- `metrics.mp4`：top-down map 和导航指标过程。

VS Code 中可以在 Explorer 里直接点击 mp4 预览；如果不能播放，右键选择 `Open With...` 再选择内置视频预览器。

## 7. 进入容器调试

进入容器 shell：

```bash
bash docker/run_hm3d_baseline.sh bash
```

进入后常用检查：

```bash
cd /workspace/STRIVE
python -m py_compile objnav_benchmark_with_process_obs.py
python docker/preflight.py
```

## 8. 常见问题

### 8.1 Docker socket permission denied

说明当前用户或沙箱不能访问 Docker daemon。需要在允许 Docker 的环境中运行，或使用具有 Docker 权限的终端。

### 8.2 缺少 GroundingDINO 权重

运行：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

或者显式设置：

```bash
export GROUNDING_DINO_CHECKPOINT=/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

### 8.3 LLM 401

检查：

```bash
echo "$ARK_API_KEY"
echo "$LLM_MODEL"
echo "$LLM_API_BASE_URL"
```

确保 `ARK_API_KEY` 是真实 key，且 `LLM_MODEL` 和 base URL 对应当前 Ark 服务。

### 8.4 AMap key 未进入容器

脚本已透传：

```bash
MAP_PROVIDER
AMAP_KEY
```

运行前在宿主机 shell 中导出即可。

### 8.5 mp4 保存失败，提示帧尺寸不一致

代码已在 `objnav_agent_with_process_obs.py::save_trajectory` 中统一 resize 三路视频帧尺寸。若未来改动 top-down map 生成逻辑，仍需保证同一个视频 writer 接收的 frame 尺寸一致。
