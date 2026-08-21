# VLN

VLN 是面向 HM3D ObjectNav 和实物机器人部署的语义导航框架。

- HM3D / Habitat 仿真评测：`objnav_benchmark_with_process_obs.py`
- SysNav 实物机器人适配与部署：`real_robot/`


## 目录

- [文档目录与产物边界](docs/README.md)
- [LVLM 接入与部署基础](docs/lvlm_server_deployment.md)
- [HM3D 语义先验地图与对比评测](docs/hm3d_prior_map_evaluation.md)
- [推荐环境](#推荐环境)
- [仿真器构建流程](#仿真器构建流程)
- [运行 HM3D 仿真评测](#运行-hm3d-仿真评测)
- [本地源码安装备选方案](#本地源码安装备选方案)
- [环境变量](#环境变量)
- [输出结果](#输出结果)
- [实物机器人入口](#实物机器人入口)

## 推荐环境

运行仿真或实物导航前，先完成 LVLM 接入。项目支持直接调用商业 API，也支持将模型
部署在独立 GPU 服务器后由机器人通过 OpenAI-compatible HTTP API 访问；配置、验收和
公网 HTTPS 反向代理要求见 [LVLM 接入与部署基础](docs/lvlm_server_deployment.md)。

推荐使用 Docker 跑仿真，原因是 Habitat-Sim、Habitat-Lab、MMDetection、CUDA、PyTorch 和 HM3D 数据路径之间的版本耦合较强。

默认项目路径：

```bash
cd "/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
```

本仓库通过 `configs/strive_weights.yaml` 记录本机数据和权重路径；启动脚本会优先
使用 `STRIVE_DATA_ROOT`/`HM3D_DATA_ROOT` 环境变量，再回退到 YAML 中的 `data_root`。
当前工作区默认复用 CogNav_ObjNav 的数据目录。迁移到其他机器时修改 YAML，或显式
覆盖数据根目录：

```bash
export STRIVE_ROOT="/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
export STRIVE_MODELS_DIR="$STRIVE_ROOT/models"

export STRIVE_DATA_ROOT="/path/to/CogNav_ObjNav/data"  # 可选，优先级高于 YAML
mkdir -p "$STRIVE_MODELS_DIR"
```

当前本机配置为：

```text
configs/strive_weights.yaml:data_root
-> /home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data
```

需要提前准备：

- NVIDIA GPU 和 Docker GPU runtime。
- 一个已经可导入 Habitat-Sim / Habitat-Lab 的基础镜像，例如 `habitat-hm3d:local`。
- HM3D 场景数据：`$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`，其中
  `$STRIVE_DATA_ROOT` 为环境变量或 YAML 中的 `data_root`。
- HM3D ObjectNav 回合数据：对应数据根目录下的
  `objectnav_hm3d_v2/val/val.json.gz` 或 `objectgoal_hm3d/val/val.json.gz`。
- SAM 权重：`$STRIVE_MODELS_DIR/sam_vit_h_4b8939.pth`。
- GroundingDINO Swin-L 权重：`$STRIVE_MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth`。

## 仿真器构建流程

### 1. 准备 HM3D 场景数据

HM3D v0.2 场景数据使用 Matterport 官方公开 tar 链接下载，不需要 MP3D 的账号密码流程。官方入口：

- `https://github.com/matterport/habitat-matterport-3dresearch`

VLN 需要每个 split 的四类文件：

```text
hm3d-<split>-glb-v0.2.tar
hm3d-<split>-habitat-v0.2.tar
hm3d-<split>-semantic-annots-v0.2.tar
hm3d-<split>-semantic-configs-v0.2.tar
```

从 0 部署时直接运行仓库脚本。默认下载 `val` 和 `minival`，tar 包保存在
`$STRIVE_DATA_ROOT/downloads/hm3d_v0.2`，解压后的标准目录是
`$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`：

```bash
chmod +x scripts/download_hm3d_v02.sh
HM3D_SPLITS="val minival" bash scripts/download_hm3d_v02.sh
```

如果要补齐 train split：

```bash
HM3D_SPLITS="train" bash scripts/download_hm3d_v02.sh
```

也可以手动下载。以 `val` split 为例：

```bash
mkdir -p "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"
cd "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"

curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar
```

完成后应看到下面的目录结构：

```text
$STRIVE_DATA_ROOT/
  scene_datasets/
    hm3d_v0.2/
      hm3d_annotated_basis.scene_dataset_config.json
      val/
        00802-wcojb4TFT35/
          wcojb4TFT35.basis.glb
          wcojb4TFT35.semantic.glb
          wcojb4TFT35.basis.navmesh
      train/
      minival/
```

如果实验室内网已经提供了下载好的 tar 包，也可以直接交给脚本处理；要求最终仍然整理成同样的 `$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`：

```bash
mkdir -p "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"
cp /path/to/hm3d-*-v0.2.tar "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2/"
HM3D_SPLITS="" bash scripts/download_hm3d_v02.sh
```

校验场景数据：

```bash
test -f "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.basis.glb" | head
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.basis.navmesh" | head
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.semantic.glb" | head
```

### 2. 准备 HM3D ObjectNav 回合数据

ObjectNav 回合数据只保存 episode、scene id、起点和目标类别，不保存 VLN 运行时的在线地图。推荐目录：

```text
$STRIVE_DATA_ROOT/
  objectnav_hm3d_v2/
    val/
      val.json.gz
      content/
        wcojb4TFT35.json.gz
        ...
```

如果你的数据版本使用旧命名，也可以放在：

```text
$STRIVE_DATA_ROOT/
  objectgoal_hm3d/
    val/
      val.json.gz
      content/
```

整理示例：

```bash
mkdir -p "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val"
cp /path/to/objectnav_hm3d_val/val.json.gz "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz"
cp -r /path/to/objectnav_hm3d_val/content "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/content"
```

校验回合数据：

```bash
test -f "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz" || \
test -f "$STRIVE_DATA_ROOT/objectgoal_hm3d/val/val.json.gz"
```

### 3. 准备视觉模型权重

权重默认放在 `$STRIVE_MODELS_DIR`：

```text
$STRIVE_MODELS_DIR/
  sam_vit_h_4b8939.pth
  grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

如果本机还没有权重，可以手动下载到 `$STRIVE_MODELS_DIR`：

```bash
wget -O "$STRIVE_MODELS_DIR/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -O "$STRIVE_MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

如果权重放在其它位置，也可以显式设置路径：

```bash
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
```

镜像和数据也准备好之后，还可以让运行脚本自动补齐缺失权重：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

### 4. 构建 HM3D 仿真 Docker 镜像

HM3D 仿真链路是 Habitat / Python / CUDA 环境，不是 ROS2 运行环境。它分成两层镜像：

```text
habitat-hm3d:local   基础镜像：CUDA、Conda、PyTorch、Habitat-Sim、Habitat-Lab
strive-hm3d:local    VLN overlay：MMDetection、SAM、pathfinding 和本项目运行依赖
```

先从 0 构建 Habitat 基础镜像：

```bash
chmod +x docker/build_habitat_base.sh
IMAGE_TAG=habitat-hm3d:local \
bash docker/build_habitat_base.sh
```

默认基础镜像配置：

```text
CUDA base image  = nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
Conda env        = strive
Python           = 3.9
PyTorch          = 2.0.1 + CUDA 11.8
Habitat-Sim      = 0.3.2
Habitat-Lab ref  = v0.3.2
```

如果需要固定 Habitat-Lab 的源码来源或分支，可以显式覆盖：

```bash
HABITAT_LAB_REPO=https://github.com/facebookresearch/habitat-lab.git \
HABITAT_LAB_REF=v0.3.2 \
IMAGE_TAG=habitat-hm3d:local \
bash docker/build_habitat_base.sh
```

基础镜像构建完成后，校验 Habitat 组件：

```bash
docker run --rm --gpus all habitat-hm3d:local python - <<'PY'
import habitat
import habitat_sim
import torch
print("habitat:", getattr(habitat, "__version__", "unknown"))
print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
```

然后构建 VLN overlay 镜像，默认输出镜像名为 `strive-hm3d:local`：

```bash
BASE_IMAGE=habitat-hm3d:local \
bash docker/build.sh
```

VLN overlay 镜像会在基础 Habitat 环境上补齐：

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

### 5. 构建实物 ROS2 Humble Docker 镜像

实物机器人链路使用 ROS2 Humble，入口是 `docker/Dockerfile.real_robot` 和 `docker/build_real_robot.sh`。这条链路与 HM3D Habitat 仿真镜像分离，不使用 ROS1 Noetic。

默认输出镜像：

```text
huawei-nav-real:orin
```

构建命令：

```bash
IMAGE_TAG=huawei-nav-real:orin \
INSTALL_ML_DEPS=0 \
INSTALL_LLM_DEPS=1 \
bash docker/build_real_robot.sh
```

如果需要把 detector/mapping 的重依赖也装进镜像，可以显式打开：

```bash
IMAGE_TAG=huawei-nav-real:orin \
INSTALL_ML_DEPS=1 \
INSTALL_LLM_DEPS=1 \
bash docker/build_real_robot.sh
```

### 6. 检查权重和数据挂载

运行 preflight；如果本地缺少 SAM / GroundingDINO 权重，可加 `STRIVE_DOWNLOAD_WEIGHTS=1` 自动下载：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

检查内容包括：

- HM3D 场景和 ObjectNav 回合数据是否存在。
- SAM / GroundingDINO 权重是否存在。
- Habitat / Habitat-Sim 是否可导入。
- PyTorch CUDA 是否可用。
- MMDetection `DetInferencer` 是否可初始化。
- SAM builder 是否可用。
- HM3D config 和相机内参是否能生成。

成功时会输出：

```text
preflight OK
```

### 7. 进入仿真容器调试

```bash
bash docker/run_hm3d_baseline.sh bash
```

进入容器后：

```bash
cd /workspace/STRIVE
python docker/preflight.py
```

容器内关键路径：

```text
/workspace/STRIVE          当前仓库，可写
/workspace/data            HM3D 场景和 ObjectNav 回合数据，只读挂载
/weights                   SAM 和 GroundingDINO 权重
```

## 运行 HM3D 仿真评测

### 离线 smoke 测试

不调用真实 LLM，只验证 Habitat、检测、分割、建图、规划和产物保存链路：

```bash
LLM_OFFLINE=1 STRIVE_LLM_FALLBACK=1 \
bash docker/run_hm3d_baseline.sh \
  --eval_episodes 1 \
  --start_episode 0 \
  --save_dir hm3d_strive_offline_smoke \
  --vlm ark
```

### 指定场景和目标物

推荐使用单场景入口：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_wcojb4TFT35_tv \
  --vlm ark
```

选择同一场景、同一目标物下的第 N 个回合：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 1 \
  --save_dir hm3d_wcojb4TFT35_tv_rank1 \
  --vlm ark
```

快速调试时可限制步数：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_debug_max_steps \
  --vlm ark \
  --max_steps 20
```

### 启用自然语言指令适配器

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_instruction_adapter \
  --vlm ark \
  --enable_instruction_adapter \
  --custom_instruction "find the television in the scene"
```

如果只想使用规则解析，不调用 LLM：

```bash
--instruction_adapter_backend rules
```

HM3D 语义先验地图的生成、批量构建、运行接入和有/无先验对比评测见
[HM3D 语义先验地图与对比评测](docs/hm3d_prior_map_evaluation.md)。

## 本地源码安装备选方案

如果不使用 Docker，可以本地安装。但这条路径更容易受 CUDA、Habitat、MMDetection 版本影响，推荐只用于开发调试。

### 1. 创建 Conda 环境

```bash
conda create -n strive python=3.12 -y
conda activate strive
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 Habitat-Sim 和 Habitat-Lab

本项目在上游 Habitat 上有少量补丁和兼容修正，建议使用以下 fork，并切到 `v0.3.2` 分支：

- `https://github.com/zwandering/habitat-sim`
- `https://github.com/zwandering/habitat-lab`

安装完成后设置：

```bash
export HABITAT_LAB_PATH="/path/to/habitat-lab"
```

### 4. 安装 Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 5. 安装 MMDetection / GroundingDINO

```bash
mim install mmengine
mim install mmcv==2.1.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

下载 GroundingDINO Swin-L 权重：

```bash
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

## 环境变量

不要把真实 key 写进仓库文件。建议只在 shell 中导出，或写入本机私有配置。

```bash
export HABITAT_LAB_PATH="/path/to/habitat-lab"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/path/to/mmdetection"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/path/to/HM3D_v2"
export MP3D_DATA_PATH="/path/to/MP3D"
```

真实 LLM 测试示例：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY="<你的-ark-api-key>"
export LLM_MODEL=doubao-seed-2-0-lite-260428
export LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```


检查关键变量：

```bash
python - <<'PY'
import os
keys = [
    "HABITAT_LAB_PATH",
    "SAM_CHECKPOINT",
    "GROUNDING_DINO_PATH",
    "GROUNDING_DINO_CHECKPOINT",
    "HM3D_DATA_PATH",
    "MP3D_DATA_PATH",
]
print({k: bool(os.getenv(k)) for k in keys})
PY
```

## 输出结果

仿真输出默认保存在：

```text
logs/<save_dir>/
```

常用文件：

```text
metrics.csv
run_manifest.json
episode-*/rgb/
episode-*/frontier_map/
episode-*/instruction_adapter/
episode-*/final_verifier/
episode-*/prior_map/
```

先验地图运行产物、动态 BEV、room/frontier 语义选择和 A/B 对比指标见独立文档。

## 实物机器人入口

实物机器人模式的输入输出、数据流、控制流和可插拔接口见
[docs/real_robot_framework.md](docs/real_robot_framework.md)。独立 LVLM 推理服务器
及 VLN 客户端配置见 [docs/lvlm_server_deployment.md](docs/lvlm_server_deployment.md)。
当前完成边界和现场执行顺序见
[docs/real_robot_deployment_todo.md](docs/real_robot_deployment_todo.md)。

常用入口：

```bash
SUDO_STDIN_PASSWORD=1 ./docker_en.sh start
SUDO_STDIN_PASSWORD=1 ./docker_en.sh enter
SUDO_STDIN_PASSWORD=1 ./docker_en.sh smoke
```

当前实物链路原则：

```text
VLN 不直接发布 /cmd_vel。
VLN 对底层控制器的正常输出是 MotionGoal -> RosWaypointController -> /way_point。
dry_run=true 或 lower_controller_enabled=false 时，不向真实 /way_point 交接。
FinalInstructionVerifier 只在 NavigationStatus.REACHED 后消费 ViewEvidence。
```
