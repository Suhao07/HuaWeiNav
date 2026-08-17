# VLN

VLN 是面向 HM3D ObjectNav 和实物机器人部署的语义导航框架。

- HM3D / Habitat 仿真评测：`objnav_benchmark_with_process_obs.py`
- SysNav 实物机器人适配与部署：`real_robot/`


## 目录

- [文档目录与产物边界](docs/README.md)
- [推荐环境](#推荐环境)
- [仿真器构建流程](#仿真器构建流程)
- [运行 HM3D 仿真评测](#运行-hm3d-仿真评测)
- [构建真实 HM3D 几何先验地图](#构建真实-hm3d-几何先验地图)
- [先验地图 A/B 对比测试](#先验地图-ab-对比测试)
- [本地源码安装备选方案](#本地源码安装备选方案)
- [环境变量](#环境变量)
- [输出结果](#输出结果)
- [实物机器人入口](#实物机器人入口)
- [技术报告配图建议](#技术报告配图建议)

## 推荐环境

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

## 构建真实 HM3D 几何先验地图

真实几何先验地图从 Habitat `semantic_scene + navmesh` 构建，读取房间、物体和拓扑，不读取 ObjectNav 回合中的目标位置，因此不会泄漏当前任务答案。

### FloorPlan-VLN-compatible room layout

HM3D v0.2 不包含 Matterport3D 的 `house_segmentations/*.house` 文件。若需要
FloorPlan-VLN 风格、便于迁移的 BEV/floorplan 格式，应使用 HM3D 的
`semantic.glb + semantic.txt + basis.navmesh` 构建 room-only layout：

```bash
COGNAV_ROOT=/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav \
IMAGE_TAG=strive-hm3d:local \
bash docker/run_hm3d_floorplan_layout.sh \
  /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
  --scene_id wcojb4TFT35 \
  --output logs/prior_maps/wcojb4TFT35_floorplan.json \
  --quality_output logs/prior_maps/wcojb4TFT35_floorplan_quality.json \
  --topdown_resolution 0.25
```

这个入口不检查 SAM/DINO，也不检查 ObjectNav episode，只挂载 CogNav 的
`data` 和当前 VLN workspace；因此 `conda activate CogNav` 不是必要条件，真正
运行环境由 `IMAGE_TAG` 指定的 Docker 镜像决定。

不要将 `docker/run_hm3d_baseline.sh` 用于布局构建。它是 ObjectNav benchmark
入口，会在启动容器前检查 SAM、GroundingDINO 和 episode 文件；布局入口只需要
Habitat-Sim 与 HM3D 的 basis/semantic/NavMesh 文件。

该输出使用 FloorPlan-VLN 同构的 `levels -> regions -> boundaries / center /
connectivity` 结构，但坐标仍保持米制：Habitat 的 `(x, z)` 平面转换为
floorplan 的 `(x, -z)` 平面。PNG 只是渲染视图，不能直接作为 motion goal。
若 Habitat-Sim 暴露的 semantic region AABB 无效，builder 会优先使用 semantic
mesh 中的 floor 几何按 parent region 聚合出保守区域包围盒，并在 quality metadata
中记录来源；若仍无法生成任何 room，CLI 会失败而不是写出空布局。

场景目录入口同时生成与实物模式对齐的语义 BEV bundle：

```text
logs/prior_maps/wcojb4TFT35_floorplan.json
logs/prior_maps/prior_map_bev.png
logs/prior_maps/prior_map_bev.svg
logs/prior_maps/prior_map_bev_markers.json
logs/prior_maps/prior_map_manifest.json
```

`prior_map_bev.png` 与 `floorplan.json` 同源，包含房间区域、拓扑和语义标签，
作为 LVLM 的地图上下文。正式先验地图不再包含 Habitat NavMesh 导出的 `.npy`；
NavMesh 仅在构建阶段帮助恢复房间几何，不进入 prompt、先验查询或实物接口。

`floorplan.json` 的 `frame_id` 为 `floorplan_metric`，并在 metadata 中保留
`source_frame_id=habitat_world`；这表示文件中的房间多边形已经采用 `(x,-z)` 平面，
不是 Habitat 原始三维坐标的直接副本。质量报告还会给出房间边界覆盖率、房间拓扑边数和
语义 BEV 产物信息。

同一 semantic region 内的 NavMesh 样本会先投影到二维 BEV 再提取连通组件，避免
不同高度采样造成重复房间。`component_N` 是几何组件标识，不是 HM3D 的自然语言
房间类别。

完整 ObjectNav 的检测权重由 `configs/strive_weights.yaml` 管理。当前配置复用本机
的 SAM ViT-H 和 GroundingDINO Swin-L 文件；权重只读挂载到容器，不进入仓库。迁移
机器时修改该 YAML，或通过 `SAM_CHECKPOINT`、`GROUNDING_DINO_CHECKPOINT` 覆盖。

layout-only 模式不写入静态 object instance；运行时目标身份仍来自 mapper 或
SysNav `ObjectNode`。因此该文件只影响 room/frontier 搜索先验，不拥有 STOP 权限。

批量构建同一数据 split：

```bash
bash docker/run_hm3d_baseline.sh bash -lc '
  cd /workspace/STRIVE
  python scripts/build_hm3d_floorplan_layouts.py \
    /workspace/data/scene_datasets/hm3d_v0.2/val \
    --output_root logs/prior_maps/hm3d_val_floorplans \
    --topdown_resolution 0.25
'
```

批处理会为每个场景写出 `floorplan.json`、语义 BEV、`quality.json` 和
`prior_map_manifest.json`，并在输出根目录生成批处理 `manifest.json`。质量报告中的
`object_instances_omitted=true` 是设计约束，不是构建失败。

### 1. 在容器内构建先验地图

示例场景：`wcojb4TFT35`

```bash
bash docker/run_hm3d_baseline.sh bash -lc '
cd /workspace/STRIVE
python scripts/build_hm3d_groundtruth_prior_map.py \
  /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
  --scene_id wcojb4TFT35 \
  --output logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json \
  --alignment_output logs/prior_maps/wcojb4TFT35_groundtruth_alignment.json \
  --topdown_resolution 0.25 \
  --min_room_area_m2 0.25 \
  --mask_dilation_radius_m 0.35
'
```

输出文件：

```text
logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json
logs/prior_maps/wcojb4TFT35_groundtruth_alignment.json
```

先验地图格式：

```text
source_format = hm3d_groundtruth_semantic_scene
frame_id      = habitat_world
authority     = semantic_scene_plus_navmesh
```

### 2. 验证先验地图内容

```bash
python - <<'PY'
import json
p = "logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json"
d = json.load(open(p))
print("scene:", d["scene_id"])
print("rooms:", len(d["rooms"]))
print("objects:", len(d["objects"]))
print("edges:", len(d["topology_edges"]))
print("tv:", [o for o in d["objects"] if o["label"].lower() == "tv"][:1])
PY
```

### 3. 可视化先验地图

启用先验地图的运行会在每个回合产物中自动保存：

```text
logs/<save_dir>/episode-*/prior_map/floorplan_global.png
logs/<save_dir>/episode-*/prior_map/floorplan_global.svg
logs/<save_dir>/episode-*/prior_map/floorplan_global_markers.json
logs/<save_dir>/episode-*/prior_map/floorplan_step_*.png
logs/<save_dir>/episode-*/prior_map/floorplan_chosen_frontier_*.png
logs/<save_dir>/episode-*/prior_map/som_global.png
logs/<save_dir>/episode-*/prior_map/search_prior_result_*.json
logs/<save_dir>/episode-*/prior_map/chosen_frontier_*.json
```

其中 `floorplan_chosen_frontier_*.png` 会叠加：

- 房间边界
- 房间到房间的拓扑边
- 先验物体点位
- 目标先验物体
- frontier 候选点
- 被选中的 frontier
- 运行轨迹
- 与先验物体匹配的在线检测结果

## 先验地图 A/B 对比测试

### 开启先验地图

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_prior_gt_ab_on \
  --vlm ark \
  --enable_prior_map \
  --prior_map_path logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json \
  --prior_map_source canonical_json \
  --prior_map_alignment logs/prior_maps/wcojb4TFT35_groundtruth_alignment.json
```

### 关闭先验地图

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_prior_gt_ab_off \
  --vlm ark
```

### 对比指标

```bash
python - <<'PY'
import csv
for run in ["hm3d_prior_gt_ab_on", "hm3d_prior_gt_ab_off"]:
    path = f"logs/{run}/metrics.csv"
    row = next(csv.DictReader(open(path)))
    print("\\n", run)
    for key in [
        "prior_map_enabled",
        "success",
        "spl",
        "distance_to_goal",
        "Episode Steps",
        "travel distance",
        "Found Goal",
        "final_stop_accept_step",
        "prior_map_top_object_label",
        "prior_map_top_object_score",
    ]:
        print(key, row.get(key, ""))
PY
```

先验地图只影响搜索排序、frontier 排序和房间排序，不直接生成运动目标，也不绕过最终验证器。最终停止权仍由现有物理停止约束、在线感知和最终验证器决定。

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

先验地图关键产物：

```text
prior_map/base_map.json
prior_map/alignment.json
prior_map/query_*.json
prior_map/search_prior_result_*.json
prior_map/runtime_state_*.json
prior_map/floorplan_global.png
prior_map/floorplan_step_*.png
prior_map/floorplan_chosen_frontier_*.png
prior_map/som_global.png
prior_map/dynamic_prior_map_bev_*.png
prior_map/dynamic_prior_map_bev_*_markers.json
```

启用高层 BEV 选择和在线房间语义标注：

```bash
python objnav_benchmark_with_process_obs.py \
  --enable_prior_map \
  --prior_map_path logs/prior_maps/<scene>_floorplan.json \
  --enable_prior_map_vlm \
  --enable_room_semantics \
  --prior_map_vlm_interval 10
```

动态 BEV 只提供 room/frontier 搜索上下文。它不改变目标确认、关系验证或
STOP authority；没有可读 RGB room evidence 时房间标签保持 `unknown`。

该高层接口使用统一的 `HighLevelCandidate` 表达房间和 frontier。候选 UID 必须
来自 mapper/SysNav snapshot，LVLM 只能在候选集合中排序，不能创建位姿、路径或
STOP。每步可复核产物包括 `room_semantics_<step>.json`、
`high_level_selection_<step>.json`、动态 BEV markers 和 LVLM raw trace；其中保存
证据哈希、候选 UID、prompt/model 版本、解析结果和请求耗时，不保存图像 base64。

未完成 prior/runtime 坐标标定时，动态 BEV 仍可作为静态语义上下文送入 LVLM，
但不会绘制未经变换的机器人、轨迹或 frontier 位置。

实物 ROS2 运行时如需复用 SysNav 的“RGB + room mask”房间语义输入，需显式打开
`enable_room_semantics:=true` 和 `persist_observation_images:=true`；调用间隔由
`room_semantic_interval` 控制。未持久化 RGB 时不会把 ROS URI 伪装成图像，也不会
发起无效的 LVLM 请求。

## 实物机器人入口

实物机器人部署详见 [docs/real_robot_deployment.md](docs/real_robot_deployment.md)，
平台无关 contract、数据/控制流与执行器模板见
[docs/real_robot_framework.md](docs/real_robot_framework.md)。独立 Qwen2.5-VL 推理服务器
及 VLN 客户端配置见 [docs/lvlm_server_deployment.md](docs/lvlm_server_deployment.md)。

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
