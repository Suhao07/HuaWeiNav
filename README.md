
## Installation

### 1. Clone the repository

```bash
git clone git@github.com:igzat1no/STRIVE.git
cd STRIVE
```

### 2. Create a conda environment (Python 3.12)

```bash
conda create -n strive python=3.12 -y
conda activate strive
```

### 3. Install pip dependencies

```bash
pip install -r requirements.txt
```

### 4. Install habitat-sim and habitat-lab from source

We ship small patches and bug fixes on top of upstream Habitat. Install from our forks and check out the `v0.3.2` branch:

- [habitat-sim](https://github.com/zwandering/habitat-sim)
- [habitat-lab](https://github.com/zwandering/habitat-lab)

### 5. Install Segment Anything (SAM)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 6. Install MMDetection (GroundingDINO)

```bash
mim install mmengine
mim install mmcv==2.1.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

Download the GroundingDINO Swin-L checkpoint:

```bash
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

---

## Environment Variables

> ⚠️ **Do not commit real secrets to the repository.** Set them in your shell config (e.g. `~/.zshrc` or `~/.bashrc`).

```bash
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
export HABITAT_LAB_PATH="/path/to/habitat-lab/"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/path/to/mmdetection/"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/path/to/HM3D_v2/"
export MP3D_DATA_PATH="/path/to/MP3D/"
```

Reload your shell:

```bash
source ~/.zshrc
```

Verify that everything is set:

```bash
python -c "import os; keys=['GEMINI_API_KEY','HABITAT_LAB_PATH','SAM_CHECKPOINT','GROUNDING_DINO_PATH','GROUNDING_DINO_CHECKPOINT','HM3D_DATA_PATH','MP3D_DATA_PATH']; print({k: bool(os.getenv(k)) for k in keys})"
```

---

## Usage

Run the HM3D evaluation benchmark (default configuration):

```bash
python objnav_benchmark_with_process_obs.py
```

---

## Technical Report Figure Notes

The main technical report is maintained in
[`docs/project_technical_whitepaper.md`](docs/project_technical_whitepaper.md).
When drawing figures in draw.io, use the following placement plan so the
visuals match the report structure.

| Figure | Suggested Section | Main Content |
| --- | --- | --- |
| Teaser / overall pipeline | Document beginning and Section 3 | Task input, perception, Room-Viewpoint-Object graph, VLM reasoning, planning, final verification, Habitat action / real-robot waypoint |
| Multi-layer scene graph | Section 5 | Room nodes, viewpoint nodes, object nodes, graph edges, explored/frontier states, dynamic object-object relation edges |
| Instruction adapter and concept grounding | Section 4 | Raw instruction, `InstructionPlan`, terminal/anchor/support `ConceptQuery`, detector vocabulary, runtime concept matcher, ledgers |
| Final verifier and view-control loop | Section 6.6 | Candidate confirmation, context-aware verification, better-view subgoal, evidence capture, accept/retry/reject, geometry/VLM responsibility split |
| Real-robot SysNav deployment | Section 8 | ROS sensors, SysNav `detection_node`, `semantic_mapping_node`, STRIVE adapters, `SemanticMapSnapshot`, `NavigationIntent`, `/way_point`, local planner |

Recommended priority:

```text
1. Teaser / overall pipeline
2. Multi-layer scene graph
3. Final verifier and view-control loop
4. Real-robot SysNav deployment
5. Instruction adapter and concept grounding
```

Keep the figures concept-level rather than code-level. The strongest message is
the division of labor: VLM handles semantic reasoning and verification, while
mapping, reachability, distance, and motion execution remain geometry/controller
responsibilities.
