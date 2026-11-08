# LVLM 接入与部署基础

LVLM 是 VLN 仿真和实物导航的前置依赖。instruction parser、concept grounding、
runtime concept matcher、relation verifier 和 final verifier 都通过统一的
OpenAI-compatible 调用合同访问多模态模型。

本文定义两种部署方式：

| 方式 | 模型位置 | 机器人侧依赖 | 适用场景 |
|---|---|---|---|
| 商业 API | 云端托管 | API key、endpoint、模型名 | 快速验证、无 GPU 设备、原型部署 |
| 自部署模型 | 独立 GPU 服务器 | 公网 HTTPS endpoint、API key | 可控延迟、数据边界、批量实物测试 |

无论使用哪种方式，VLN 上层只依赖统一的 `client.beta.chat.completions.parse(...)`
接口。模型服务不发布 ROS topic，不生成 waypoint，也不拥有物理 STOP 权限。

## 1. 统一调用合同

### 1.1 服务端接口

商业 API 或自部署服务至少应支持：

```text
POST /v1/chat/completions
```

自部署服务另外提供：

```text
GET /health
GET /v1/models
```

`/health` 和 `/v1/models` 是自部署服务的正式验收接口，不是所有商业 API 都提供。
因此商业 API 的 smoke 可以跳过这两个非标准端点，直接验证模型调用。

请求使用 OpenAI-compatible multimodal message：

```jsonc
{
  "model": "<served-model-name>",
  "messages": [
    {
      "role": "system",
      "content": "任务 prompt + Pydantic JSON schema"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,..."}
        },
        {
          "type": "text",
          "text": "原始指令、候选对象、几何事实和 ledger 上下文"
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 1024
}
```

机器人发送图片前应完成 resize/padding，并优先发送对象 crop 或验证所需的局部视图，
避免将完整全景图重复上传给每个 verifier。API key、原始图片和模型响应不得写入
公开仓库。

### 1.2 停止权限边界

```text
LVLM owns:
  semantic interpretation, relation evidence, view quality feedback

Planner / geometry owns:
  distance, reachability, collision, physical stop contract

Final STOP:
  semantic constraints + physical stop contract + final evidence
```

网络、供应商拒绝或 schema 校验错误必须保留原因并向上传播；证据不可用时保持 unknown，不得产生 final STOP。`STRIVE_LLM_FALLBACK` 仅是显式历史兼容选项，正式运行关闭，不将异常改写为目标接受。

## 2. 方式 A：商业 API 访问

商业 API 模式不需要下载模型、不需要安装 vLLM，也不需要在机器人上运行 GPU 推理。
机器人容器只安装项目运行依赖，通过供应商提供的 HTTPS endpoint 发送请求。

### 2.1 Ark / OpenAI-compatible API

以 Ark 为例，配置实际的模型 ID、API key 和 endpoint：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY="<provider-api-key>"
export LLM_MODEL="<provider-model-id>"
export LLM_API_BASE_URL="https://<provider-endpoint>/api/v3"
```

OpenAI-compatible 供应商使用同一配置合同：

```bash
export LLM_PROVIDER=ark
export OPENAI_API_KEY="<provider-api-key>"
export LLM_MODEL="<provider-model-id>"
export LLM_API_BASE_URL="https://<provider-endpoint>/v1"
```

代码通过 `llm_utils.cognav_llm_adapter.get_client_and_model()` 创建统一客户端。
`LLM_MODEL` 同时作为 provider model name；不要把一个供应商的模型名与另一个
endpoint 混用。

### 2.2 Gemini API

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="<provider-api-key>"
export GEMINI_MODEL="<provider-model-id>"
```

Gemini 的 OpenAI-compatible endpoint 由客户端适配器固定配置。若供应商账户不提供
OpenAI-compatible 多模态接口，应使用该供应商对应的适配器，而不是修改 verifier
内部 prompt 逻辑。

### 2.3 商业 API 调用 smoke

商业 API 不一定暴露 `/health` 或 `/v1/models`。使用项目 smoke 工具时跳过非标准
能力检查，只验证一次真实 chat completion：

```bash
python deployment/lvlm_server/smoke_client.py \
  --base-url "$LLM_API_BASE_URL" \
  --model "$LLM_MODEL" \
  --api-key "$ARK_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --skip-health \
  --skip-model-discovery
```

然后使用生产 prompt 和 schema 做结构化 smoke：

```bash
python -m deployment.lvlm_server.schema_smoke \
  --base-url "$LLM_API_BASE_URL" \
  --model "$LLM_MODEL" \
  --api-key "$ARK_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --skip-health \
  --skip-model-discovery \
  --output outputs/lvlm/commercial_schema_smoke.json
```

如果供应商提供 `/models` 但没有 `/health`，只使用 `--skip-health`；如果两者都提供，
不传 skip 参数以获得完整能力校验。

### 2.4 商业 API 导航 smoke

结构化 smoke 通过后再运行一个 episode。仿真：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir logs/lvlm_commercial_smoke \
  --vlm ark \
  --max_steps 50 \
  --clean_save_dir
```

实物机器人容器使用同一组 `LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_BASE_URL` 和 API
key 环境变量；服务端不需要能访问机器人网络。

## 3. 方式 B：自部署模型服务

自部署模式将模型放在独立 GPU 服务器，机器人只通过公网 HTTPS 访问模型服务：

```text
Robot / Habitat client
  -- outbound HTTPS /v1/chat/completions -->
Public DNS + TLS reverse proxy + authentication
  --> 127.0.0.1:8000
      ms-swift / vLLM / Qwen-VL
```

机器人和模型服务器之间只传输 LVLM 请求与响应；模型服务器不连接机器人 ROS 网络，
不访问 `/cmd_vel`、`/way_point` 或传感器 topic。

### 3.1 当前 Qwen3.5-4B 部署包

当前 Qwen3.5 服务与仓库内的 ms-swift 启动器是**两条不同路径**：

| 路径 | 入口 / 引擎 | 交付依赖 |
|---|---|---|
| Qwen3.5-4B | 独立包 `start_qwen.sh` → `qwen35_transformers_api.py` / Transformers + XGrammar | 额外交付部署脚本、镜像、完整权重及 processor；仅克隆 STRIVE 不够 |
| 可选 Qwen2.5-VL | 仓库 `deployment/lvlm_server/run_server.sh` / ms-swift + vLLM | 按后续章节安装并验收 |

Qwen3.5 独立脚本当前位于 STRIVE 仓库外的 `STRIVE-deploy-105/`，不受本仓库 Git 管理；部署者应单独保存脚本 SHA256、镜像 ID 和依赖版本，不能用 STRIVE commit 代表整个模型服务版本。镜像标签声明的 vLLM 版本也不代表实际运行的是 vLLM。

准备条件：NVIDIA Docker GPU runtime、已分配的空闲 GPU、完整本地 Qwen3.5-4B 权重（含索引引用的所有 shard、tokenizer、processor）、独立部署包及其固定镜像。包内入口当前使用的镜像标签为：

```text
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3
```

实际 Python 服务依赖 torch、transformers、xgrammar、fastapi、Pillow、uvicorn；尚无独立的 Transformers / XGrammar 版本锁文件，迁移时必须从验收镜像记录真实版本。下载或更新其他镜像不能视为等价复现。

从新机器启动一份独占服务的示例（路径、GPU、端口按实际填写；已有服务时先检查，不重复启动或自动替换）：

```bash
export DEPLOY_ROOT="/path/to/STRIVE-deploy-105"
export MODEL_ROOT="/models/Qwen3.5-4B"
export QWEN_CONTAINER_NAME="strive-qwen35-api"
export QWEN_GPU=0
export QWEN_PORT=18182
bash "$DEPLOY_ROOT/start_qwen.sh"
```

启动器以只读方式挂载脚本/权重，离线加载，监听 `127.0.0.1`；默认端口原为18180，上例显式设置为客户端使用的18182。启动前的 shard 存在检查不等于内容完整性检查；仍需权重哈希、实际加载和多模态请求验收。

**认证边界：**这个启动脚本没有向容器传入 `QWEN_API_KEY`，服务未配置该变量时不校验密钥。它只能保持回环访问；客户端设置 API key 不会自动开启服务端认证。跨机交付必须另行配置经过验收的鉴权网关（或显式为服务注入认证配置），不能直接暴露端口，也不能把下文仅转发 Authorization 的 Nginx 示例当成鉴权实现。

服务通过 `enable_thinking=False` 关闭思考输出；收到 `guided_json` 时使用 XGrammar 约束生成。`GET /health` 返回 `thinking=false` 和 `structured_output=xgrammar`，但仍应实际调用生产 schema 验证，而不只看声明。

配置与 smoke：

```bash
export LLM_PROVIDER=self_hosted
export STRIVE_LLM_CLIENT=self_hosted
export VLN_LVLM_BASE_URL="http://127.0.0.1:18182/v1"
export VLN_LVLM_MODEL="huaweivln-qwen3.5-4b"
export VLN_LVLM_GUIDED_JSON=1
export VLN_LVLM_TIMEOUT_S=180
export VLN_LVLM_TRANSPORT_RETRIES=0
export VLN_LVLM_PARSE_RETRIES=1
export STRIVE_LLM_FALLBACK=0
export LLM_OFFLINE=0

python deployment/lvlm_server/smoke_client.py \
  --base-url "$VLN_LVLM_BASE_URL" --model "$VLN_LVLM_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" --image /path/to/navigation_frame.jpg

python -m deployment.lvlm_server.schema_smoke \
  --base-url "$VLN_LVLM_BASE_URL" --model "$VLN_LVLM_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" --image /path/to/navigation_frame.jpg \
  --output outputs/lvlm/qwen35_schema_smoke.json
```

私有环境预先提供 `VLN_LVLM_API_KEY`；其值必须与已启用的认证端一致。同机回环未认证测试与跨机正式服务分别验收。通用 schema smoke 不覆盖全部运行时动态 ID 约束、关系身份或功能证据 schema，仍需一例实际导航请求链。

### 3.2 可选 ms-swift / vLLM 服务器参数

以下为原有 Qwen2.5-VL 部署路径，不用于声称已部署上述 Qwen3.5 服务。变量是目标服务器参数：

```bash
export VLN_ROOT=/opt/vln
export MS_SWIFT_ROOT="$VLN_ROOT/ms-swift"
export MODEL_ROOT=/models
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export MODEL_PATH="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
export MS_SWIFT_REVISION="<approved-ms-swift-tag-or-commit>"
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_API_KEY="<long-random-token>"
```

模型来源、版本和权重路径由部署者配置。本仓库不依赖某台开发机的缓存目录，也不把
模型文件提交到 Git。

### 3.3 准备模型和 ms-swift

```bash
mkdir -p "$VLN_ROOT/venvs" "$MODEL_ROOT"
git clone https://github.com/modelscope/ms-swift.git "$MS_SWIFT_ROOT"
git -C "$MS_SWIFT_ROOT" checkout "$MS_SWIFT_REVISION"

huggingface-cli download "$MODEL_ID" \
  --local-dir "$MODEL_PATH" \
  --local-dir-use-symlinks False

python3 -m venv "$VLN_ROOT/venvs/lvlm"
source "$VLN_ROOT/venvs/lvlm/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$MS_SWIFT_ROOT"
python -m pip install vllm openai
```

模型目录必须包含 `config.json`、tokenizer、processor、generation config、
`model.safetensors.index.json` 及其引用的全部 safetensors shard。

### 3.4 权重完整性检查

```bash
python deployment/lvlm_server/preflight.py \
  --model "$MODEL_PATH" \
  --model-only \
  --sha256 \
  > "$MODEL_PATH.model-manifest.json"
```

正式迁移比较 `weight_shards`、`weight_bytes` 和 `weight_sha256`。缓存目录存在、目录
总大小或部分 shard 数量都不能证明权重完整。

### 3.5 启动内部服务

服务进程只监听回环地址，公网流量由 TLS reverse proxy 接收：

```bash
export MS_SWIFT_ROOT=/opt/vln/ms-swift
export VLN_LVLM_MODEL_PATH="$MODEL_PATH"
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_HOST=127.0.0.1
export VLN_LVLM_PORT=8000
export VLN_LVLM_API_KEY="<long-random-token>"

bash deployment/lvlm_server/run_server.sh
```

启动器会执行 model/runtime preflight，然后启动 `swift.cli.deploy` 和 vLLM。
显存、上下文长度和视觉 token 参数见 `deployment/lvlm_server/model_profile.env.example`，需显式导出环境变量，启动器不会自动加载该模板。原脚本和模板的 host 默认值为 `0.0.0.0`，因此必须保留上例 `VLN_LVLM_HOST=127.0.0.1` 覆盖，并核对实际监听地址与认证状态。

## 4. 公网 HTTPS 访问服务

这里的“公网 HTTP 服务”是 HTTP API 语义，不应使用明文 HTTP 传输 API key 和图像。
生产环境必须使用域名、TLS 和认证：

```text
https://<public-domain>/v1/chat/completions
```

### 4.1 反向代理边界

以 Nginx 为例，公网只开放443，内部服务只监听回环地址。下面只示意 TLS 与转发，**不包含鉴权实现**；必须配合服务端密钥校验或独立鉴权网关。Qwen3.5 使用18182时应同时修改 upstream 端口；未配置认证前不可对公网开放：

```nginx
server {
    listen 443 ssl http2;
    server_name <public-domain>;

    ssl_certificate     /etc/letsencrypt/live/<public-domain>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<public-domain>/privkey.pem;
    client_max_body_size 20m;

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_read_timeout 30s;
    }

    location /v1/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header Authorization $http_authorization;
        proxy_read_timeout 120s;
        proxy_send_timeout 120s;
    }
}
```

防火墙只允许 443；不要公开暴露 8000、vLLM 管理接口或模型文件目录。应配置访问
频率限制、请求体上限、TLS 证书自动续期和服务端日志脱敏。

### 4.2 从机器人侧验证公网 endpoint

```bash
export VLN_LVLM_BASE_URL="https://<public-domain>/v1"
export VLN_LVLM_API_KEY="<same-long-random-token>"
export VLN_LVLM_MODEL="$VLN_LVLM_SERVED_MODEL"

curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer $VLN_LVLM_API_KEY" \
  "$VLN_LVLM_BASE_URL/../health"

curl --fail-with-body --silent --show-error \
  -H "Authorization: Bearer $VLN_LVLM_API_KEY" \
  "$VLN_LVLM_BASE_URL/models"
```

随后从机器人或仿真容器运行：

```bash
python deployment/lvlm_server/smoke_client.py \
  --base-url "$VLN_LVLM_BASE_URL" \
  --model "$VLN_LVLM_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg
```

如果公网网关不转发 `/health`，可使用 `--skip-health`，但正式自部署验收仍应在
服务器本机完成完整 health/model/schema gate。

## 5. 机器人侧配置

机器人容器不加载模型权重，也不安装 vLLM。选择 `self_hosted` 后，客户端通过公网
HTTPS endpoint 访问服务器：

```bash
export LLM_PROVIDER=self_hosted
export STRIVE_LLM_CLIENT=self_hosted
export STRIVE_VLM=self_hosted
export STRIVE_INSTRUCTION_PLAN_BACKEND=llm
export VLN_LVLM_BASE_URL="https://<public-domain>/v1"
export VLN_LVLM_API_KEY="<same-long-random-token>"
export VLN_LVLM_MODEL="<served-model-name>"
export VLN_LVLM_TIMEOUT_S=90
export VLN_LVLM_TRANSPORT_RETRIES=2
export VLN_LVLM_PARSE_RETRIES=1
```

Docker 实物入口会透传这些变量；ROS2 runtime 仍只负责 semantic snapshot、
NavigationIntent、waypoint 和 status/evidence 闭环。服务器端不需要安装 ROS2，也不
需要能够访问机器人局域网。

旧的 `STRIVE_LVLM_*`、`STRIVE_LLM_CLIENT` 和 `STRIVE_VLM` 变量继续兼容；新部署优先
使用 `VLN_LVLM_*`。

## 6. 结构化输出与安全边界

所有上层模块保持同一调用形状：

```python
completion = client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=ParsedResult,
    trace_label="final_instruction_verifier",
)
result = completion.choices[0].message.parsed
```

自部署客户端使用普通 chat completion，并按服务能力发送 schema：

```text
Pydantic schema → system prompt / guided_json
  → HTTP chat completion → 提取 JSON → Pydantic 校验
  → 合法值；或保留原始非法响应的结构化错误
```

传输重试与格式修正是两类独立预算。`VLN_LVLM_PARSE_RETRIES` 配置适配器默认修正次数；初始解释和运行时 Agent 使用单次调用覆盖 `parse_retries=0`，由上层明确管理修正/决策预算。初始解释最多一次初次解析和一次结构修正，不因401、超时、供应商拒绝而发起“语义修正”。感知/验证的调用数仍另列，不能把初始解释加 Agent 的上限当作全模型调用总上限。

语法受约束不代表语义正确；目标/参照身份、方向关系与功能条件仍需实际观测证据。商业 API 的结构化能力也不能替代这些验证。模型上下文与错误 trace 可能包含原文和图像内容，应按运行数据保护。

## 7. 分层验收

### 7.1 商业 API

商业 API 不执行本地权重和 CUDA preflight。最低验收要求：

1. 供应商 endpoint、模型 ID 和 API key 可用；
2. multimodal chat smoke 通过；
3. 项目通用 schema smoke 通过，并检查实际任务的动态 schema 请求；
4. 一次仿真或实物 replay 的请求、响应、超时与错误产物完整；
5. 失败响应不能产生 final STOP。

### 7.2 自部署模型

原有 ms-swift / vLLM 路径在服务器本机执行下列验收器；它检查 ms-swift 依赖，不应直接用于证明 Transformers Qwen3.5 服务通过。Qwen3.5 分别记录部署脚本/镜像/实际依赖版本与权重哈希，完成第3.1节 smoke、认证检查及实际导航请求验收：

```bash
python deployment/lvlm_server/accept_deployment.py \
  --model-path "$MODEL_PATH" \
  --ms-swift-root "$MS_SWIFT_ROOT" \
  --expected-ms-swift-revision "$MS_SWIFT_REVISION" \
  --base-url http://127.0.0.1:8000/v1 \
  --served-model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --sha256 \
  --output outputs/lvlm/deployment_acceptance.json
```

机器人侧再执行公网 endpoint smoke。完整上线条件是：权重清单、运行环境、内部
服务、TLS 网关、公网认证、实际使用的 schema 和机器人侧请求全部通过。

记录每个 trace label 的请求数、缓存命中、图像字节数、p50/p95 延迟、timeout、
schema repair、GPU 显存和利用率。部署回执与模型清单保存在 `outputs/lvlm/`，不提交
Git；部署摘要不记录图像载荷或 API key。实际模型请求 trace 可能包含图像内容，属于受控运行证据，不应发布。

## 8. 相关入口

- [实物模式接口与数据流](real_robot_framework.md)
- [项目文档与产物目录约定](README.md)
