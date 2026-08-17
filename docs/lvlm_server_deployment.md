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

网络、解析或 schema 校验失败必须返回保守结果，不得产生 final STOP。

## 2. 方式 A：商业 API 访问

商业 API 模式不需要下载模型、不需要安装 vLLM，也不需要在机器人上运行 GPU 推理。
机器人容器只安装项目运行依赖，通过供应商提供的 HTTPS endpoint 发送请求。

### 2.1 Ark / OpenAI-compatible API

以 Ark 为例，配置实际的模型 ID、API key 和 endpoint：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY=<provider-api-key>
export LLM_MODEL=<provider-model-id>
export LLM_API_BASE_URL=https://<provider-endpoint>/api/v3
```

OpenAI-compatible 供应商使用同一配置合同：

```bash
export LLM_PROVIDER=ark
export OPENAI_API_KEY=<provider-api-key>
export LLM_MODEL=<provider-model-id>
export LLM_API_BASE_URL=https://<provider-endpoint>/v1
```

代码通过 `llm_utils.cognav_llm_adapter.get_client_and_model()` 创建统一客户端。
`LLM_MODEL` 同时作为 provider model name；不要把一个供应商的模型名与另一个
endpoint 混用。

### 2.2 Gemini API

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=<provider-api-key>
export GEMINI_MODEL=<provider-model-id>
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

### 3.1 服务器参数

以下变量是目标服务器的部署参数：

```bash
export VLN_ROOT=/opt/vln
export MS_SWIFT_ROOT="$VLN_ROOT/ms-swift"
export MODEL_ROOT=/models
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export MODEL_PATH="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
export MS_SWIFT_REVISION=<approved-ms-swift-tag-or-commit>
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_API_KEY=<long-random-token>
```

模型来源、版本和权重路径由部署者配置。本仓库不依赖某台开发机的缓存目录，也不把
模型文件提交到 Git。

### 3.2 准备模型和 ms-swift

```bash
mkdir -p "$VLN_ROOT" "$MODEL_ROOT"
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

### 3.3 权重完整性检查

```bash
python deployment/lvlm_server/preflight.py \
  --model "$MODEL_PATH" \
  --model-only \
  --sha256 \
  > "$MODEL_PATH.model-manifest.json"
```

正式迁移比较 `weight_shards`、`weight_bytes` 和 `weight_sha256`。缓存目录存在、目录
总大小或部分 shard 数量都不能证明权重完整。

### 3.4 启动内部服务

服务进程只监听回环地址，公网流量由 TLS reverse proxy 接收：

```bash
export MS_SWIFT_ROOT=/opt/vln/ms-swift
export VLN_LVLM_MODEL_PATH="$MODEL_PATH"
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_HOST=127.0.0.1
export VLN_LVLM_PORT=8000
export VLN_LVLM_API_KEY=<long-random-token>

bash deployment/lvlm_server/run_server.sh
```

启动器会执行 model/runtime preflight，然后启动 `swift.cli.deploy` 和 vLLM。
具体显存、上下文长度、视觉 token 和并发参数通过
`deployment/lvlm_server/model_profile.env.example` 配置。

## 4. 公网 HTTPS 访问服务

这里的“公网 HTTP 服务”是 HTTP API 语义，不应使用明文 HTTP 传输 API key 和图像。
生产环境必须使用域名、TLS 和认证：

```text
https://<public-domain>/v1/chat/completions
```

### 4.1 反向代理边界

以 Nginx 为例，公网只开放 443，内部 vLLM 只监听 `127.0.0.1:8000`：

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
export VLN_LVLM_BASE_URL=https://<public-domain>/v1
export VLN_LVLM_API_KEY=<same-long-random-token>
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
export VLN_LVLM_BASE_URL=https://<public-domain>/v1
export VLN_LVLM_API_KEY=<same-long-random-token>
export VLN_LVLM_MODEL=<served-model-name>
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

自部署服务通常只提供普通 `chat.completions.create`。客户端因此执行：

```text
Pydantic schema
  -> 注入 system prompt
  -> HTTP chat completion
  -> 提取 JSON object
  -> Pydantic 严格校验
  -> 有限 schema repair
  -> 失败时 conservative fallback
```

商业 API 若原生支持 OpenAI structured output，则由 provider SDK 执行 schema 校验；
上层 verifier 不写 provider-specific 分支。

## 7. 分层验收

### 7.1 商业 API

商业 API 不执行本地权重和 CUDA preflight。最低验收要求：

1. 供应商 endpoint、模型 ID 和 API key 可用；
2. multimodal chat smoke 通过；
3. 五个生产 schema smoke 通过；
4. 一次仿真或实物 replay 的 LVLM 调用、超时和 fallback 产物完整；
5. 失败响应不能产生 final STOP。

### 7.2 自部署模型

服务器本机执行：

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
服务、TLS 网关、公网认证、五类 schema 和机器人侧请求全部通过。

记录每个 trace label 的请求数、缓存命中、图像字节数、p50/p95 延迟、timeout、
schema repair、GPU 显存和利用率。部署回执与模型清单保存在 `outputs/lvlm/`，不提交
Git，也不记录图片 base64 或 API key。

## 8. 相关入口

- [实物模式接口与数据流](real_robot_framework.md)
- [项目文档与产物目录约定](README.md)
