# VLN 自部署 LVLM 服务

本目录实现 **自部署模型** 方案。商业 API、机器人到模型服务器的 HTTPS 配置、供应商
环境变量以及完整验收边界，请先阅读项目级文档
[`docs/lvlm_server_deployment.md`](../../docs/lvlm_server_deployment.md)。

本目录将视觉语言模型部署为 OpenAI-compatible HTTP 服务。模型权重和 GPU 推理由服务
器负责；VLN 负责 prompt 构造、结构化输出校验、导航语义和客户端安全边界。

本文面向一台全新 GPU 服务器，不依赖开发机上的模型缓存或固定目录。模型服务通常只
监听服务器本机回环地址，机器人通过公网 TLS 反向代理访问该服务。

## 1. 部署合同

自部署服务必须提供以下接口：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

服务需要接受文本以及 `image_url` data URL，返回 assistant message，并通过
`/v1/models` 暴露配置的服务模型名。VLN 客户端负责按照生产环境 Pydantic schema 校验
响应；模型服务本身不负责导航停止决策。

仓库中的部署组件如下：

| 组件 | 作用 |
|---|---|
| `run_server.sh` | 执行 preflight 并启动 `ms-swift` 服务 |
| `preflight.py` | 检查模型快照、权重 shard、依赖和 CUDA 环境 |
| `smoke_client.py` | 检查 HTTP 服务、模型发现和多模态调用 |
| `schema_smoke.py` | 执行五类生产 prompt/schema 验收用例 |
| `accept_deployment.py` | 生成版本化的目标服务器验收回执 |
| `model_profile.env.example` | 部署参数模板 |

## 2. 前置条件

在目标 GPU 服务器上准备：

- 与目标 CUDA runtime 兼容的 Linux、NVIDIA 驱动和 GPU 环境；
- Python 3.10，或所选 `ms-swift` 版本支持的 Python 版本；
- 完整的指令微调 Qwen-VL 或兼容的多模态模型权重；
- 干净且固定版本的 `ms-swift` 源码；
- 面向机器人访问的公网 DNS、TLS 证书和带认证的反向代理。模型服务端口保持私有。

模型和 `ms-swift` 版本属于部署参数。应根据目标 GPU 选择经过验证的版本，并将模型
标识、源码 commit 和运行环境记录到部署回执中。

## 3. 目录与环境变量

先选择目标服务器上实际存在或准备创建的目录：

```bash
export VLN_ROOT=/opt/vln
export MS_SWIFT_ROOT="$VLN_ROOT/ms-swift"
export MODEL_ROOT=/models
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export MODEL_PATH="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
export MS_SWIFT_REVISION=<approved-ms-swift-tag-or-commit>
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_HOST=127.0.0.1
export VLN_LVLM_PORT=8000
export VLN_LVLM_API_KEY=<private-token>
```

`VLN_LVLM_SERVED_MODEL` 是 API 层的模型标识，不必与模型仓库 ID 相同，但服务启动、
smoke 测试和 VLN 客户端必须使用同一个值。

### 3.1 获取并固定 `ms-swift`

```bash
mkdir -p "$VLN_ROOT" "$MODEL_ROOT"
git clone https://github.com/modelscope/ms-swift.git "$MS_SWIFT_ROOT"
git -C "$MS_SWIFT_ROOT" checkout "$MS_SWIFT_REVISION"
```

创建隔离环境，并安装与目标 GPU 驱动兼容的依赖：

```bash
python3 -m venv "$VLN_ROOT/venvs/lvlm"
source "$VLN_ROOT/venvs/lvlm/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$MS_SWIFT_ROOT"
python -m pip install vllm openai
```

正式部署时，应冻结最终环境，并记录 GPU、驱动、CUDA、PyTorch、vLLM 和 `ms-swift`
版本。

## 4. 获取并校验模型

从经过批准的模型仓库、对象存储或模型制品服务获取权重。以下是一个示例流程：

```bash
mkdir -p "$MODEL_ROOT"
huggingface-cli download "$MODEL_ID" \
  --local-dir "$MODEL_PATH" \
  --local-dir-use-symlinks False
```

如果权重由其他系统传输，只需使用传输后的本地目录。该目录必须包含 `config.json`、
tokenizer、processor、generation config，以及 `model.safetensors.index.json` 引用的
全部文件。

执行带 shard 哈希的仓库 preflight：

```bash
python "$VLN_ROOT/deployment/lvlm_server/preflight.py" \
  --model "$MODEL_PATH" \
  --model-only \
  --sha256 \
  > "$MODEL_PATH.model-manifest.json"
```

manifest 会记录索引中的 shard 列表、文件字节数和每个 shard 的 SHA256。权重转移后应
对比这些字段。缓存目录存在、模型目录总大小正确，或只存在部分 shard，都不能证明
模型快照完整。manifest 应保存在 Git 之外，并随验收回执归档。

## 5. 配置并启动服务

将配置模板复制到私有目录，再按目标服务器修改参数：

```bash
cp "$VLN_ROOT/deployment/lvlm_server/model_profile.env.example" \
  /private/config/vln_lvlm.env
source /private/config/vln_lvlm.env
```

至少应配置：

```bash
MS_SWIFT_ROOT=/opt/vln/ms-swift
VLN_LVLM_MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct
VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
VLN_LVLM_HOST=127.0.0.1
VLN_LVLM_PORT=8000
VLN_LVLM_API_KEY=<private-token>
```

从 VLN 仓库根目录启动服务：

```bash
cd "$VLN_ROOT"
bash deployment/lvlm_server/run_server.sh
```

启动器会在启动 `swift.cli.deploy` 和 vLLM 后端前执行模型与运行环境 preflight。最大
模型长度、图像数量、视觉 token 和 GPU 显存利用率等资源参数属于配置项，应根据所选
模型和 GPU 调整，并将最终配置记录到部署回执。

## 6. HTTP 与结构化输出 smoke

在模型服务器本机检查服务状态，并发送一次多模态请求：

```bash
python deployment/lvlm_server/smoke_client.py \
  --base-url "http://127.0.0.1:8000/v1" \
  --model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg
```

随后执行五类生产调用合同：

```bash
python -m deployment.lvlm_server.schema_smoke \
  --base-url "http://127.0.0.1:8000/v1" \
  --model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --output artifacts/lvlm_schema_smoke_receipt.json
```

schema gate 覆盖：

1. 指令解析；
2. 概念 grounding；
3. 批量概念匹配；
4. 语义关系验证；
5. 最终指令验证。

同时，验收会检查 final verifier 在规划器负责的物理停止合同不满足时，不能授权
`STOP`。

## 7. 正式部署验收

在提供模型服务的同一台服务器和同一运行环境中执行完整验收：

```bash
python -m deployment.lvlm_server.accept_deployment \
  --model-path "$MODEL_PATH" \
  --ms-swift-root "$MS_SWIFT_ROOT" \
  --expected-ms-swift-revision "$MS_SWIFT_REVISION" \
  --base-url "http://127.0.0.1:8000/v1" \
  --served-model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --sha256 \
  --output artifacts/lvlm_deployment_acceptance.json
```

正式验收至少要求以下项目全部通过：

```text
干净且固定版本的 ms-swift 源码
完整模型快照和 shard SHA256
服务依赖与 CUDA preflight
/health 和 /v1/models
五类生产结构化输出 schema
```

验收回执是部署记录，应包含模型 manifest、源码版本、运行环境、服务 endpoint、模型
名、schema smoke 结果，以及失败阶段的具体原因。

## 8. 配置 VLN 客户端

机器人或仿真客户端不需要加载模型权重，也不需要安装 vLLM。客户端通过公网 HTTPS
访问模型服务器：

```bash
export LLM_PROVIDER=self_hosted
export VLN_LVLM_BASE_URL=https://<public-domain>/v1
export VLN_LVLM_API_KEY=<same-token-as-server>
export VLN_LVLM_MODEL="$VLN_LVLM_SERVED_MODEL"
export VLN_LVLM_TIMEOUT_S=45
export VLN_LVLM_TRANSPORT_RETRIES=2
export VLN_LVLM_PARSE_RETRIES=1
```

使用自然语言指令解析时，将对应 instruction backend 设置为 `llm`。自部署 adapter
保持项目内部统一的 client 调用形式，将 Pydantic schema 注入 prompt，在客户端校验
响应，并在传输或解析失败时返回保守结果。

现有启动文件仍兼容 `STRIVE_LVLM_*`、`STRIVE_LLM_CLIENT` 和 `STRIVE_VLM` 变量。新
部署配置优先统一使用 `VLN_LVLM_*` 变量。

## 9. 安全与运行维护

- 不要提交模型权重、API key、模型 manifest，或包含敏感 endpoint 信息的部署回执；
- 模型服务保持在回环地址或私有网络中，公网只暴露带认证的 TLS 反向代理；
- 不要将未认证的推理端口直接暴露到公网；
- 图像通过 data URL 发送，服务端不能依赖只存在于机器人上的本地路径；
- 保留延迟、超时、重试、schema repair、GPU 显存和利用率指标，用于容量规划；
- HTTP 端口可访问不等于部署验收通过，模型、运行环境、schema 行为和失败语义都必须
  通过正式回执。

## 10. 相关文档

- [`docs/lvlm_server_deployment.md`](../../docs/lvlm_server_deployment.md)：两种部署方式、
  HTTP 合同、商业 API 和公网 HTTPS 接入；
- [`docs/real_robot_framework.md`](../../docs/real_robot_framework.md)：平台无关的实物
  机器人集成框架。
