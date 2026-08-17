# VLN Self-Hosted LVLM Service

This package deploys a vision-language model as an OpenAI-compatible HTTP
service. The deployment host owns model weights and GPU inference; VLN owns
prompt construction, structured-output validation, navigation semantics, and
the client-side safety boundary.

The procedure below is intended for a clean target machine. It does not rely
on a development workstation, a shared model cache, or a particular local
directory layout.

## 1. Deployment Contract

The server must provide:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

The service must accept text and `image_url` data URLs, return an assistant
message, and expose the configured served model name through `/v1/models`.
The VLN client validates the response against the production Pydantic schema;
the server is not responsible for navigation stop decisions.

The repository provides:

| Component | Responsibility |
|---|---|
| `run_server.sh` | Preflight and `ms-swift` service startup |
| `preflight.py` | Model snapshot, shard, dependency, and CUDA checks |
| `smoke_client.py` | HTTP health, model discovery, and multimodal smoke |
| `schema_smoke.py` | Five production prompt/schema acceptance cases |
| `accept_deployment.py` | Versioned target-server acceptance receipt |
| `model_profile.env.example` | Deployment parameter template |

## 2. Prerequisites

Prepare the following on the target GPU server:

- Linux with a compatible NVIDIA driver and CUDA runtime.
- Python 3.10 or a version supported by the selected `ms-swift` release.
- A complete instruction-tuned Qwen-VL or compatible multimodal checkpoint.
- A clean, pinned `ms-swift` source checkout.
- Network access from the VLN client host to the server's private HTTP port.

The model and `ms-swift` versions are deployment parameters. Select them from
the versions approved for the target GPU and record the selected IDs and
commits in the deployment receipt.

## 3. Directory and Environment Setup

Choose paths that exist on the target machine:

```bash
export VLN_ROOT=/opt/vln
export MS_SWIFT_ROOT="$VLN_ROOT/ms-swift"
export MODEL_ROOT=/models
export MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
export MODEL_PATH="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
export MS_SWIFT_REVISION=<approved-ms-swift-tag-or-commit>
export VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
export VLN_LVLM_HOST=0.0.0.0
export VLN_LVLM_PORT=8000
export VLN_LVLM_API_KEY=<private-token>
```

The served model name is an API identifier. It does not need to equal the
model repository ID, but the same value must be used by the server, smoke
tests, and VLN clients.

Clone and pin `ms-swift`:

```bash
mkdir -p "$VLN_ROOT" "$MODEL_ROOT"
git clone https://github.com/modelscope/ms-swift.git "$MS_SWIFT_ROOT"
git -C "$MS_SWIFT_ROOT" checkout "$MS_SWIFT_REVISION"
```

Create an isolated environment and install versions compatible with the target
GPU driver:

```bash
python3 -m venv "$VLN_ROOT/venvs/lvlm"
source "$VLN_ROOT/venvs/lvlm/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$MS_SWIFT_ROOT"
python -m pip install vllm openai
```

For a production deployment, freeze the resulting environment and record the
GPU, driver, CUDA, PyTorch, vLLM, and `ms-swift` versions.

## 4. Obtain and Verify the Model

Obtain the checkpoint from an approved model registry, object store, or model
artifact service. The following is one possible registry workflow:

```bash
mkdir -p "$MODEL_ROOT"
huggingface-cli download "$MODEL_ID" \
  --local-dir "$MODEL_PATH" \
  --local-dir-use-symlinks False
```

If the checkpoint is transferred by another system, use the resulting local
directory directly. It must contain `config.json`, tokenizer and processor
files, generation configuration, and all files referenced by
`model.safetensors.index.json`.

Run the repository preflight with shard hashes:

```bash
python "$VLN_ROOT/deployment/lvlm_server/preflight.py" \
  --model "$MODEL_PATH" \
  --model-only \
  --sha256 \
  > "$MODEL_PATH.model-manifest.json"
```

The manifest records the indexed shard list, byte sizes, and per-shard SHA256
digests. Compare those fields after any transfer. A cache directory, a model
directory's total size, or the presence of only some shards is not sufficient
evidence of a complete snapshot. Keep manifests outside Git and archive them
with the acceptance receipt.

## 5. Configure and Start the Server

Copy the template to a private file and adjust the deployment parameters:

```bash
cp "$VLN_ROOT/deployment/lvlm_server/model_profile.env.example" \
  /private/config/vln_lvlm.env
source /private/config/vln_lvlm.env
```

At minimum, the profile must define:

```bash
MS_SWIFT_ROOT=/opt/vln/ms-swift
VLN_LVLM_MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct
VLN_LVLM_SERVED_MODEL=vln-qwen2.5-vl-7b
VLN_LVLM_HOST=0.0.0.0
VLN_LVLM_PORT=8000
VLN_LVLM_API_KEY=<private-token>
```

Start the service from the VLN repository root:

```bash
cd "$VLN_ROOT"
bash deployment/lvlm_server/run_server.sh
```

The launcher runs model and runtime preflight before starting
`swift.cli.deploy` with the vLLM backend. Resource parameters such as maximum
model length, image limits, and GPU memory utilization are configuration
values; tune them for the selected model and GPU, then record the final
profile in the deployment receipt.

## 6. Transport and Schema Smoke Tests

From the VLN checkout, verify service readiness and one multimodal request:

```bash
python deployment/lvlm_server/smoke_client.py \
  --base-url "http://SERVER_IP:8000/v1" \
  --model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg
```

Then run the five production contracts:

```bash
python -m deployment.lvlm_server.schema_smoke \
  --base-url "http://SERVER_IP:8000/v1" \
  --model "$VLN_LVLM_SERVED_MODEL" \
  --api-key "$VLN_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --output artifacts/lvlm_schema_smoke_receipt.json
```

The schema gate covers instruction parsing, concept grounding, batched
concept matching, relation verification, and final instruction verification.
It also checks that a final verifier cannot authorize STOP while the
planner-owned physical stop contract is unsatisfied.

## 7. Formal Acceptance

Run the complete acceptance gate on the same machine and environment used to
serve the model:

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

Acceptance requires all of the following:

```text
clean pinned ms-swift checkout
complete model snapshot and shard SHA256
serving dependency and CUDA preflight
/health and /v1/models
five production structured-output schemas
```

The receipt is the deployment record. It should include the model manifest,
source revision, runtime inventory, endpoint, served model name, schema smoke
results, and failure details when a stage does not pass.

## 8. Configure the VLN Client

The client host does not need model weights or vLLM. Configure the preferred
VLN environment variables:

```bash
export LLM_PROVIDER=self_hosted
export VLN_LVLM_BASE_URL=http://SERVER_IP:8000/v1
export VLN_LVLM_API_KEY=<same-token-as-server>
export VLN_LVLM_MODEL="$VLN_LVLM_SERVED_MODEL"
export VLN_LVLM_TIMEOUT_S=45
export VLN_LVLM_TRANSPORT_RETRIES=2
export VLN_LVLM_PARSE_RETRIES=1
```

For natural-language instruction parsing, set the corresponding instruction
backend to `llm`. The self-hosted adapter keeps the existing internal client
shape, injects the Pydantic schema into the prompt, validates the response
locally, and returns a conservative result on transport or parsing failure.

The legacy `STRIVE_LVLM_*`, `STRIVE_LLM_CLIENT`, and `STRIVE_VLM` names remain
supported for existing launch files. New deployment profiles should use the
`VLN_LVLM_*` names consistently.

## 9. Security and Operations

- Do not commit model weights, API keys, model manifests, or deployment
  receipts containing sensitive endpoint metadata.
- Keep the service on a private network, VPN, or authenticated TLS reverse
  proxy. Do not expose an unauthenticated inference port to the public
  Internet.
- Image inputs are sent as data URLs; the server must not depend on paths that
  exist only on the client host.
- Retain latency, timeout, retry, schema-repair, GPU-memory, and utilization
  metrics for capacity planning.
- A reachable HTTP port is not deployment acceptance. The model, runtime,
  schema behavior, and failure semantics must all pass the formal receipt.

## 10. Related Documentation

- [`docs/lvlm_server_deployment.md`](../../docs/lvlm_server_deployment.md):
  architecture, HTTP contract, and integration details.
- [`docs/real_robot_framework.md`](../../docs/real_robot_framework.md):
  platform-neutral real-robot integration.
