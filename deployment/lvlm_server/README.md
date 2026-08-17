# VLN Self-Hosted LVLM Service

This directory deploys a local Qwen-VL model through the OpenAI-compatible
`ms-swift` API. Model templates and inference engines remain owned by
`ms-swift`; VLN only supplies deployment profiles, preflight checks, and
client smoke tests.

The development machine currently has two complete models in the requested
Qwen2.5-VL family:

- `Qwen2.5-VL-7B-Instruct`: five safetensors shards, 16,584,414,560 bytes;
- `Qwen2.5-VL-3B-Instruct`: two safetensors shards, 7,509,337,976 bytes.

The 7B snapshot is the default quality-oriented profile. The 3B snapshot is a
lower-memory fallback. Separate Hugging Face cache entries with `.incomplete`
shards are not deployable snapshots and must not be copied.

## Server layout

```text
/opt/strive/ms-swift/             ms-swift source checkout
/models/Qwen2.5-VL-7B-Instruct/   complete model snapshot
/opt/strive/STRIVE/deployment/lvlm_server/
```

Create an environment with a CUDA-compatible PyTorch, `transformers`, `vllm`,
`openai`, and the checked-out `ms-swift` package. Copy
`model_profile.env.example` to a private profile, source it, and run:

```bash
bash deployment/lvlm_server/run_server.sh
```

Use `preflight.py --model-only --sha256` before and after copying a model to
compare every indexed shard. The audited ms-swift source revision for this
deployment profile is `0f3875d40ebda34862519971100e7188a00273e3`; record any
intentional revision change with the target server's package and GPU inventory.

The script first verifies every shard referenced by
`model.safetensors.index.json`, imports the serving dependencies, and checks
CUDA. It then launches:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Run the client smoke from another machine:

```bash
python deployment/lvlm_server/smoke_client.py \
  --base-url http://SERVER_IP:8000/v1 \
  --model strive-qwen2.5-vl-7b \
  --api-key "$STRIVE_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg
```

After transport smoke passes, run the five production VLN schemas and keep
the versioned deployment receipt:

```bash
python -m deployment.lvlm_server.schema_smoke \
  --base-url http://SERVER_IP:8000/v1 \
  --model strive-qwen2.5-vl-7b \
  --api-key "$STRIVE_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --output artifacts/lvlm_schema_smoke_receipt.json
```

This gate uses the production prompts and Pydantic response contracts for
instruction parsing, concept grounding, batch concept matching, relation
verification, and final instruction verification. It fails if the final
verifier authorizes STOP while the planner-owned hard stop contract is false.

On the target GPU server, combine source revision, weight integrity, CUDA,
service discovery, and schema checks into one formal receipt:

```bash
python -m deployment.lvlm_server.accept_deployment \
  --model-path /models/Qwen2.5-VL-7B-Instruct \
  --ms-swift-root /opt/strive/ms-swift \
  --base-url http://127.0.0.1:8000/v1 \
  --served-model strive-qwen2.5-vl-7b \
  --api-key "$STRIVE_LVLM_API_KEY" \
  --image /path/to/navigation_frame.jpg \
  --sha256 \
  --output artifacts/lvlm_deployment_acceptance.json
```

Do not expose this port directly to a public network. Use a private LAN, VPN,
or reverse proxy with TLS and authentication. VLN sends image data URLs, so
server-local image paths are neither required nor portable across machines.

## VLN client variables

```bash
export LLM_PROVIDER=self_hosted
export STRIVE_LLM_CLIENT=self_hosted
export STRIVE_VLM=self_hosted
export STRIVE_LVLM_BASE_URL=http://SERVER_IP:8000/v1
export STRIVE_LVLM_API_KEY=<same-token-as-server>
export STRIVE_LVLM_MODEL=strive-qwen2.5-vl-7b
export STRIVE_LVLM_TIMEOUT_S=45
export STRIVE_LVLM_TRANSPORT_RETRIES=2
export STRIVE_LVLM_PARSE_RETRIES=1
```

For the real-robot launcher, set `STRIVE_INSTRUCTION_PLAN_BACKEND=llm` when the
natural-language parser should also use this service. Direct `ros2 launch`
invocations must pass `vlm:=self_hosted` explicitly.

The self-hosted VLN adapter injects the Pydantic JSON schema into the system
prompt and validates the returned object locally. This is intentional:
`ms-swift` provides OpenAI-compatible chat completions, but its current request
contract does not expose the full OpenAI `beta.parse` response-format API.
