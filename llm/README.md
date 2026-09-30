# llm/ · the model server on the robot (stage C)

The local model behind `mission_ctl/r1_mission_cli.py ask`. It is an Ollama build for
JetPack 5, running in user space on the robot's Orin NX: no sudo, no systemd, no apt, no
internet at run time. What the model does, and why it cannot move the robot by itself,
is in [docs/stageC_nl_eval.md](../docs/stageC_nl_eval.md).

| file | where it runs | what it does |
|---|---|---|
| `ollama_ctl.sh` | robot | `start`, `warm`, `health`, `status`, `stop`, `restart`, `watchdog`, `models`, `log` |
| `coexist.py` | robot | control-loop timing (obs Hz, setpoint lag, inference p99) while the model decodes on the GPU, the CPU, and in the real `ask` loop at `mission.yaml`'s placement (plan 3.6); judges the control loop's limits and plan 3.6's targets separately, and says what each placement measured; `--selftest` runs anywhere |
| `env_check.sh` | robot | memory, disk, clocks, and whether any route to a hosted model exists (plan 3.1) |
| `make_runtime.sh` | dev box | builds the runtime package: Ollama for JetPack 5 plus the models |

The runtime (`ollama/`) and the models (`models/`) are not in git (licences, ~3.5 GB).
They come in `~/r1_stageC_llm_runtime_<date>.tar.gz`, which extracts next to these
scripts.

## The runtime, and why it is built this way

Ollama **v0.34.4** (2026-09-23). The candidate runtimes were checked on 2026-09-28:
TensorRT-LLM needs JetPack 6.1, MLC-LLM on L4T R35 predates Qwen2, and
onnxruntime-genai needs L4T 36. v0.35.0 (2026-09-28) adds only a decision-model API
whose models are 4.5–9.5 GB.

- **Two release files:** `ollama-linux-arm64.tar.zst` (the binary and the CPU and
  llama.cpp libraries) plus `ollama-linux-arm64-jetpack5.tar.zst` (`cuda_jetpack5/`:
  the CUDA 11.4 runner and its own cuBLAS 11.6). Their SHA256 come from the GitHub
  release API and are checked by `make_runtime.sh`.
- **Unpacked on the dev box and repacked as gzip**, because the robot's tar (1.30,
  Ubuntu 20.04) cannot read zstd.
- **`cuda_v12/` and `cuda_v13/` are left out** (2.2 GB). They are for SBSA server GPUs.
  Ollama picks `cuda_jetpack5/` on L4T R35 (from `/etc/nv_tegra_release`, or
  `JETSON_JETPACK`, which `ollama_ctl.sh` sets to 5.1.1).
- **Checked on the dev box before shipping:**
  - every ELF file needs at most GLIBC_2.28 and GLIBCXX_3.4.25; Ubuntu 20.04 has 2.31
    and 3.4.28;
  - the CUDA runtime is 11.4.298, and JetPack 5.1.1 is CUDA 11.4;
  - `libggml-cuda.so` carries SASS for sm_72 and **sm_87** (Orin).

  None of this proved it runs on the Orin. **The robot session did (2026-09-30):**
  - it found the GPU at the first start: `library=CUDA compute=8.7 ...
    libdirs=ollama,cuda_jetpack5 ... type=iGPU`, with all 29 layers on the GPU;
  - the first load from disk took 3.7 s;
  - it ran about 70 minutes, through 7 model loads and 3 placements, with no restart
    and no error in the log.
- **The integrated GPU:** Ollama admits CUDA integrated GPUs by default.
  `ollama_ctl.sh` sets `OLLAMA_IGPU_ENABLE=1` anyway, so a future default cannot
  silently put the model on the CPU. `status` shows where the model landed
  (`size_vram`).

## Known faults, and what handles each

| fault | handled by |
|---|---|
| llama-server on the Orin NX sometimes starts and never serves (llama.cpp #29499, open), while `/api/version` keeps answering. Not seen in the 2026-09-30 session | `health` makes one real `ask` request within a deadline (`r1_mission_cli.py ask-check --once`), and `watchdog` restarts after 2 failures. `ask` times out with the restart command in its message |
| Ollama reloads the model whenever a request asks for another placement, and keeps only the last prompt in its cache | everything that talks to the server sends what `ask` sends, from `mission.yaml`: `health`/`warm`/`watchdog` through `ask-check --once`, and `coexist.py`'s `ask` phase. Until 2026-09-30 the health check was a bare generate at Ollama's default placement, which would have moved a CPU-placed model back to the GPU every 30 s |
| a vision projector reserves a 32 GB VMM pool, which fails on Jetson (llama.cpp #29142, open) | only text-only models are shipped: qwen3 and qwen2.5 have no projector |
| tar 1.30 has no zstd | repacked as gzip on the dev box |
| the first request loads the model from disk | `ollama_ctl.sh warm` before the first `ask`: it loads the model where `ask` will use it and leaves the `ask` prompt cached. `keep_alive` 30 min |

## Where the model runs: GPU or CPU (plan 3.6, measured 2026-09-30)

The Orin NX's GPU is shared with the policy's TensorRT engine. `coexist.py` on the
robot, qwen3:1.7b, the stack at kp 1.3, in two rounds (output off, then on):

| model on | policy inference p99 | setpoint lag p95 | obs Hz | decode | limits | plan 3.6 targets |
|---|---|---|---|---|---|---|
| — (idle) | 486 µs | 0.6 ms | 50.0 | | | |
| **GPU (kept)** | 5010 µs | 5.1 ms | 50.0 | 32 tok/s | kept | **missed** (2000 µs; idle + 1 ms) |
| CPU, 4 threads | 542 µs | 0.9 ms | 50.0 | 21 tok/s | kept | met |

`coexist.py` judges each phase in two tiers:
- **Limits:** what the control loop cannot give up. Obs ≥ 45 Hz with no slow window and
  no DEGRADED; no failed inference, and the slowest inference inside the 20 ms step;
  setpoint lag p95 inside the one step the policy was trained with. A broken limit fails
  the run.
- **Targets:** plan 3.6 as written. A miss is reported, and the run still passes.

**Decision (user, 2026-09-30): the model stays on the GPU; the GPU side is optimized
later.**
- On the GPU the policy waits for the GPU while the model decodes. Its inference runs at
  3.5 ms p50, with a steady ceiling of about 5 ms, which looks like the GPU's time slice.
- Nothing misbehaved: no DEGRADED and no policy failure, with the robot standing too.
  5 ms is a quarter of a control step.
- The targets were set before the measurement. Keeping the GPU relaxes them after the
  measurement, and that is recorded as such. They stay as the optimization's goal (the
  candidates are in
  [docs/system_architecture.md](../docs/system_architecture.md#the-shared-gpu-decision-2026-09-30)).
- For the record, the CPU: with 8 threads, qwen3:1.7b decodes at 24 tok/s, p50 1.1 s per
  instruction (0.65 s on the GPU). The first request after a load takes 11 s, because the
  prompt is processed on the CPU.
- Switching is two lines in `mission.yaml` (`llm_num_gpu: 0`, `llm_num_thread: 4`).
  `warm`, `health` and `ask` then all send the same placement.

## If Ollama will not run: the llama-server fallback

`ask --backend openai --llm-url http://127.0.0.1:<port>` speaks the OpenAI API, which
llama-server also serves. The request switches thinking off in both spellings:
`reasoning_effort` (Ollama) and `chat_template_kwargs` (llama-server). Without that,
qwen3 used up its token budget thinking. It is verified against Ollama's
OpenAI-compatible endpoint (77/80, the same as the native API); the llama-server build
itself is not.

## Models

`qwen3:1.7b` (the eval's pick, `mission.yaml` default), `qwen3:0.6b` (the smaller
fallback) and `qwen2.5:1.5b` (the plan's baseline). All are Q4_K_M, Apache-2.0, from the
Ollama library, and copied into the package as manifests and blobs, so the robot needs
neither `ollama pull` nor `ollama create`.

## Build the runtime package (dev box)

```bash
mkdir -p outputs/llm/downloads && cd outputs/llm/downloads
# the two arm64 files of the pinned release, and their sha256 from the GitHub API
for f in ollama-linux-arm64.tar.zst ollama-linux-arm64-jetpack5.tar.zst; do
  curl -LO https://github.com/ollama/ollama/releases/download/v0.34.4/$f; done
curl -s https://api.github.com/repos/ollama/ollama/releases/tags/v0.34.4 | python3 -c '
import json,sys
for a in json.load(sys.stdin)["assets"]:
    if a["name"].startswith("ollama-linux-arm64"): print(a["digest"][7:], "", a["name"])' > SHA256SUMS
# a dev-box model store with the models pulled (any Ollama of the same version):
OLLAMA_MODELS=$PWD/../ollama_models ollama serve &   # then: ollama pull qwen3:1.7b ...
cd ~/R1process && bash llm/make_runtime.sh            # -> ~/r1_stageC_llm_runtime_<date>.tar.gz
```

The package's `RUNTIME_SHA256SUMS` lists every file. On the robot:
`cd ~/kdw_deploy/llm && sha256sum -c --quiet RUNTIME_SHA256SUMS`.
