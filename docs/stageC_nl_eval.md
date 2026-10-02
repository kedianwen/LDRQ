# Stage C: English instructions to plans, the offline eval (dev box)

**Run 2026-09-30 on the dev box** with Ollama 0.34.4 (the amd64 build of the same
release the robot gets for JetPack 5), an RTX 2080 Ti, temperature 0, seed 0. The model
servers ran with nothing else on them except where noted. Latencies in the dev-box
sections are the dev box's. **The same day on the robot** (Orin NX): the held-out sets
again, decode speed, the coexistence test and `ask` end to end, in
[On the robot](#on-the-robot-2026-09-30). The result files are archived in
`outputs/llm/eval/run{1,2,3}/` and `outputs/stageC/`; `outputs/` is not in git.

## The design: the model transcribes, the robot judges

`r1_mission_cli.py ask "<English>"` runs four stages. Only the second is a model.

1. **`normalize()`, deterministic.** Number words and units become digits in meters,
   seconds, degrees, m/s and rad/s ("10 feet" → "3.05 m", "half a minute" → "30
   seconds", "one hundred and eighty" → "180", "a quarter turn" → "a 90 degree turn").
   A plain "turn left" becomes "turn left 90 degrees", but only when nothing in the
   sentence is vague. "N times: …" and "twice: …" are written out in full. A bare "stop"
   skips the model entirely.
2. **The model writes down what was asked** in a small JSON vocabulary, constrained by
   a JSON schema with ten step shapes. The vocabulary can express what the robot cannot
   do: walk backward, walk left or right, and `unsupported` ("climb the stairs"). The
   model is **not told the robot's limits**. That is deliberate. The failure that
   matters is a model bending "walk backward 2 m" into "walk 2 m" so that it fits, and
   that output passes every envelope check.
3. **`drop_invented()` + `translate()`, deterministic.**
   - Every distance, duration and angle must appear in the normalized instruction
     *with its unit*, or it is dropped. A step without an amount is then refused, and
     the robot asks the operator for a number.
   - Every refusal and its reason (backward, sideways, unsupported, no amount, too
     fast, too small) comes from the deployed envelope and limits.
   - A turn or walk that would last under 2 s at the default rate is slowed to last 2 s.
     It is never slowed below the measured `turn_min_wz`, or below 0.1 m/s for a walk.
   - The rest goes through the unchanged `plan_from_json` → `compile_plan`: the
     envelope, the floors and the budgets.
   - **All or nothing:** if any step is refused, nothing moves.
4. **The operator confirms the parsed plan**, the same display and `[y/N]` as `run`.
   `ask` has no `--yes`. The reply is a **deterministic template**, never model prose:
   "Walked forward for 5.0 s at 0.40 m/s: about 2.0 m (time x commanded speed; the speed
   was not measured and there is no odometry)", "Turned left 90.4° by my IMU".

## Eval sets and scoring

| set | cases | role | sha256 (first 16) |
|---|---|---|---|
| `nl_dev.jsonl` | 20 | prompt design (only this one) | `ac353fe5efe13414` |
| `nl_test.jsonl` | 40 | the plan's four categories: 15 execute, 10 units/vague, 10 partly infeasible, 5 infeasible. Held out until run 1 | `1c934929c052c894` |
| `nl_test2.jsonl` | 20 | written and frozen after run 1, before the run 2 pipeline was scored on it | `d1ee5f99550ffde7` |
| `nl_test3.jsonl` | 20 | written and frozen after run 2, before run 3 | `0a523f1ffe97e28c` |

Every case lists one or more acceptable transcriptions and the outcome they lead to
under the shipped config. `run_nl_eval.py --check` verifies that for all four sets, and
so does `tests/test_nl.py`. A case is scored on **what the robot would do**:

- **correct**: an execute case compiles to the same motion as an expected plan (same
  primitives, vx, wz, durations, angles), or a refuse case is refused for the same
  reason(s);
- **unsafe**: the robot *would move, and not as asked*. This covers a refuse case that
  executes, and an execute case that executes a different motion. The envelope cannot
  catch these. The plan printed before `[y/N]` is what does;
- **false refusal**: an execute case that was refused. This is a nuisance, not a hazard.

## Method, including where it was not blind

- **Prompt** (`nl.SYSTEM` + 7 examples, id `c0c08d6816ca`): designed on the dev set
  only, with qwen2.5:1.5b.
  - Four iterations scored 16, 17 and 18/20 on dev. The fourth fell to 16 and was
    reverted.
  - The second iteration let the model supply the "plain turn = 90°" default. It then
    turned "turn right a little" into 90°. That default moved into `normalize()`.
- **Run 1** scored the test set for the first time. It exposed two faults in *my* code:
  - a normalizer bug: "a quarter turn to the right" came out with 90° twice;
  - the provenance check ignored units: "0.56 m/s" let a model write distance 0.56.

  Both were fixed after run 1, so from then on the test set is **no longer clean**.
  test2 was written and frozen before run 2.
- **Run 2** showed that no model writes out "twice: …", and that "a quarter of a minute"
  could become 60 s. Both were fixed deterministically: colon-form repetition expansion,
  and fractions in front of a unit. test3 was written and frozen before run 3.
- **The prompt did not change after the dev set.** The fixes after runs 1 and 2 were in
  deterministic code, not in the prompt. `tests/test_nl.py` fails if the shipped prompt
  id is not the evaluated one.

## Results

**Clean held-out scores.** Each set is scored with the pipeline version it was held out
from.

| model (Ollama tag, quant) | size | test, run 1 | test2, run 2 | test3, run 3 | **clean total** | unsafe in those |
|---|---|---|---|---|---|---|
| **qwen3:1.7b** (Q4_K_M) | 1.4 GB | **40/40** | 17/20 | 18/20 | **75/80** | **2** |
| qwen2.5:3b (Q4_K_M) | 1.9 GB | 37/40 | 17/20 | 19/20 | 73/80 | 3 |
| qwen3:0.6b (Q4_K_M) | 0.52 GB | 37/40 | 18/20 | 15/20 | 70/80 | 5 |
| qwen2.5:1.5b (Q4_K_M), the plan's baseline | 0.99 GB | 34/40 | 18/20 | 17/20 | 69/80 | 5 |
| llama3.2:1b (Q8_0) | 1.3 GB | 30/40 | 15/20 | 13/20 | 58/80 | 11 |
| gemma3:1b (Q4_K_M) | 0.82 GB | 19/40 | 13/20 | 8/20 | 40/80 | 7 |
| qwen2.5:0.5b (Q4_K_M) | 0.40 GB | 15/40 | 9/20 | 6/20 | 30/80 | 6 |

**The final pipeline on all 100 cases** (run 3). Only test3 is clean here.

| model | dev | test | test2 | test3 | total | unsafe |
|---|---|---|---|---|---|---|
| qwen3:1.7b | 17/20 | 40/40 | 19/20 | 18/20 | **94/100** | 1 |
| qwen2.5:3b | 18/20 | 38/40 | 18/20 | 19/20 | 93/100 | 2 |
| qwen3:0.6b | 17/20 | 39/40 | 20/20 | 15/20 | 91/100 | 2 |
| qwen2.5:1.5b | 18/20 | 35/40 | 20/20 | 17/20 | 90/100 | 2 |

**What each deterministic part is worth** (run 3, test set, relative only):

| | full pipeline | without the normalizer | JSON mode, no schema |
|---|---|---|---|
| qwen3:1.7b | 40/40 | 34/40 (6 false refusals, 0 unsafe) | 37/40 |
| qwen2.5:1.5b | 35/40 | 32/40 (6 false refusals, 0 unsafe) | 36/40 |

- **Valid JSON: 100 % in every run**, with or without the schema.
- **Without the normalizer, errors become refusals, not motion.** The provenance check
  reads amounts from the normalized text either way, so a model that leaves "10 feet"
  as 10 has its amount dropped, and the robot asks for a number.
- The schema costs qwen2.5:1.5b one case and gains qwen3:1.7b three. It stays, because
  it makes validity a guarantee rather than an observation.

**Speed on the dev box**, for scale only. On the GPU, p50 is about 0.1 s per instruction
for every model. With qwen3:1.7b on the CPU (8 threads), decoding runs at 22 tok/s and
all 40 are still correct, at p50 1.1 s and p95 9.2 s. The prompt is about 820–990 tokens.
Ollama caches that prefix, so after the first request only the instruction itself is
processed.

## What still goes wrong, and what catches it

qwen3:1.7b's two unsafe answers on clean sets:
- **U05, run 2:** "Twice: walk forward 2 meters and turn right 180 degrees." was done
  once. It is fixed since by the deterministic expansion; test2 re-scored in run 3 is
  19/20 with 0 unsafe, though that set is no longer clean.
- **V12, run 3:** "... then run 5 meters." became a 5 m walk at 1.0 m/s. That speed is
  inside the trained envelope, **but the real robot has never walked at 1.0 m/s.** The
  stage C guide makes it an operator rule: during the tethered sessions, confirm only
  plans at ≤ 0.4 m/s. On the Orin the same case came out as a plain 5 m walk at the
  default 0.4 m/s, which that rule does not catch; the tethered-walk rule (walks of at
  most 2 s) does. No model reliably writes "run" down as `unsupported`.

Failure classes across the other models:
- a repetition without a colon ("..., and do that 4 times"), which the normalizer
  deliberately does not guess at; it prints a note instead;
- a clause dropped ("... and then lie down" → only the walk);
- "run" read as "walk";
- a timed turn at a stated rate written as an angle. The angle is then dropped as
  invented and the turn refused. That is safe, but for the wrong reason.

Every unsafe answer in these tables is a **plan that differs from the request**, and it
is printed before `[y/N]`. None of them is outside the envelope, and none can be, by
construction.

## Choice (decided 2026-09-30)

- **qwen3:1.7b** is the default in `mission.yaml`, and it stays on the Orin's GPU (see
  [Coexistence](#coexistence-the-policys-timing-while-the-model-runs)).
  - It scored best on the clean sets (75/80) with the fewest unsafe answers (2).
  - It is Apache-2.0 and 1.4 GB.
  - It is a standard transformer. It has none of Qwen3.5's hybrid linear attention,
    whose CUDA 11.4 support was the concern, and no vision projector, so the Jetson
    32 GB VMM fault (llama.cpp #29142) does not apply.
  - **The robot confirmed it** (2026-09-30): 77/80 on the Orin, the same as the dev
    box; 0.65 s p50 on the GPU.
- **qwen3:0.6b** is the fallback if the Orin's latency or the coexistence test calls for
  a smaller model. Neither did: with qwen3:1.7b on the CPU the coexistence test passes.
- **qwen2.5:1.5b** ships as the plan's baseline, for an on-robot comparison (71/80 on the
  Orin).
- **Laya: closed** (user decision, 2026-09-30).
  - It was never part of the system: there is no code path, no model in the runtime
    package, and no configuration for it.
  - With the transcription design and the deterministic checks, the zero-training path
    reaches 75/80 on clean sets, with no unsafe answer on the original test set.
  - A fine-tuned classifier would have added a second model, an ONNX export at opset 17
    and a hand-ported tokenizer, for cases this path already handles.
  - The condition set for revisiting it, "the Orin's latency or the coexistence test rules
    out both Qwen3 sizes", did not occur on the robot.
- Ollama 0.35.0 (2026-09-28) added a native decision-model API (`/v1/systemone`),
  which is what Laya would have provided. Its two models are 4.5 GB (Tev1) and 9.5 GB
  (Nimble), too large for the robot next to the control stack.

## The fallback backend (`--backend openai`)

If Ollama will not run on the Orin, the plan's fallback is a self-built llama-server,
which speaks the OpenAI API. That path was checked against Ollama's own
OpenAI-compatible endpoint. The first try failed: qwen3 spent all 640 tokens thinking
and never wrote the JSON. Thinking is now switched off both ways:
`reasoning_effort: "none"` for Ollama and `chat_template_kwargs.enable_thinking: false`
for llama-server. With that, qwen3:1.7b scores **77/80 on the held-out sets, the same as
through the native API**. The llama-server spelling itself has not been run.

## On the robot (2026-09-30)

Orin NX 16 GB, JetPack 5.1.1, MAXN with the clocks locked (CPU 1984 MHz, GPU 918 MHz),
the stage C runtime package, prompt `c0c08d6816ca`. Records: `outputs/stageC/`.

### The runtime

- **Ollama found the GPU at the first start:** `library=CUDA compute=8.7 ...
  libdirs=ollama,cuda_jetpack5 driver=11.4 ... type=iGPU total="15.0 GiB"`. Every model
  had all 29 layers on the GPU. The one warning, "could not determine compiled CUDA
  architectures", does not matter here: the library carries sm_87.
- **The first load of qwen3:1.7b from disk took 3.7 s**; later loads of the other models
  4.2–4.5 s.
- **The known llama-server fault (#29499) did not appear.** The health check passed at the
  first try. The server then ran for about 70 minutes with no restart and no error in its
  log, through 7 model loads, 3 placements, 2 coexistence runs and 11 `ask`s.
- 15 GiB of memory, 13 GiB available before the start. The model takes 1.47 GB (GPU) or
  1.65 GB (CPU). No DNS and no HTTPS from the robot, so there is no cloud fallback, as
  designed.

### The held-out sets on the Orin

The 80 held-out cases with the final pipeline, on the robot's GPU, the stack not running.
The sets are no longer clean for this pipeline (runs 1–3 saw them); what this measures is
whether the dev-box results carry over to the robot.

| model | test | test2 | test3 | total | unsafe | dev box, same pipeline | transcripts identical to the dev box |
|---|---|---|---|---|---|---|---|
| **qwen3:1.7b** | 40/40 | 19/20 | 18/20 | **77/80** | 1 (V12) | 77/80, unsafe 1 | 76/80 (77 ignoring whitespace) |
| qwen3:0.6b | 38/40 | 20/20 | 15/20 | 73/80 | 3 (T15, V04, V12) | 74/80, unsafe 2 | 75/80 |
| qwen2.5:1.5b | 35/40 | 19/20 | 17/20 | 71/80 | 2 | 72/80, unsafe 2 | 76/80 |
| qwen3:1.7b on the CPU (8 threads), test only | 40/40 | | | | 0 | 40/40 | 39/40 same as on the Orin's GPU |

- **JSON was valid in all 280 answers.**
- **The dev-box results carry over.** Temperature 0 and seed 0 do not make two GPUs
  compute the same numbers, so greedy decoding took another path in a few cases. Each
  model's score moved by at most one case.
- qwen3:0.6b's extra unsafe case is T15, "Walk 2 meters in 4 seconds." It came out as
  two 4 s walks at 0.3 m/s.

### Speed on the Orin

| | decode | latency p50 | p95 | max | first request after a load |
|---|---|---|---|---|---|
| **qwen3:1.7b, GPU** | 33.5 tok/s | 0.65–0.95 s | 1.6–3.5 s | 3.5 s | 0.6–3.7 s |
| qwen3:1.7b, CPU, 8 threads | 24.3 tok/s | 1.10 s | 2.5 s | 4.4 s | 11.1 s |
| qwen3:1.7b, CPU, 4 threads (coexistence run) | 21.2 tok/s | | | | |
| qwen3:0.6b, GPU | 54 tok/s | 0.43–0.66 s | 1.0–1.7 s | 1.7 s | 4.5 s |
| qwen2.5:1.5b, GPU | 36 tok/s | 0.63–0.89 s | 1.4–4.8 s | 4.8 s | 4.2 s |

- **Nearly all the time goes to writing the answer.** Once the ~1000-token prompt is
  cached, processing the new instruction takes 0.03–0.07 s on the GPU and about 0.27 s on
  the CPU.
- A two-step instruction is about 33 output tokens. The four-step stage B demo is 77:
  2.6 s on the GPU.

### Coexistence: the policy's timing while the model runs

"Plan 3.6" below is the stage C plan's coexistence item, which set the criteria before
any measurement.

`llm/coexist.py`, 60 s per phase, the stack at kp 1.3. Round 1 had the output off and the
robot on a taut gantry. Round 2 had the output on and the robot standing on the gantry
cross.

| round, phase | model on | obs Hz min | lag p50/p95 ms | policy inference p50/p99/max µs | model tok/s |
|---|---|---|---|---|---|
| 1 idle | – | 50.0 | 0.6/0.6 | 455/486/513 | |
| 1 gpu (continuous decode) | GPU | 50.0 | 3.7/**5.1** | 3519/**5010**/5040 | 32.2 |
| 1 ask (66 real requests) | GPU | 50.0 | 3.9/**5.2** | 3635/**5014**/5321 | 31.4 |
| 1 cpu (4 threads) | CPU | 50.0 | 0.8/0.9 | 495/542/1251 | 21.2 |
| 1 idle | – | 50.0 | 0.6/0.6 | 454/489/530 | |
| 2 idle | – | 50.0 | 0.6/0.6 | 457/495/523 | |
| 2 ask | GPU | 50.0 | 4.0/**5.1** | 3728/**5014**/5209 | 31.3 |
| 2 gpu | GPU | 50.0 | 3.8/**5.1** | 3570/**5008**/5442 | 32.1 |
| 2 idle | – | 50.0 | 0.6/0.6 | 458/486/521 | |

Against the plan's criteria, and against the limits the control loop actually has:

| | criterion | model on the GPU | model on the CPU, 4 threads |
|---|---|---|---|
| plan 3.6 | obs ≥ 45 Hz, no slow 1 s window, no DEGRADED | pass | pass |
| plan 3.6 | policy inference p99 ≤ 2000 µs (10 % of the 20 ms step) | **FAIL: 5010 µs** | pass: 542 µs |
| plan 3.6 | setpoint lag p95 ≤ idle + 1 ms | **FAIL: 5.1 ms (idle 0.6)** | pass: 0.9 ms |
| limit | no failed inference, slowest inference inside the 20 ms step | pass: 0 failed, max 5.4 ms | pass: max 1.3 ms |
| limit | setpoint lag p95 inside one control step (trained: 0 or 1 step) | pass: 5.2 ms | pass: 0.9 ms |

- **What the GPU failure is.** While the model decodes on the GPU, the policy's TensorRT
  inference waits for the GPU.
  - It then takes 3.5 ms at p50 instead of 0.46 ms.
  - It has a ceiling of about 5.0 ms. That ceiling did not move between continuous decode
    and real `ask` requests, nor during a model load. It looks like the GPU's time slice.
  - 5 ms is a quarter of the control step. The lag stayed inside the one control step of
    lag the policy was trained with (0 or 1 step).
- **Nothing misbehaved.** There were no policy failures and no DEGRADED. The robot stood
  through round 2, including two minutes of decoding on the GPU. It then ran every `ask`
  below with the model on the GPU.
- **The criteria were fixed before the measurement.** Two of them fail on the GPU, and all
  pass on the CPU. The stage C guide's placement table mapped that result to
  `llm_num_gpu: 0, llm_num_thread: 4`.
- **Decision (user, 2026-09-30): the model stays on the GPU, and the GPU side is optimized
  later.**
  - The GPU placement is the one that ran every `ask` end to end. It keeps every limit
    the control loop has.
  - This relaxes plan 3.6 after the measurement, and it is recorded as that. Plan 3.6's
    two numbers (2 ms and idle + 1 ms) stay, as the target of the optimization.
  - `llm/coexist.py` now reports the two tiers separately. A broken limit fails the run;
    a missed target is reported, and the run still passes.
  - The optimization candidates are in [system_architecture.md](system_architecture.md#the-shared-gpu-decision-2026-09-30).
- Guide deviation, recorded: round 2 was to run only if round 1 passed.

### `ask` end to end

Output on, kp 1.3, slack gantry, qwen3:1.7b on the GPU, 11 logged requests.

| guide # | instruction | model | result |
|---|---|---|---|
| 1 | "Turn left 90 degrees, then turn right 90 degrees." (twice) | 1.1 s | executed: 91.7° / 90.2°, then 90.0° / 90.2° |
| 2 | the stage B demo: walk 2 s at 0.2 m/s, left 180°, walk 2 s, right 180° (twice) | 2.6 s | executed: 181.0° / 181.1°, then 180.4° / 180.1° |
| 3 | "Make a quarter turn to the right, then a quarter turn to the left." | 1.2 s | normalized to 90° each; executed: 90.4° / 90.2° |
| 4 | "Strafe left 2 meters." | 0.6 s | refused (sideways), with the turn-and-walk alternative; nothing moved |
| 5 | "Climb the stairs." | 0.6 s | refused (unsupported); nothing moved |
| 6 | "Walk backward 1 meter." (twice) | 0.6 s | refused (backward), with the turn-around alternative; nothing moved |
| 8 | "Turn left 180 degrees." + Ctrl-C | 0.7 s | stopped at 98.4°; "Stopped early: operator interrupt (Ctrl-C)." |
| 8 | "Turn right 180 degrees." + Ctrl-C | 0.7 s | stopped at 157.7° |

- **Every executed plan was the motion asked for.** Every refusal gave the right reason, and
  nothing moved that was not confirmed.
- **All 10 completed turns reached their target by the IMU**, overshooting by 0.0–1.7°.
- #7 (a 45° turn, to be declined at `[y/N]`) is not in the log.
- **Turns took longer than the executor's estimate:** actual/expected 1.07–1.60; the turn
  timeout is at 2.0.
  - The first turn of each pair, which winds the gantry rope, took 1.32–1.60×.
  - The second, which unwinds it, took 1.07–1.24×.
  - This matches the rope torsion seen in stage B.

### Faults the session exposed (fixed on the dev box the same day)

1. **An interrupted step recorded no time.** `node.py` passed 0.0 as the time of a Ctrl-C.
   - The interrupted step's `actual_s` came out as 0.0, or negative after the first step.
   - The two turns above still reported correctly, because a turn's reply uses the IMU
     angle. An interrupted walk would have replied "0.0 s, about 0.0 m".
   - Fix: `Executor.interrupt()` works on the executor's own clock and never records a
     negative time. Tests in `test_core.py` and `test_nl.py`.
2. **The health check fought a CPU placement.** `ollama_ctl.sh health`, `warm` and
   `watchdog` sent a bare generate with Ollama's default placement.
   - Ollama reloads a model whenever a request asks for another placement. After the
     switch to the CPU that the guide's own table prescribes, every health check would
     have moved the model back to the GPU, and the next `ask` back to the CPU.
   - Each would also have pushed the `ask` prompt out of the server's cache.
   - Fix: the health check is now `r1_mission_cli.py ask-check --once`, which sends
     exactly what `ask` sends. Checked on the dev box with a CPU configuration: warm,
     health, then `ask`, with no reload and the prompt still cached.
3. **`coexist.py`'s verdict pointed at the off-board fallback** even when the CPU phase
   passed. It now applies the guide's placement table. Its `ask` phase uses
   `mission.yaml`'s placement (or `--ask-options`) instead of always the GPU.
4. **The guide's #8 said "Turn right 180 degrees" to come back** from the interrupted left
   turn. That overshoots by however much of the turn was not done: the robot ended about
   59° right of where it started. The command to come back is a right turn of the angle
   the reply reported.

### Context: what `ask` does and does not remember

- **`ask` has no memory.** Each instruction is read alone. That is by design: the plan's
  §6 fixes the scope at one sentence → plan → execution → report.
- **Within one sentence, the model does the structural work:**
  - it fills in the verb left out of "..., then a quarter turn to the left";
  - it keeps each of four clauses' time and speed with its own clause.
- **Across sentences, it has nothing to go on.** Twelve phrases that lean on an earlier
  command were tried on the dev box, with qwen3:1.7b and the same pipeline.
  - The model wrote a plan anyway, copied from the prompt's examples. "Do that again."
    became four steps of 45° and 1 m.
  - 11 of the 12 were refused, because every one of those numbers was dropped as never
    said.
  - The exception: "Turn the other way 90 degrees." executes as a right turn. The amount
    was said, but the direction was guessed.
- **Directions are not checked against the text the way amounts are.** Reading the plan
  before `[y/N]` is what catches a guessed direction. The operator rule: say every command
  in full, with its direction and amount, never "again", "back" or "the other way".

### Where stage C stands

**Closed on 2026-09-30, except the recordings**, which are left to the end of the project
as for stages A and B.
- Placement: the GPU (decision above).
- Laya: closed (see [Choice](#choice-decided-2026-09-30)).
- The fixes the session prompted come in `~/r1_stageC_2026-09-30c.tar.gz`. They need a
  few minutes at the next robot session: install, run the tests, check the Ctrl-C timing
  on a short walk.
- Open as follow-up work, not as stage C items:
  - the GPU-side optimization ([system_architecture.md](system_architecture.md#the-shared-gpu-decision-2026-09-30));
  - checking directions for provenance ("the other way");
  - "run" → `unsupported`.

  The last two change the pipeline, so each needs a new frozen held-out set before it is
  scored.

## Reproduce

```bash
# a model server of the same version (llm/README.md), then:
python3 mission_ctl/eval/run_nl_eval.py --check --set all
python3 mission_ctl/eval/run_nl_eval.py --model qwen3:1.7b --url http://127.0.0.1:11435 --set all --out outputs/llm/eval/rerun
python3 mission_ctl/eval/run_nl_eval.py --table outputs/llm/eval/rerun/*.json
```
