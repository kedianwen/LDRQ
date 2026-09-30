# mission_ctl · driving the robot by time, angle and speed

Uses the bridge's **existing** topics (publishes `~/cmd_vel`, reads `~/imu` and
`~/status`) to turn "walk 5 seconds, turn left 90 degrees, walk 10 metres" into one
command. **The bridge does not change by a single line.** The core has no model in it;
`ask` (stage C) puts an English front end on the same compiler, the same checks and the
same executor.

```bash
python3 r1_mission_cli.py capability                      # English capability statement + JSON schemas
python3 r1_mission_cli.py capability --json               # the same, machine-readable (incl. the LLM prompt id)
python3 r1_mission_cli.py walk --seconds 5 --speed 0.4
python3 r1_mission_cli.py turn --deg 90 --left            # closed on the IMU heading
python3 r1_mission_cli.py run "walk 5s@0.4; turn left 90; walk 10m@0.4"
python3 r1_mission_cli.py run "..." --dry-run             # simulate and print the command trace; never touches the robot
python3 r1_mission_cli.py json --file plan.json           # a JSON plan
python3 r1_mission_cli.py ask "walk 2 meters, then turn left"   # English, parsed by a local model (stage C)
python3 r1_mission_cli.py ask-check                       # is the model up and answering right? moves nothing
python3 r1_mission_cli.py ask-check --once                # one request, exactly as `ask` sends it (the health check)
```

> **Status (2026-09-28): on the real robot.** 68/68 on the robot's Python 3.8.10;
> `stand 3s` and `walk 3s@0.2` both `DONE`; it drove all ten recordings of the `kp_scale`
> sweep. Two things were changed after that session (see *First contact with the robot*);
> the suite is now 101 tests. Still to be accepted on their own: turn angle against ground
> truth, and whether Ctrl-C sends a zero under foxy (stage B).

> **Stage C (2026-09-30): `ask` works on the robot.** Evaluated offline first. The same
> day on the Orin, the held-out sets scored the same as on the dev box, and 11 English
> instructions ran end to end. All of them executed or were refused as asked. The session
> exposed one bug here: a Ctrl-C'd step recorded no running time. It is fixed, and the
> suites are now 106 + 132 tests. See *`ask`* below and
> [docs/stageC_nl_eval.md](../docs/stageC_nl_eval.md).

## Why the interface is Python, not bash, not `ros2 param set`

Chosen by what the task needs to be **able** to do, not by convenience:

| option | keeps publishing cmd_vel for the deadman | reads `~/imu` to close the turn | aborts when the bridge goes DEGRADED | chains primitives |
|---|---|---|---|---|
| `ros2 topic pub -r 10` | yes | **no** | **no** | **no** (stopping means killing it) |
| `ros2 param set` | no (a parameter is not a command channel) | no | no | no |
| a bash script around the two | barely | **no** | **no** | barely |
| **a Python (rclpy) node** | yes | yes | yes | yes |

The middle columns are hard requirements: **a closed-loop turn has to subscribe to
`~/imu`**, and `ros2 topic pub` can only publish. So the executor must be a node. **The
operator interface is still a CLI** — in the field, efficiency means one line and no file
edits:

- a one-off action: `walk --seconds 5`
- a repeatable sequence: `run "walk 5s@0.4; turn left 90; walk 10m"` or `run --file demo.mission`
- before going near the robot: the same command with `--dry-run` prints the whole trace on the dev box

The CLI's script string and the model's transcription (via `nl.translate`) **compile to
the same primitive list** (`plan.compile_plan`), so the LLM layer is only a new front end: the executor and every
check are reused unchanged.

## One call = one process = one plan (no resident daemon)

This is a safety property, not a simplification: the bridge's `cmd_vel_timeout_ms=500`
decays a stale command to zero, so if the mission process **exits, crashes or is killed,
the robot stops** — it does not depend on anyone remembering to stop it. At start-up the
node also counts the publishers on `~/cmd_vel` and refuses to become a second writer (two
schedulers fighting over one topic look, from the robot's side, exactly like "the policy
cannot follow the command").

## The two primitives are not symmetric (the core of this directory)

| | closed loop? | basis | accuracy |
|---|---|---|---|
| `turn <deg>` | **yes** | the bridge already publishes `[qw,qx,qy,qz,...]` on `~/imu` at 50 Hz; `yaw = atan2(2(wz+xy), 1−2(y²+z²))`, unwrapped | the controller **aims at the target angle**, not at the tolerance band (which undershot systematically to 87°). Residual < 1° at 10 Hz; the 3° tolerance only triggers a warning |
| `walk <m>` | **no** | there is no base linear velocity in the 425-dim observation and no odometry topic anywhere | only `time = distance / speed`; the speed is below |

`v_cal` in `mission.yaml` takes three values, each meaning something different:

| `v_cal:` | distance commands | how a distance is reported |
|---|---|---|
| **`commanded` (current default, decided 2026-09-28)** | allowed, converted at the **commanded** speed | `~10 m if speed = command; not measured`, **no error bar** — none was measured, so none is invented |
| `0.35` (a measured number) | allowed, converted at the measured speed | `10 ± 1.5 m` (`v_cal_rel_err`) |
| `unmeasured` / absent | **refused** | — |

Why `commanded`: there was no venue for a speed test, and the task needs distance only to
the level of "a few metres". **Absent still means refuse**: the config has to state the
assumption, or "10 metres" becomes a number that looks measured. The capability statement
tells the LLM the same: "my real walking speed has not been measured". If the speed is
ever measured, put the number in and nothing else changes (`deploy/tools/probe_cpp/vcal.py`
is kept for that).

## Two limits set by the training distribution

- **`min_primitive_s: 2.0`** — training stepped the velocity command only every 10 s
  (`resampling_time_range=(10.0, 10.0)`). A step is in distribution, but a chain of
  half-second primitives is a transient the policy rarely saw. 2 s is a reasoned floor.
- **`ramp_s: 0.5`** — for the same reason; a ramp costs nothing and keeps the first
  control cycles of each primitive out of the sharpest transient.
- Also: standing was only `rel_standing_envs=0.02` of training, so `stand` is weakly
  trained. Measure how long it holds; do not assume.

## Checks and budgets: a plan passes whole or is refused whole

`compile_plan` does every refusal **before the first cmd_vel**: the envelope
(`vx ∈ [0, 1]`, `wz ∈ [−0.5, 0.5]`, the pinned `vy` axis **refused, not clipped**), the
primitive-length floor, and the total time / distance / primitive-count budgets. The
envelope is read from `$R1_DEPLOY_ROOT/interface/command_envelope.json` (installed by
policy_pack), falling back to the bridge's `bridge.yaml` — **the same numbers the bridge
clamps to**, so the "I cannot move sideways" printed by `capability` cannot drift from
the envelope actually enforced.

## Tests

```bash
python3 tests/test_core.py      # 106 tests; no ROS, no pytest, no robot
python3 tests/test_nl.py        # 132 tests: normalize, translate, provenance, replies, and
                                # `ask` end to end against a stand-in model server on 127.0.0.1
python3 eval/run_nl_eval.py --check --set all   # the eval sets agree with the shipped config
```

Not depending on pytest is deliberate: the robot's apt is broken and its Python is 3.8,
and a test suite that cannot run on the machine that matters is not a test suite.

Writing the suite caught four real bugs that are hard to see by reading: `Output` was
missing an argument on primitive transitions (**every multi-primitive plan crashed**);
the settle timer was reset every cycle (**no plan ever reached DONE**); the last
primitive's `actual_s` was rewritten during settle (1 s too long); and `stop` had no
duration, so **a plan ending in `stop` never finished**.

**A stand-in bridge caught three more that unit tests cannot** (they always pass state
`RUNNING` and never touch the ROS side):
- `~/imu` was subscribed with the default RELIABLE QoS while the bridge publishes
  BEST_EFFORT — the two **never connect**, so every closed-loop turn on the robot would
  have aborted with "no IMU yaw".
- The node started at `UNKNOWN` and ticked after 0.1 s, and the executor treated any
  non-RUNNING state as an abort — unless a status message happened to arrive within
  0.1 s, **every real run would abort immediately**; and with no status ever arriving it
  waited forever.
- From humble on, rclpy shuts its context on SIGINT, so "send zero on exit" silently
  failed: after Ctrl-C the bridge's last message was still `vx=0.30`, and only the 500 ms
  deadman stopped it.

Now: subscriptions match the bridge's QoS; no status means wait, with a 10 s timeout that
names the likely cause; status going silent for 3 s mid-run aborts; the node takes SIGINT
itself and the last message after Ctrl-C was measured to be zero. **Confirmed on the robot
under foxy (2026-09-30): after Ctrl-C the last `~/cmd_vel` was `[0, 0, 0]`.**

The capability statement is **English by default** (the English front end's refusals are
built from the same envelope numbers); `describe("zh")` is kept for operators. The model
itself is deliberately *not* given it: it transcribes, and the robot judges.

**Stage C on the robot (2026-09-30) found one more:** a Ctrl-C'd step recorded
`actual_s` 0.0.
- `live_run` passed 0.0 as the time of the interrupt, so the step's running time came
  out as zero, or negative when it was not the first step.
- The two interrupted turns still reported correctly, because a turn is reported by its
  IMU angle. An interrupted walk would have replied "Walked forward for 0.0 s ...:
  about 0.0 m".
- `Executor.interrupt()` now takes the interrupt on the executor's own clock (the last
  `step()` when none is given) and never records a negative time.

**First contact with the robot (2026-09-28, the `kp_scale` sweep, all on a slack gantry)
exposed two more:**
- **The turn timeout had the wrong base.** It was `2 × angle / rate`, but the executor
  slows over the last 25° (to no less than 0.35 of the rate), so 180° at 0.4 rad/s takes
  9.25 s, not 7.85 s, and the "2×" margin was really 1.7×. Four of ten recordings timed out
  on the first turn (at 108°–177°). The timeout and the plan total now use
  `plan.expected_turn_s()` (ramp and taper included; within 0.1 s of the executor
  simulated cycle by cycle).
- **Turning on a gantry does not measure the robot's turning.** Completed turns took
  1.1–1.4× the ideal; after the first 180°, a 5 s straight walk yawed back by 6°–61°, and
  at kp = 0.9 the second left turn actually rotated 48° to the right — rope torsion, not
  the gain (the same kp = 1.10 recorded twice gave one completed turn and one timeout).
  Turn angle was later left to the IMU reading, not checked against the floor (decided
  2026-09-29: the task does not need it).

**Turning, measured on the real robot (2026-09-29).**
- **While walking:** untethered, vx 0.1 with a 0.15 rad/s turn command did not turn in 80 s
  (the PG-2 run). The simulator follows the same command at 1.03.
- **On the spot:** measured with `deploy/tools/probe_cpp/turn_response.py`, gantry attached but slack, at
  kp 1.0, 1.2 and 1.3 ([docs/stageB_turn_response.md](../docs/stageB_turn_response.md)).
  Every rate from 0.15 to 0.5 turns at about 0.8 of the command. 6 of 24 segments had a
  sudden lurch (at least some were the slack rope pulling once the robot had moved off the
  gantry point), and the tool flags those and leaves them out.

What follows from this:
- `capability` says the robot turns **on the spot only**, and gives the angle **as the IMU
  reads it**. Turn angle has not been checked against the floor, deliberately: the task
  does not need it.
- **`turn_min_wz: 0.15`**, the lowest rate tested. The taper never commands less than
  this, and a closed-loop turn requested below it is refused.
- A `turn_min_wz` above the trained |wz| is a config error that refuses every plan. The
  reason: a gain (1.20) was once typed into this field, and every turn was then refused
  one at a time.

## `ask`: English instructions (stage C)

```bash
python3 r1_mission_cli.py ask "Walk forward 10 feet, then turn around." --log ~/orin_commissioning/ask_log.jsonl
```

```
you said:   Walk forward 10 feet, then turn around.
read as:    walk forward 3.05 m, then turn 180 degrees.
            (10 feet -> 3.05 m)
model:      qwen3:1.7b via ollama, 2.1 s (36 tokens, ...)
understood:
  1. walk forward 3.05 m
  2. turn right 180 deg
plan (2 primitives, ~16.8s, ~3.0m):  ... the same display and [y/N] as `run` ...
robot:
1. Walked forward for 7.6 s at 0.40 m/s: about 3.0 m (time x commanded speed; the speed was not measured and there is no odometry).
2. Turned right 180.2° by my IMU (target 180°; not checked against the floor).
Done.
```

**The model transcribes and the robot judges** (`r1_mission/nl.py`; results and method
in [docs/stageC_nl_eval.md](../docs/stageC_nl_eval.md)):

1. `normalize()` turns number words and units into digits in m, seconds, degrees, m/s
   and rad/s before the model sees anything. It also gives a plain "turn left" its 90°
   (never when the sentence is vague) and writes "N times: …" out in full.
2. The model writes the steps down in a JSON schema whose vocabulary includes what the
   robot cannot do (backward, sideways, `unsupported`). It is not told the robot's
   limits.
3. Deterministic checks decide:
   - a distance, duration or angle that was not said, with its unit, is dropped, and
     the robot asks for a number;
   - every refusal and its reason come from the deployed envelope;
   - a step under 2 s is slowed to last 2 s (never below `turn_min_wz`, or 0.1 m/s for
     a walk);
   - `compile_plan` runs unchanged;
   - **one refused step refuses the plan.**
4. The operator confirms the **parsed plan**, and `ask` has no `--yes`. The reply is a
   template, never model prose: the angle is "by my IMU" and the distance is "time x
   commanded speed".

Exit codes: 0 done, 1 aborted, 2 refused or declined, 3 model unreachable. The model
server and its settings are in `mission.yaml` (`llm_url`, `llm_model`, `llm_num_gpu`,
...) and `../llm/`. A bare "stop" never calls the model. Stopping a running plan is
still Ctrl-C or the handheld, as in stage B.

**Each `ask` stands alone: there is no memory of the previous one.**
- Say every command in full, with its direction and its amount. Never "again", "back",
  "undo" or "the other way".
- Tried on the dev box, 11 of 12 such phrases were refused. The model copies a plan out of
  its prompt's examples, and those numbers are dropped because nobody said them.
- The exception was "Turn the other way 90 degrees.", which executes as a right turn: the
  amount was said, but the direction was guessed. Only reading the plan before `[y/N]`
  catches that.
- To come back from an interrupted turn, turn the other way by the angle the reply
  reported, not by the angle first asked for.

`ask-check --once` is one request shaped exactly like an `ask`: the same model, placement
(`llm_num_gpu`, `llm_num_thread`) and prompt. It is the health check in
`llm/ollama_ctl.sh`. Ollama reloads the model when a request asks for another placement,
so anything that talks to the server has to send the same options as `ask`.

## Before using it on the robot

The bridge must be `RUNNING` (the executor does not start the first primitive while it is
`WAITING_POLICY`); the handheld must be in **developer mode**, or the factory motion
service keeps writing `rt/lowcmd` at 500 Hz; before `enable_output:=true`, run the plan
with `--dry-run`, then once with the interactive confirmation rather than `--yes`.
