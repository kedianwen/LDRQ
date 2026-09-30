# Stage B: closed-loop turns and the demo sequence on the real robot

**Recorded 2026-09-29.** Setup: kp_scale 1.3, `turn_min_wz` 0.15, stage B package c,
developer mode, and the FP32 TensorRT engine on the onboard Orin NX. The gantry stayed
attached but slack, and every run started from a floor cross under the gantry point.
The terminal reports are archived in `outputs/stageB/2026-09-29/`; `outputs/` is not in
git.

Turns are closed on IMU yaw at the default 0.4 rad/s. `expected_s` is the executor's
estimate for a robot that tracks the command exactly, ramp and taper included. A turn
times out at 2× `expected_s`.

| run | sequence | result | first turn: actual / expected | second turn: actual / expected | IMU error (°) |
|---|---|---|---|---|---|
| smoke 90 | `turn left 90; turn right 90` | DONE | 8.0 / 5.25 s = **1.52** | 5.9 / 5.25 s = 1.12 | +0.2, −1.1 |
| smoke 180 #1 | `turn left 180; turn right 180` | DONE | 13.9 / 9.17 s = **1.52** | 10.6 / 9.17 s = 1.16 | +0.6, −0.4 |
| smoke 180 #2 | same | DONE | 14.4 / 9.17 s = **1.57** | 11.1 / 9.17 s = 1.21 | +0.1, −0.2 |
| demo #1, **rope went taut** | `walk 2s@0.2; turn left 180; walk 2s@0.2; turn right 180` | DONE | 13.3 / 9.17 s = 1.45 | 11.0 / 9.17 s = 1.20 | +0.3, −1.6 |
| demo #2, rope slack | same | DONE | 10.6 / 9.17 s = 1.16 | 10.7 / 9.17 s = 1.17 | +0.1, −0.1 |

For the demo rows, the "first" and "second" turns are the sequence's two 180° turns.

## What it shows

- **Every plan finished DONE: 10 closed-loop turns, 0 timeouts, 0 aborts.** With
  `turn_min_wz` set, turning is usable at the demo gain. None of stage A's gantry stalls
  at 170°–177° happened.
- **Every turn stopped within 1.6° of its target by the IMU,** inside the 3° tolerance.
  This is the IMU's own reading, not ground truth; by decision, the angle is not checked
  against the floor.
- **Timing depends on which turn it was.**
  - Most turns, including both turns of the untaut demo, took **1.12–1.21×** the
    estimate. That fits the robot turning at about 0.8 of the command
    ([stageB_turn_response.md](stageB_turn_response.md)).
  - The **first turn of each smoke run took 1.52–1.57×**. So did the first turn of the
    demo in which the rope went taut (1.45×).
  - Three explanations fit, and these runs cannot separate them:
    - the first turn always went left;
    - it started from standing after the robot was repositioned;
    - the rope was disturbed during repositioning.
  - The worst case, 1.57×, still leaves 27% margin to the 2× timeout. The task needs
    neither precise angle nor precise time, so this is recorded rather than chased.
- **Rope pull shows up as time, not as failure.** The taut-rope demo was slower on its
  first turn (1.45× vs 1.16×) and still finished.

## Abort paths (2026-09-30, kp 1.3, on the spot)

The operator's record is archived as `outputs/stageB/2026-09-29/abort_paths_record_0930.md`.

| path | how | observed | verdict |
|---|---|---|---|
| Ctrl-C | `turn left 8s@0.3`, Ctrl-C mid-turn; `ros2 topic echo /r1_hw_bridge/cmd_vel` in another terminal | last message `[0, 0, 0]`; the robot stopped | **pass**. This is the first confirmation under foxy that the node's own SIGINT handler sends a zero |
| `kill -9` | `turn right 8s@0.3`, `pkill -9 -f r1_mission_cli` mid-turn | last message `[0, 0, −0.3]`, as expected, since the process could send nothing; the robot **stopped turning but kept stepping in place** | **pass**. The bridge's deadman replaces a command older than 500 ms with zero in the policy's observation, so the policy sees exactly the same zero command as after Ctrl-C. Stepping in place is a policy behaviour under a zero command (standing was only 2% of training), not a failure of the abort path. The record does not say whether the robot also stepped after Ctrl-C |
| DEGRADED, first try | stack started with `min_control_rate_hz:=55` | the bridge died at start-up: `parameter 'min_control_rate_hz' has invalid type: expected [double] got [integer]`. mission_ctl aborted after 10 s with "no ~/status received", which is the no-stack path | **not a test of DEGRADED.** The cause was the integer launch argument (see [deploy/README](../deploy/README.md#degrade-behaviour)) |
| DEGRADED, redone | `min_control_rate_hz:=55.0`, `enable_output:=false`, robot hanging on a taut gantry; waited for the bridge's `DEGRADED: control rate below threshold`, then `stand 5s` | `ABORTED` immediately: `bridge is DEGRADED -- clear it (~/resume) first` | **pass** |
| no stack | `stand 5s` with no stack running | `ABORTED` after 10 s: "never saw the bridge RUNNING within 10 s: no ~/status received at all -- is the stack up, and is this terminal on the same ROS_DOMAIN_ID ...", with nothing executed | **pass** |

The DEGRADED run was not a waste. It showed that the whole stack does **not** go down
when the bridge dies: the policy node kept running with nothing to drive. It also showed
that mission_ctl then refused to start, and said why.

## Stage B: closed (2026-09-30)

| exit criterion | result |
|---|---|
| turn response and gain choice | on the spot, 0.15–0.5 rad/s turn at ~0.8 of the command at kp 1.0/1.2/1.3; demo gain **1.3**; `turn_min_wz` **0.15** ([stageB_turn_response.md](stageB_turn_response.md)) |
| closed-loop turns | 10 turns (90°, 180°): all DONE, 0 timeouts, within 1.6° by the IMU |
| demo sequence | `walk 2s@0.2; turn left 180; walk 2s@0.2; turn right 180`: DONE ×2 (the rope went taut once) |
| abort paths | Ctrl-C, `kill -9`, DEGRADED and no stack: **4/4 pass** |
| turn angle vs ground truth, heading drift, ground speed | **not measured, by decision.** The capability text says so |
| video | **deferred** to the end of the project, to be recorded untethered with a helper |

Everything was run with the slack gantry attached and the robot kept on the gantry point.
The PG-2 run in [pg2_untethered.md](pg2_untethered.md) was untethered.
