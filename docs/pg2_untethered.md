# Gate PG-2 (60 s of continuous walking on the real robot): the evidence, without the gantry

**Recorded 2026-09-29** on the robot, `kp_scale = 1.0` (the trained gains), developer mode,
FP32 TensorRT engine on the onboard Orin NX. Command: `ros2 topic pub -r 10
/r1_hw_bridge/cmd_vel ... [0.1, 0.0, 0.15]` — a constant vx = 0.1 m/s, wz = 0.15 rad/s.
Recorded with `deploy/tools/probe_cpp/walk_metrics.py` for 90 s (raw file ~17 MB, not in git).

| measure | value |
|---|---|
| continuous walking under command | **≥ 79.7 s** (from 10.3 s to the end of the 90 s window, still walking when it ended — a lower bound) |
| bridge state | RUNNING for all 90 s; no DEGRADED; `crc_fail = 0` |
| torso tilt | median 7–10° (pitch; standing 7.2°), p99 20.4°, max 28.0° (at 30–40 s; brief excursions to ~21° at 40–60 s; 1.06 s in total above 20°) |
| leg tracking rms | 0.098 rad |
| peak torque | 83 % of rating (left hip roll; a spike) |
| action chatter | 5.95 Hz, same band as every gantry recording |

**PG-2 (60 s continuous walking on the real robot) is met at the target line**, by the
survival criterion it was defined with. The stage A stability definition (tilt ≤ 20°) is
a stricter, different test and is not what PG-2 asks; this run would not pass it — the
same marginal tilt the gantry sweep found at kp = 1.0.

**Unexpected, and the most useful number in this file: the robot did not turn.** Over the
78 s of steady command it turned 12° in total — a yaw rate of 0.001 rad/s against a
commanded 0.15 (gyro z mean −0.003). The same deployed policy in simulation turns at
0.98–1.06 of every command from 0.15 to 0.5 rad/s, including exactly this (0.1, 0.15)
at 1.03. So the real robot has a turning deadband the simulator does not: small yaw-rate
commands are lost. It also explains part of stage A: two of its four turn timeouts
stalled at 170° and 177°, inside the last 25° where the closed-loop turn tapers its rate
down to 0.14 rad/s. Measuring the response curve is now the first item of stage B
(`deploy/tools/probe_cpp/turn_response.py`), and `mission_ctl` gained a `turn_min_wz`
setting to keep the taper above whatever it finds.

*Follow-up (same day, [stageB_turn_response.md](stageB_turn_response.md)):* the
"deadband" above is real **while walking**, not on the spot. Turning on the spot
(gantry attached, slack), every rate from 0.15 to 0.5 rad/s turns at about 0.8 of the command, at kp 1.0,
1.2 and 1.3. The two stage A stalls were on the gantry, so rope torsion is the likelier
cause. `turn_min_wz` is 0.15, and the capability statement says the robot turns on the
spot only.

**Video: deferred.** The recording from this run is not usable as evidence (limited space,
no second person); a clean take is scheduled for the end of the project, with a helper.
