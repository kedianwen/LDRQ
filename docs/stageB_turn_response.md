# Stage B: how fast the real robot turns for a given wz command

**Recorded 2026-09-29** on the robot, turning on the spot (vx = 0). For safety the
gantry stayed attached but slack in all runs: it carries no load under the gantry
point, and pulls once the robot has moved away from it. The runs were in developer mode, with the FP32 TensorRT engine on the onboard Orin NX. There is
one recording per gain, each made by `deploy/tools/probe_cpp/turn_response.py --record`.
Each recording is open-loop, constant-rate turning at wz = 0.15 / 0.25 / 0.35 / 0.5 rad/s:
6 s left, then 6 s right, at each rate, with 2 s of standing between turns.

- **Analysis window:** the first 1.5 s of each segment is skipped, which leaves about 4.1 s.
- **Yaw rate:** the unwrapped IMU yaw change divided by time. It agrees with the gyro-z
  mean to within 0.03 rad/s in every segment.
- **Bridge state:** RUNNING throughout, with `crc_fail = 0`.

| recording | kp_scale |
|---|---|
| `turn_0929_135939.json` | 1.0 |
| `turn_0929_140239.json` | 1.2 |
| `turn_0929_140618.json` | 1.3 |

## Measured / commanded yaw rate

The table gives left / right for each gain. **Struck-through** segments were disturbed, as
explained below.

| wz cmd | kp 1.0 | kp 1.2 | kp 1.3 | mean, undisturbed | sim |
|---|---|---|---|---|---|
| 0.15 | 0.74 / 0.68 | ~~−0.33~~ / 1.06 | ~~−0.25~~ / 0.85 | **0.83** (n = 4) | 0.98 |
| 0.25 | 0.79 / ~~0.85~~ | 0.86 / ~~0.59~~ | 0.83 / ~~0.59~~ | **0.83** (n = 3) | — |
| 0.35 | 0.76 / 0.75 | 0.80 / 0.84 | 0.80 / 0.81 | **0.79** (n = 6) | — |
| 0.50 | 0.76 / ~~0.78~~ | 0.73 / 0.76 | 0.80 / 0.82 | **0.78** (n = 5) | 1.01 |

Torso tilt over the whole grid:

| kp_scale | median | p99 | max |
|---|---|---|---|
| 1.0 | 3.5° | 16.2° | 19.9° |
| 1.2 | 1.9° | 10.0° | 12.7° |
| 1.3 | 2.2° | 9.3° | 10.9° |

## What it shows

1. **On the spot there is no deadband.** Every rate tested, from 0.15 to 0.5 rad/s, turns
   at about 0.8 of the command, at every gain. The simulator turns at about 1.0. Some of the shortfall may be drag
   from the slack rope; it cannot be separated here, and the task does not need it to be.
   The closed-loop turn stops on the IMU heading, so its angle is unaffected; it only
   takes about 1.25× as long.
2. **Six of the 24 segments were disturbed by a lurch.** Here, disturbed means the torso
   tilted past 8° or the robot yawed the wrong way at more than 0.1 rad/s for half a
   second. Undisturbed turning stays under about 5.5°.
   - **What a lurch looks like:** the robot turns normally, then within half a second the
     tilt jumps from about 2° to 10–20° and the yaw kicks the other way at 0.4–0.55 rad/s
     for about a second. Then it recovers. There was no fall and no DEGRADED.
   - **When lurches happened:** at several rates, and two per gain. The right-hand 0.25
     segment was hit in all three recordings, at the same point in the sequence.
   - **Cause:** by the operator's account, at least some lurches were the slack gantry
     pulling. Rotating on the spot walks the feet off the gantry point, and the rope
     tightens. The repeat at the same point in the sequence fits this: by then the
     robot has turned for several segments. They are a test-rig effect, not the robot's
     response to the command.
   - **Why the earlier reading was wrong:** a lurch in the slowest segment made 0.15 look
     like it turns backwards. That was the first reading of this data (and briefly set
     `turn_min_wz: 0.25`), and it was wrong. `turn_response.py` now flags these segments
     and leaves them out of the summary.
3. **The PG-2 "no turn" happened while walking** (vx 0.1 with wz 0.15), not on the spot. On
   the spot, the same 0.15 turns at 0.83. So a small yaw rate is lost while walking. Arcs
   were not measured further: mission_ctl has no walk-and-turn primitive, and the
   capability statement now says "I do not turn while walking".
4. **Stage A's two turn stalls (170° and 177°) happened on the gantry.** On-the-spot data
   now shows the robot does turn at the taper floor rate (0.14–0.15 rad/s). That points to
   rope torsion near 180° as the likelier cause, not a deadband. The closed-loop smoke
   test in stage B, which keeps the robot on the gantry point and pairs left with right
   turns, checks this directly.

## What was set

- **`turn_min_wz: 0.15`** in `mission_ctl/config/mission.yaml`. This is the lowest rate
  tested, so the true edge is at or below it.
  - With `cruise_wz = 0.4`, the taper floor moves from 0.14 to 0.15. In practice that
    changes almost nothing.
  - The real change is that a closed-loop turn requested below 0.15, a rate never measured,
    is now refused.
  - `compile_plan` now also rejects a `turn_min_wz` above the trained |wz| limit. On the
    robot, a gain (1.20) was typed into this field, which would have refused every
    closed-loop turn.
- **Demo gain: kp_scale 1.3.**
  - Turning response is the same at 1.2 and 1.3.
  - Tilt is slightly lower at 1.3 (p99 9.3° vs 10.0°). Both are well below 1.0.
  - 1.3 is also nearer the middle of the real stability domain, 1.10–1.50.
  - The margin over 1.2 is small, so this is a choice rather than a clear win.

## Not measured, by decision (2026-09-29)

The lab has no room, no props and no second person. That rules out ground-truth turn
angle (photos), straight-line heading drift, and ground speed. The task needs neither
angle nor speed to any precision, and motion-control availability is already shown. So
each is stated in the capability text and the report, not left as a gap:

- **Turn angle:** reported as the **IMU reading**. The capability text says "as my IMU
  reads it (not checked against the floor)".
- **Ground speed:** **taken to equal the command** (`v_cal: commanded`), and reported as
  "not measured".
- **Heading drift while walking:** not measured. Walking does not hold its heading.
