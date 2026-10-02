# Stage D · swapping the policy with one command, on the robot (2026-10-02)

Stage D asks two things of `policy_pack/`:
- one command installs a policy bundle into the robot's deploy tree;
- the same command refuses a bad bundle before it changes anything.

The test was a **regression**: the bundle installed was the policy already running
(`2026-08-19_11-03-32_week04_nohead`). Its ONNX, parity fixture, observation/action
contract and gains are byte-identical to what was on the robot, so after the install
nothing should have changed but the engine. How the installer works:
[policy_pack/README.md](../policy_pack/README.md).

**Verdict: accepted.** Every exit criterion of the stage plan is met. One criterion, that
the rebuilt engine has the same file fingerprint, turned out to be wrong and was replaced
by the parity gate (*What the session taught* below).

## Results

| check | result |
|---|---|
| install, 8 steps | all pass: colcon 27.3 s, engine build 25.2 s (FP32, TF32 cleared) |
| numerical parity, old engine → new engine | max_abs **1.144e-05 → 1.144e-05**, worst element the same (sample 472, action 4); gate 1e-3 |
| what changed in the deploy tree (`diff -rq` against a backup) | exactly four entries: `installed.json` (new), `interface/command_envelope.json` (new), `policy_interface.yaml` (comment lines only), `policy_fp32.plan` (rebuilt) |
| `bridge.yaml`, `joint_map.hpp`, `joints.tsv` | byte-identical |
| `installed.json` | run id, ONNX sha256 `f3b04c995f0c…`, bundle manifest hash, plan size and fingerprint |
| `mission_ctl capability` | envelope now read from the installed bundle's `command_envelope.json` |
| three bad bundles through the real installer | **3/3 refused at step 1**, reason named, deploy-tree hash unchanged: `history_length` 5→6; gains missing a joint; an observation term the bridge cannot compute (`base_lin_vel`) |
| stack on the new engine, output off | bridge log shows the trained envelope vx [0, 1], vy 0, wz ±0.5; 50.0 Hz, `crc_fail=0`, no DEGRADED |
| stack on the new engine, output on, gantry, kp 1.3 | stand 3 s → walk 3 s at 0.2 m/s → stand 2 s: **DONE** (sent through `ask`) |

### Inference time, old engine against new

| engine | in the 50 Hz loop, clocks locked | tight loop (`r1_parity_check`, 2000 calls) |
|---|---|---|
| old: 942,776 B, `fnv1a=0x1a174cb54dcdaa5d` | p50 453–456 µs (stage C session, 2026-09-30) | p50 176 µs |
| new: 1,436,476 B, `fnv1a=0x41a49a46311607fb` | **p50 452–455 µs, p99 ≤ 482 µs** | p50 130 µs |

The two engines are the same speed where it matters, in the control loop. The first
output-on run measured p50 ≈ 905 µs because the clocks had not been locked: the guide had
left that step out. Between 50 Hz calls the GPU then drops to a low clock. With clocks
locked the number returned to 452–455 µs. This repeats the W05 lesson that an unlocked
timing number is a lottery.

## What the session taught

1. **A TensorRT engine rebuilt on the Orin is not byte-identical.** The builder picks
   kernels by timing them, so each build can differ. The W06 note "four runs, same
   fingerprint" meant one file run four times, and the same note records an earlier
   rebuild that changed the size, 941,504 → 942,776 B. The plan's criterion "the
   fingerprint reproduces" was therefore wrong.
   - The fingerprint identifies **which file** is running.
   - **Correctness** is shown by parity against the bundle's fixture, which is what the
     installer gates on.
2. **The engine that ran from W07 to stage C had been built on a different device
   model.** TensorRT warned on every load: "Using an engine plan file across different
   models of devices is not recommended". The warning is in every session log since W07,
   and nobody noticed it. The engine built by the installer on this robot loads without
   the warning. That is the point of building on the target, and one more reason to keep
   the new engine.
3. **The rehearsal found two bugs that would have made the installer exit silently on
   the robot.** It ran the whole installer under Python 3.8 against a copy of the
   robot's tree, rebuilt from every package ever sent to it, with colcon and the TensorRT
   executables stood in for:
   - sourcing ROS's setup script under `set -u` killed the script right after the build,
     with no message;
   - the build-failure message claimed nothing had been installed, when steps 3–5 had
     already staged the bundle.

   Both were fixed before the session, and neither appeared on the robot.
4. **The step-by-step guide itself had errors, all found at the robot:**
   - the fingerprint criterion above;
   - a `run … --log` command (only `ask` takes `--log`);
   - no clock lock before timing;
   - a rollback written as `rm -rf deploy && mv backup deploy`, which deletes the tree
     if the backup is missing.

   The rollback is now a guarded `mv` that does nothing without a usable backup. The
   lesson: run every command in a guide verbatim before it goes to the robot.

## What this does not show

- Only the policy that was already running has been installed. No new policy has been
  trained, so a *different* policy has not yet gone through this path. That is the next
  retrain's first use of it.
- A policy that needs an observation term the bridge does not compute still needs C++
  changes. Such a bundle is refused by name (case 3 above), not deployed.

## Data

`outputs/stageD/` on the dev box (not in git): the install, parity and negative-case
logs, `installed.json`, the stack logs and the `ask` record.
