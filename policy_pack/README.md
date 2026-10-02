# policy_pack: swapping the trained policy with one command

Turns "deploy a different policy" from **three config edits, a C++ header edit and a
rebuild** into **one command**, and makes that command refuse a bad policy before it
changes anything on the robot.

| file | runs on | role |
|---|---|---|
| `make_bundle.py` | dev box | build a bundle from an exported policy and its training config |
| `verify_bundle.py` | anywhere | check a bundle; writes nothing |
| `gen_configs.py` | robot | generate the two ROS configs from a bundle (`--check` previews) |
| `install_bundle.sh` | robot | the eight-step install |
| `tests/run_tests.sh` | anywhere | 12 kinds of bad bundle, each refused for the right reason |
| `tests/negative_installs.sh` | robot (or a copy of `deploy/`) | bad bundles through the real installer: refused at step 1, nothing changed |

## How to use it

```bash
# dev box: after exporting a policy, build a bundle (checked as it is built)
python3 policy_pack/make_bundle.py

# robot: first see what would change, then install
source deploy/env.sh
bash policy_pack/install_bundle.sh bundles/<run_id> --dry-run
cp -a deploy deploy.bak        # a failure after step 3 leaves the bundle's files staged
bash policy_pack/install_bundle.sh bundles/<run_id>
```

The last lines print the engine's path and fingerprint, and `deploy/artifacts/installed.json`
records what is installed. The fingerprint identifies the file and changes with every
build (TensorRT picks kernels by timing them); **parity** is what proves the numbers.

### The eight steps, and what each one guards against

| step | action | guards against |
|---|---|---|
| 1 | `verify_bundle.py` | the refusal list below. **Writes nothing** |
| 2 | `gen_configs.py --check` | changing things the operator has not seen (comment-only changes are labeled as such) |
| 3 | copy the json / onnx / fixture into the deploy tree | — |
| 4 | `gen_joints.py` + `gen_joint_map.py --header=` | a second, independent check: gain order ≠ joint order exits here |
| 5 | `gen_configs.py` writes both yamls | a new policy running inside the previous policy's command envelope |
| 6 | `colcon build` | a generated header that does not compile = a wrong length or type |
| 7 | `r1_build_engine` (reused if the ONNX is unchanged; `--force-engine` rebuilds) | — |
| 8 | `r1_parity_check --tol 1e-3`, then write `installed.json` | an engine that computes wrong numbers on this machine |

`installed.json` records the run_id, the onnx sha256, the bundle manifest hash, and the
plan's size and fingerprint, so "which policy is running on the robot right now" is a fact
you read, not something you have to remember.

## Design

### What a bundle is

```
bundles/<run_id>/
  policy.onnx               the exported policy (not a .plan -- see below)
  parity_fixture.bin        this policy's own numerical anchor
  policy_interface.json     observation/action contract: term names, widths, history;
                            action mapping, scale, default positions
  actuator_gains.json       per-joint kp / kd / torque limit (six groups, not one scalar)
  command_envelope.json     the command range training covered (parsed from the training
                            config, not copied from bridge.yaml)
  provenance.json           run_id, git sha, onnx sha256, build date and host
  MANIFEST.sha256           a hash of every file above
```

**A bundle contains no engine.** A TensorRT `.plan` encodes the TensorRT version, the GPU
architecture and kernel tactics timed on the machine that built it; it is not portable.
So the ONNX travels, the engine is built on the Orin, and it is immediately checked
against **the bundle's own fixture** — a new policy is proven to compute correctly before
it moves a joint.

**A bundle contains no motor-slot map either.** `kJointSlot` is a property of the
**robot**, and it was measured (the vendor enum has `RightShoulderPitch = 19` written as
29; used as a slot it drives head_pitch). Putting it in a per-policy file invites someone
to "fix" it in a yaml. It stays in `joint_map_measured.tsv`.

### Why "regenerate and rebuild" rather than "the bridge reads parameters"

The arrays in `joint_map.hpp` are `constexpr`, so the **compiler** checks their lengths
against `kNumJoints` / `kNumActions` — a wrong length is a build error, not a run-time
surprise on a robot that is already standing. Rebuilding both packages on the Orin takes
**29 s** measured (`r1_hw_bridge` 24.3 s and `r1_policy_runner` 25.3 s, in parallel).
That is cheaper than changing the one node that has been stable since W06.

### What `verify_bundle.py` refuses

`tests/run_tests.sh` actually builds each of these and runs it, and asserts
the **reason** for the refusal, not only the exit code — a validator that refuses for the
wrong reason points people the wrong way.

| bad bundle | why it is dangerous |
|---|---|
| `history_length` disagrees with `total_dim` | otherwise found only after a 25 s engine build, at load time |
| an observation term the bridge **cannot compute** (e.g. `base_lin_vel`) | the bridge assembles six terms; an extra one would be zero-filled at 50 Hz, reads like a sim2real gap, and can take a week to find |
| all six terms present but in a different **order** | every dimension adds up; a completely silent error |
| `actuator_gains.json` missing a joint, or sorted alphabetically | the bridge takes gains by joint index: a different order puts the ankle's kp on the hip |
| a leg with `damping = 0` | the file form of W06's "undamped legs" defect |
| an action index whose name says A but points at B | the knee's command goes to the shoulder, silently |
| an envelope with `lo > hi` | — |
| a json edited without recomputing the manifest | **the accident most likely to actually happen** |
| an extra file in the bundle that is not in the manifest | this is how a hand-edited yaml gets installed |
| the onnx replaced without updating provenance | the engine is no longer the policy on record |

## Tests

```bash
bash tests/run_tests.sh [<bundle>]            # what verify_bundle.py refuses (any machine)
bash tests/negative_installs.sh <good_bundle>  # three bad bundles through the REAL installer:
                                               # each must stop at step 1, name the reason and
                                               # leave $R1_DEPLOY_ROOT byte-for-byte unchanged
```

Both run in `../run_tests.sh` and CI; `negative_installs.sh` there works on a temporary copy
of `deploy/`. On the robot it is part of stage D's acceptance.

## Limits (what this version cannot do)

Only policies that use **the six observation terms the bridge already computes** can be
deployed by configuration. A new policy that adds base linear velocity, foot contact or a
height scan **needs C++ changes** — the correct behavior is to name the term at start-up
and refuse, not to guess. The `BRIDGE_TERMS` table in `verify_bundle.py` is that boundary;
change it together with the bridge's frame-assembly code.

## Known pitfalls

- **Sourcing ROS's setup script under `set -u` exits the shell.** ROS reads unset
  variables (`AMENT_TRACE_SETUP_FILES`, `COLCON_TRACE`). The installer relaxes `set -u`
  around that one line; before the fix it exited right after the build, silently.
- **The `ros2` CLI is unreliable on the robot's network** (a DDS library clash), so the
  installer runs the TensorRT executables by path.
- **An engine's fingerprint is not a regression test.** Rebuilding the same ONNX on the
  same Orin gives a different file; compare parity, not bytes.
- **Time a new engine only with clocks locked.** Unlocked, the 50 Hz loop measured
  0.9 ms where the locked number was 0.45 ms, the same as the old engine.

## History

Written in W08, rehearsed on the dev box against a copy of the robot's deploy tree, and
accepted on the robot as stage D on 2026-10-02
([docs/stageD_policy_swap.md](../docs/stageD_policy_swap.md)):

The policy already running was installed through all eight steps: parity was unchanged at
1.144e-05, only the expected files changed, and in-loop inference was unchanged with
clocks locked. The three bad bundles of `tests/negative_installs.sh` were refused at step
1 with the deploy tree untouched. A rebuilt engine has a different file fingerprint from
the old one: TensorRT picks kernels by timing, so builds are not byte-identical, and
parity is the gate. A dev-box rehearsal beforehand (Python 3.8, a copy of the robot's
tree) found and fixed two bugs that would have made the installer exit silently.
