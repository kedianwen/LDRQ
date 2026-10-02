# policy_pack · swapping the trained policy with one command

Turns "deploy a different policy" from **three config edits + a C++ header edit + a
rebuild** into **one command**.

```bash
# dev box: after exporting, build a bundle
python3 policy_pack/make_bundle.py

# robot: install it (eight steps; any failure stops, nothing is left half-installed)
bash policy_pack/install_bundle.sh bundles/<run_id>
bash policy_pack/install_bundle.sh bundles/<run_id> --dry-run   # only show what would change
```

> **Status (2026-10-02):** this is stage D of the project (see the
> [top-level README](../README.md#how-the-project-was-organised)). **Rehearsed, not yet run
> on the robot.** The whole installer ran under Python 3.8 (the robot's) against a copy of
> the robot's deploy tree rebuilt from every package sent to it, with only colcon and the two
> TensorRT executables stood in for: the regression install changed exactly the expected
> files (`joint_map.hpp`, `joints.tsv` and `bridge.yaml` byte-identical), and the three
> negative installs stopped at step 1 with the tree unchanged. The rehearsal found two bugs,
> both fixed: sourcing ROS's setup script under `set -u` killed the installer right after the
> build with no message, and the build-failure message claimed nothing had been installed
> when the bundle's files were already staged. The installer now also calls the TensorRT
> executables by path rather than through the `ros2` CLI. Steps 6–8 (colcon, engine build,
> parity) are stage D's acceptance on the robot.

## What a bundle is

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

## Why "regenerate and rebuild" rather than "the bridge reads parameters"

The arrays in `joint_map.hpp` are `constexpr`, so the **compiler** checks their lengths
against `kNumJoints` / `kNumActions` — a wrong length is a build error, not a run-time
surprise on a robot that is already standing. Rebuilding both packages on the Orin takes
**29 s** measured (`r1_hw_bridge` 24.3 s and `r1_policy_runner` 25.3 s, in parallel).
That is cheaper than changing the one node that has been stable since W06.

## The eight steps, and what each one guards against

| step | action | guards against |
|---|---|---|
| 1 | `verify_bundle.py` | the refusal list below. **Writes nothing** |
| 2 | `gen_configs.py --check` | changing things the operator has not seen (comment-only changes are labelled as such) |
| 3 | copy the json / onnx / fixture into the deploy tree | — |
| 4 | `gen_joints.py` + `gen_joint_map.py --header=` | a second, independent check: gain order ≠ joint order exits here |
| 5 | `gen_configs.py` writes both yamls | a new policy running inside the previous policy's command envelope |
| 6 | `colcon build` | a generated header that does not compile = a wrong length or type |
| 7 | `r1_build_engine` (reused if the ONNX is unchanged; `--force-engine` rebuilds) | — |
| 8 | `r1_parity_check --tol 1e-3`, then write `installed.json` | an engine that computes wrong numbers on this machine |

`installed.json` records the run_id, the onnx sha256, the bundle manifest hash, and the
plan's size and fingerprint, so "which policy is running on the robot right now" is a fact
you read, not something you have to remember.

## What `verify_bundle.py` refuses

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
height scan **needs C++ changes** — the correct behaviour is to name the term at start-up
and refuse, not to guess. The `BRIDGE_TERMS` table in `verify_bundle.py` is that boundary;
change it together with the bridge's frame-assembly code.
