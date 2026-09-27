#!/usr/bin/env python3
"""Validate a policy bundle BEFORE anything is built or installed.

Runs anywhere: no ROS, no CUDA, no TensorRT, no onnx package. Pure stdlib, and
Python 3.8 compatible because the robot is ROS 2 foxy on Ubuntu 20.04.

The point of this script is what it REFUSES. A bundle that is merely
self-consistent is not enough -- it also has to be a bundle the C++ bridge can
actually serve, and the bridge's observation assembler knows how to compute
exactly six terms and no others. A policy trained with a seventh term (base
linear velocity, foot contacts, height scan) cannot be deployed by swapping
config, and the only safe outcome is to say so by name here rather than to
zero-fill a term at 50 Hz and let it read as a sim2real gap for a week.

    ./verify_bundle.py bundles/<run_id>            # validate
    ./verify_bundle.py bundles/<run_id> --quiet    # exit code only

Exit codes: 0 ok, 1 the bundle is invalid, 2 the bundle is unreadable.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

# The observation terms deploy/ros2_ws/src/r1_hw_bridge/src/bridge_node.cpp
# actually assembles, in the order it assembles them, with the width it
# produces. Keep this in step with that file's frame builder -- it is the
# contract, and a mismatch here is the whole reason this script exists.
#
#   "J" = one value per articulation joint (26), "A" = one per action (24).
BRIDGE_TERMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos": "J",
    "joint_vel": "J",
    "actions": "A",
}

REQUIRED = [
    "policy.onnx",
    "parity_fixture.bin",
    "policy_interface.json",
    "actuator_gains.json",
    "command_envelope.json",
    "provenance.json",
    "MANIFEST.sha256",
]

GAIN_ARRAYS = ("stiffness", "damping", "effort_limit")


class Invalid(Exception):
    """A specific, actionable reason the bundle cannot be installed."""


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(root: pathlib.Path, name: str) -> dict:
    try:
        return json.loads((root / name).read_text())
    except json.JSONDecodeError as exc:
        raise Invalid("{}: not valid JSON ({})".format(name, exc))


def check_present(root: pathlib.Path) -> None:
    missing = [n for n in REQUIRED if not (root / n).is_file()]
    if missing:
        raise Invalid("missing from the bundle: " + ", ".join(missing))


def check_manifest(root: pathlib.Path) -> int:
    """Every file the manifest names must hash to what it says, and every file
    in the bundle must be named. An unlisted file is as bad as a changed one:
    it is how a hand-edited json ends up installed."""
    listed = {}
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # sha256sum format: "<hex>  <relative path>"
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise Invalid("MANIFEST.sha256: cannot parse line: " + line)
        listed[parts[1].lstrip("*").strip()] = parts[0]

    on_disk = sorted(
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.sha256")

    unlisted = [n for n in on_disk if n not in listed]
    if unlisted:
        raise Invalid("in the bundle but not in MANIFEST.sha256: "
                      + ", ".join(unlisted))
    for name, want in sorted(listed.items()):
        path = root / name
        if not path.is_file():
            raise Invalid("MANIFEST.sha256 names a missing file: " + name)
        got = sha256(path)
        if got != want:
            raise Invalid("{} does not match MANIFEST.sha256\n"
                          "  manifest {}\n  on disk  {}".format(name, want, got))
    return len(listed)


def check_interface(spec: dict) -> dict:
    """policy_interface.json: the observation and action contract."""
    for key in ("observation", "action", "articulation", "control"):
        if key not in spec:
            raise Invalid("policy_interface.json: no '{}' section".format(key))

    obs, act, art = spec["observation"], spec["action"], spec["articulation"]
    n_joints = int(art["num_joints"])
    n_act = int(act["dim"])

    if len(art["joint_names"]) != n_joints:
        raise Invalid("articulation: num_joints={} but {} names"
                      .format(n_joints, len(art["joint_names"])))
    if len(art["default_joint_pos"]) != n_joints:
        raise Invalid("articulation: default_joint_pos has {} entries, expected {}"
                      .format(len(art["default_joint_pos"]), n_joints))

    for field in ("joint_names", "scale", "default_pos", "joint_ids_in_articulation"):
        if len(act[field]) != n_act:
            raise Invalid("action.{} has {} entries but action.dim={}"
                          .format(field, len(act[field]), n_act))

    # The action's joints must be real articulation joints, and the declared
    # indices must agree with the names. A mismatch here sends a knee command to
    # a shoulder, silently.
    for i, (name, art_i) in enumerate(
            zip(act["joint_names"], act["joint_ids_in_articulation"])):
        if not 0 <= art_i < n_joints:
            raise Invalid("action {} -> articulation index {} out of range"
                          .format(i, art_i))
        if art["joint_names"][art_i] != name:
            raise Invalid(
                "action {} says '{}' but articulation[{}] is '{}'"
                .format(i, name, art_i, art["joint_names"][art_i]))

    # Observation terms: known to the bridge, in the bridge's order, at the
    # bridge's width.
    names = [t["name"] for t in obs["terms"]]
    unknown = [n for n in names if n not in BRIDGE_TERMS]
    if unknown:
        raise Invalid(
            "observation term(s) the bridge cannot assemble: {}\n"
            "  the bridge computes exactly: {}\n"
            "  this policy needs a bridge change (C++), not a config change."
            .format(", ".join(unknown), ", ".join(BRIDGE_TERMS)))
    if names != [n for n in BRIDGE_TERMS if n in names]:
        raise Invalid("observation terms are out of order.\n  bundle: {}\n"
                      "  bridge assembles in this order: {}"
                      .format(names, list(BRIDGE_TERMS)))
    for term in obs["terms"]:
        want = BRIDGE_TERMS[term["name"]]
        want = n_joints if want == "J" else (n_act if want == "A" else want)
        if int(term["width"]) != want:
            raise Invalid("observation term '{}' width {} but the bridge "
                          "produces {}".format(term["name"], term["width"], want))

    frame = sum(int(t["width"]) for t in obs["terms"])
    if frame != int(obs["frame_dim"]):
        raise Invalid("observation: terms sum to {} but frame_dim says {}"
                      .format(frame, obs["frame_dim"]))
    total = frame * int(obs["history_length"])
    if total != int(obs["total_dim"]):
        raise Invalid("observation: frame_dim {} x history_length {} = {} but "
                      "total_dim says {}".format(frame, obs["history_length"],
                                                 total, obs["total_dim"]))
    if float(spec["control"]["control_rate_hz"]) <= 0:
        raise Invalid("control.control_rate_hz must be positive")
    return {"n_joints": n_joints, "n_act": n_act, "total_dim": total,
            "art_names": list(art["joint_names"]),
            "rate": float(spec["control"]["control_rate_hz"])}


def check_gains(gains: dict, art_names, n_joints: int) -> None:
    """Per joint, in articulation order. Six groups, not one scalar -- the W06
    export omission that left the legs undamped and the head vibrating was
    exactly this file being wrong in a way nothing checked."""
    if list(gains.get("joint_names", [])) != list(art_names):
        raise Invalid(
            "actuator_gains.json joint order != articulation joint order.\n"
            "  the bridge indexes gains by articulation index, so a different "
            "order applies one joint's gains to another.")
    for field in GAIN_ARRAYS:
        if field not in gains:
            raise Invalid("actuator_gains.json: no '{}'".format(field))
        if len(gains[field]) != n_joints:
            raise Invalid("actuator_gains.json {} has {} entries, expected {}"
                          .format(field, len(gains[field]), n_joints))
        bad = [art_names[i] for i, v in enumerate(gains[field]) if float(v) <= 0]
        if bad:
            raise Invalid("actuator_gains.json {} is <= 0 for: {}"
                          .format(field, ", ".join(bad)))


def check_envelope(env: dict) -> list:
    """The command envelope as TRAINED. Not a comfort limit: an axis pinned to
    zero for all of training is refused, not clamped, and that distinction has
    to survive into the bundle."""
    notes = []
    for axis in ("vx", "vy", "wz"):
        if axis not in env:
            raise Invalid("command_envelope.json: no '{}'".format(axis))
        rng = env[axis]
        if not isinstance(rng, list) or len(rng) != 2:
            raise Invalid("command_envelope.json {}: expected [lo, hi]".format(axis))
        lo, hi = float(rng[0]), float(rng[1])
        if lo > hi:
            raise Invalid("command_envelope.json {}: lo {} > hi {}"
                          .format(axis, lo, hi))
        if lo == 0.0 and hi == 0.0:
            notes.append("{} is pinned to 0: commands on this axis are REFUSED, "
                         "not clamped".format(axis))
    if float(env["vx"][0]) < 0:
        notes.append("vx lo < 0: reverse walking is declared as trained -- "
                     "confirm that is real before shipping")
    return notes


def check_provenance(prov: dict, root: pathlib.Path) -> None:
    for key in ("run_id", "created", "onnx_sha256"):
        if key not in prov:
            raise Invalid("provenance.json: no '{}'".format(key))
    got = sha256(root / "policy.onnx")
    if got != prov["onnx_sha256"]:
        raise Invalid("provenance.json onnx_sha256 does not match policy.onnx\n"
                      "  provenance {}\n  on disk    {}".format(
                          prov["onnx_sha256"], got))


def verify(root: pathlib.Path, quiet: bool = False) -> int:
    def say(*a):
        if not quiet:
            print(*a)

    check_present(root)
    n_listed = check_manifest(root)
    say("manifest      ok  ({} files hashed)".format(n_listed))

    spec = load_json(root, "policy_interface.json")
    facts = check_interface(spec)
    say("interface     ok  obs[{}] = {} terms x {} frames -> actions[{}] @ {:.0f} Hz"
        .format(facts["total_dim"], len(spec["observation"]["terms"]),
                spec["observation"]["history_length"], facts["n_act"], facts["rate"]))

    check_gains(load_json(root, "actuator_gains.json"),
                facts["art_names"], facts["n_joints"])
    say("gains         ok  per joint, {} joints, articulation order"
        .format(facts["n_joints"]))

    env = load_json(root, "command_envelope.json")
    notes = check_envelope(env)
    say("envelope      ok  vx {}  vy {}  wz {}".format(env["vx"], env["vy"], env["wz"]))
    for n in notes:
        say("              ..  " + n)

    prov = load_json(root, "provenance.json")
    check_provenance(prov, root)
    say("provenance    ok  run {}  onnx {}".format(
        prov["run_id"], prov["onnx_sha256"][:12]))
    say("\nVERDICT: PASS -- this bundle can be installed.")
    return 0


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    quiet = "--quiet" in argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    root = pathlib.Path(args[0]).expanduser().resolve()
    if not root.is_dir():
        print("not a directory: {}".format(root), file=sys.stderr)
        return 2
    try:
        return verify(root, quiet)
    except Invalid as exc:
        print("\nVERDICT: FAIL -- {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
