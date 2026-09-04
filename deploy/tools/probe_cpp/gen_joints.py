#!/usr/bin/env python3
"""Generate joints.tsv for probe_lowstate --map from the interface spec.

The joint names and the articulation<->action index correspondence have exactly
one source of truth, `deploy/interface/policy_interface.json`, which is itself
generated from the trained run. Hand-copying that list into the probe would
reintroduce the transcription error the probe exists to prevent, so the list is
generated instead.

  python3 gen_joints.py            # writes joints.tsv next to this script
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SPEC = HERE.parents[1] / "interface" / "policy_interface.json"


def main() -> int:
    spec = json.loads(SPEC.read_text())
    art = spec["articulation"]["joint_names"]
    act = {n: i for i, n in enumerate(spec["action"]["joint_names"])}

    out = HERE / "joints.tsv"
    with out.open("w") as f:
        f.write(f"# GENERATED from {SPEC.name} by gen_joints.py -- do not hand-edit.\n")
        f.write("# art_idx\tjoint_name\taction_idx (-1 = not policy-actuated)\n")
        for i, name in enumerate(art):
            f.write(f"{i}\t{name}\t{act.get(name, -1)}\n")
    print(f"{out}: {len(art)} joints, {len(act)} policy-actuated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
