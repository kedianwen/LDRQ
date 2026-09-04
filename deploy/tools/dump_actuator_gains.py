#!/usr/bin/env python3
"""Extract the per-joint PD gains the policy was TRAINED with.

Why this exists: `deploy/tools/dump_interface.py` exported the observation
layout, the action mapping and the default pose, but not the actuator gains --
so the bridge shipped a single global kp/kd applied to all 24 joints while
training used six groups spanning kp 20..100 and kd 1..2. On the robot that
showed up as legs with no perceptible damping (their group wants kp 100, kd 2)
and a head oscillating at high frequency (its group wants kd 1, and it was
getting 3). The gains are part of the policy contract exactly like the default
pose is; they belong in a generated file, gated like everything else.

Reading `assets/r1/r1.py` with the ast module rather than importing it: the
module pulls in isaaclab, which is not importable outside the Isaac Python
environment, and this has to run on the robot too. Every value we need is a
literal in an ImplicitActuatorCfg call, so the parse is exact -- and if the file
ever stops being literals, ast fails loudly instead of guessing.

  python3 dump_actuator_gains.py            # writes interface/actuator_gains.json
  python3 dump_actuator_gains.py --check    # verify the existing file is current
"""
import argparse
import ast
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = HERE.parent
REPO = DEPLOY.parent
ASSET = REPO / "assets" / "r1" / "r1.py"
SPEC = DEPLOY / "interface" / "policy_interface.json"
OUT = DEPLOY / "interface" / "actuator_gains.json"

WANT = ("stiffness", "damping", "effort_limit")


def parse_groups(path):
    """Every ImplicitActuatorCfg(...) in the file, as
    {name: {joint_names_expr, stiffness, damping, effort_limit}}."""
    tree = ast.parse(path.read_text())
    groups = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Dict)):
            continue
        for key, val in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not (isinstance(val, ast.Call) and
                    getattr(val.func, "id", "") == "ImplicitActuatorCfg"):
                continue
            g = {}
            for kw in val.keywords:
                if kw.arg == "joint_names_expr":
                    g["patterns"] = ast.literal_eval(kw.value)
                elif kw.arg in WANT:
                    g[kw.arg] = float(ast.literal_eval(kw.value))
            missing = [f for f in ("patterns",) + WANT if f not in g]
            if missing:
                raise SystemExit(f"[fail] actuator group '{key.value}' is missing {missing}")
            groups[key.value] = g
    if not groups:
        raise SystemExit(f"[fail] no ImplicitActuatorCfg found in {path}")
    return groups


def resolve(groups, joint_names):
    """Assign each articulation joint to exactly one group.

    Isaac Lab matches joint_names_expr with re.fullmatch. A joint matching two
    groups, or none, is a silent mis-assignment in training and a wrong gain on
    the robot, so both are hard errors here rather than a first-match-wins
    guess."""
    out, problems = [], []
    for name in joint_names:
        hits = [gname for gname, g in groups.items()
                if any(re.fullmatch(p, name) for p in g["patterns"])]
        if len(hits) != 1:
            problems.append(f"{name}: matched {len(hits)} groups {hits}")
            out.append(None)
        else:
            out.append(hits[0])
    if problems:
        print("[fail] actuator group assignment is not one-to-one:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)
    return out


def build():
    spec = json.loads(SPEC.read_text())
    names = spec["articulation"]["joint_names"]
    groups = parse_groups(ASSET)
    assign = resolve(groups, names)
    return {
        "source": str(ASSET.relative_to(REPO)),
        "run": spec["run"],
        "note": ("Per-joint PD gains as trained, in articulation order. The bridge "
                 "applies kp_scale * stiffness and kd_scale * damping; both scales "
                 "default to 1.0, so the default IS the trained controller."),
        "joint_names": names,
        "group": assign,
        "stiffness": [groups[g]["stiffness"] for g in assign],
        "damping": [groups[g]["damping"] for g in assign],
        "effort_limit": [groups[g]["effort_limit"] for g in assign],
        "groups": {g: {k: v for k, v in gv.items()} for g, gv in groups.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if the committed file differs from a fresh extract")
    args = ap.parse_args()

    data = build()
    text = json.dumps(data, indent=2) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"[fail] {OUT} does not exist", file=sys.stderr)
            return 1
        if OUT.read_text() != text:
            print(f"[fail] {OUT} is stale -- rerun without --check", file=sys.stderr)
            return 1
        print(f"[ok] {OUT.name} matches {ASSET.name}")
        return 0

    OUT.write_text(text)
    n = len(data["joint_names"])
    print(f"{OUT}: {n} joints across {len(data['groups'])} actuator groups")
    for g, gv in data["groups"].items():
        members = [data["joint_names"][i] for i, a in enumerate(data["group"]) if a == g]
        print(f"  {g:<8} kp={gv['stiffness']:>5.1f} kd={gv['damping']:>4.1f} "
              f"tau<={gv['effort_limit']:>5.1f}  x{len(members)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
