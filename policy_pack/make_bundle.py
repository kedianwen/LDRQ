#!/usr/bin/env python3
"""Assemble a policy bundle. Run on the DEV BOX, after export.

A bundle is the unit that travels to the robot. It carries everything that is a
property of the trained policy -- the ONNX, the observation/action contract, the
per-joint gains, the command envelope as trained, and its own parity fixture --
so that installing a new policy is a bundle swap instead of an archaeology
expedition across three config files and a C++ header.

What a bundle deliberately does NOT carry: the unitree_hg motor slot map. That
is a property of the ROBOT, it was established by measurement (the vendor enum
has RightShoulderPitch = 19 written as 29, which as a slot number addresses the
head), and putting it in a per-policy file invites someone to "fix" it.

    ./make_bundle.py                                  # from ../deploy + ../tasks
    ./make_bundle.py --run 2026-08-19_11-03-32_week04_nohead
    ./make_bundle.py --out /tmp/b --deploy ../deploy

The envelope is read from the TRAINING config (tasks/r1_flat/flat_env_cfg.py,
COMMAND_RANGES_FINAL) rather than from bridge.yaml, so the bundle records what
training actually covered and not what the last operator happened to allow.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import socket
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo: pathlib.Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10)
            suffix = "-dirty" if dirty.stdout.strip() else ""
            return out.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def envelope_from_task(task_cfg: pathlib.Path) -> dict:
    """Parse COMMAND_RANGES_FINAL and the pinned lin_vel_y out of the task cfg.

    Parsed rather than imported: importing flat_env_cfg.py pulls in Isaac Lab,
    which needs a working GPU driver, and a bundle must be buildable on a box
    where the simulator is not runnable.
    """
    text = task_cfg.read_text()
    m = re.search(r"COMMAND_RANGES_FINAL\s*=\s*\{(.+?)\}", text, re.S)
    if not m:
        raise SystemExit("[fail] COMMAND_RANGES_FINAL not found in " + str(task_cfg))
    body = m.group(1)

    def pair(key):
        mm = re.search(re.escape(key) + r"\"\s*:\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", body)
        if not mm:
            raise SystemExit("[fail] {} missing from COMMAND_RANGES_FINAL".format(key))
        return [float(mm.group(1)), float(mm.group(2))]

    vx, wz = pair("lin_vel_x"), pair("ang_vel_z")

    # lin_vel_y is not in the curriculum dict: it is pinned in the command cfg
    # and stays pinned. Confirm that rather than assume it.
    my = re.search(r"lin_vel_y\s*=\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", text)
    if not my:
        raise SystemExit("[fail] lin_vel_y not found in " + str(task_cfg))
    vy = [float(my.group(1)), float(my.group(2))]

    return {
        "source": "tasks/r1_flat/flat_env_cfg.py COMMAND_RANGES_FINAL + "
                  "UniformVelocityCommandCfg.ranges.lin_vel_y",
        "vx": vx,
        "vy": vy,
        "wz": wz,
        "note": "The end point of W04's command curriculum. This is what "
                "training covered, not a comfort limit. An axis pinned to "
                "[0, 0] is REFUSED by the bridge, not clamped: it does not "
                "exist in the training distribution.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", default=str(REPO / "deploy"),
                    help="deploy/ tree holding interface/ and artifacts/")
    ap.add_argument("--task-cfg", default=str(REPO / "tasks/r1_flat/flat_env_cfg.py"),
                    help="training task config, for the command envelope")
    ap.add_argument("--out", default=str(HERE / "bundles"),
                    help="directory to create the bundle under")
    ap.add_argument("--run", default=None,
                    help="run id (default: the 'run' field of policy_interface.json)")
    ap.add_argument("--artifacts", default=None,
                    help="directory holding policy.onnx + parity_fixture.bin (default: "
                         "<deploy>/artifacts, else the models/<name>/ matching the run id)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = ap.parse_args()

    deploy = pathlib.Path(args.deploy).expanduser().resolve()
    iface, artifacts = deploy / "interface", deploy / "artifacts"

    spec_path = iface / "policy_interface.json"
    if not spec_path.is_file():
        raise SystemExit("[fail] no " + str(spec_path))
    spec = json.loads(spec_path.read_text())
    run_id = args.run or spec.get("run")
    if not run_id:
        raise SystemExit("[fail] no run id: pass --run")
    if args.artifacts:
        artifacts = pathlib.Path(args.artifacts).expanduser().resolve()
    elif not (artifacts / "policy.onnx").is_file():
        # A fresh clone has no deploy/artifacts/ (it is a working area, not tracked);
        # the deployed policy's portable files are tracked under models/<name>/.
        for cand in sorted((REPO / "models").glob("*")):
            if run_id.endswith(cand.name) and (cand / "policy.onnx").is_file():
                artifacts = cand
                print("  (artifacts from " + str(cand) + ")")
                break

    root = pathlib.Path(args.out).expanduser().resolve() / run_id
    if root.exists():
        if not args.force:
            raise SystemExit("[fail] {} exists (use --force)".format(root))
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # Copies, not symlinks: a bundle has to survive being tarred up and landing
    # on a machine where none of these source paths exist.
    copies = [
        (artifacts / "policy.onnx", "policy.onnx"),
        (artifacts / "parity_fixture.bin", "parity_fixture.bin"),
        (iface / "policy_interface.json", "policy_interface.json"),
        (iface / "actuator_gains.json", "actuator_gains.json"),
    ]
    for src, name in copies:
        if not src.is_file():
            raise SystemExit("[fail] missing source file: " + str(src))
        shutil.copy2(src, root / name)
        print("  + {:<24} {:>10} B".format(name, (root / name).stat().st_size))

    env = envelope_from_task(pathlib.Path(args.task_cfg).expanduser().resolve())
    (root / "command_envelope.json").write_text(json.dumps(env, indent=2,
                                                          ensure_ascii=False) + "\n")
    print("  + {:<24} vx {} vy {} wz {}".format(
        "command_envelope.json", env["vx"], env["vy"], env["wz"]))

    prov = {
        "run_id": run_id,
        "created": datetime.date.today().isoformat(),
        "created_on": socket.gethostname(),
        "repo_git_sha": git_sha(REPO),
        "onnx_sha256": sha256(root / "policy.onnx"),
        "obs_dim": spec["observation"]["total_dim"],
        "action_dim": spec["action"]["dim"],
        "control_rate_hz": spec["control"]["control_rate_hz"],
        "sources": {
            "interface": str(iface),
            "artifacts": str(artifacts),
            "task_cfg": args.task_cfg,
        },
        "note": "The engine is NOT in this bundle. A TensorRT plan encodes the "
                "TensorRT version, GPU architecture and timed kernel tactics of "
                "the machine that built it, so it is built on the robot by "
                "install_bundle.sh and checked against parity_fixture.bin there.",
    }
    (root / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print("  + {:<24} run {} git {}".format("provenance.json", run_id,
                                            prov["repo_git_sha"][:12]))

    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        lines.append("{}  {}".format(sha256(path), path.relative_to(root)))
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print("  + {:<24} {} entries".format("MANIFEST.sha256", len(lines)))

    print("\nbundle: {}".format(root))
    sys.stdout.flush()

    # Never hand out a bundle that has not been validated. The validator is the
    # same one install_bundle.sh runs on the robot, so a bundle that passes here
    # cannot fail there for a reason this script could have caught.
    rc = subprocess.run([sys.executable, str(HERE / "verify_bundle.py"), str(root)]).returncode
    if rc != 0:
        print("\n[fail] the bundle this script just wrote does not validate.",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
