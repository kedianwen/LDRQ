#!/usr/bin/env python3
"""Report the deployment toolchain's version matrix, and flag known-bad skews.

Run it on the dev box and again on the robot: the point is the *comparison*
between the two columns, because everything that makes a TensorRT engine
non-portable shows up here.

    deploy/.venv/bin/python deploy/tools/check_env.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parents[1]

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def row(status: str, name: str, value: str, note: str = "") -> None:
    colour = {"ok  ": "\033[32m", "warn": "\033[33m", "FAIL": "\033[31m"}[status]
    print(f"  {colour}[{status}]\033[0m {name:<22} {value}" + (f"   ({note})" if note else ""))


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    problems = 0
    is_jetson = Path("/etc/nv_tegra_release").exists()

    print("\nhost")
    row(OK, "platform", f"{platform.system()} {platform.release()} {platform.machine()}")
    row(OK, "target kind", "Jetson (JetPack)" if is_jetson else "x86 dev box")
    if is_jetson:
        row(OK, "L4T", Path("/etc/nv_tegra_release").read_text().splitlines()[0])

    print("\ngpu / driver")
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if smi:
        row(OK, "gpu", smi)
    else:
        row(BAD, "gpu", "nvidia-smi unavailable", "engine build will fail")
        problems += 1

    print("\ntensorrt")
    try:
        import tensorrt as trt  # noqa: PLC0415

        row(OK, "python module", trt.__version__)
        py_ver = trt.__version__
    except ImportError:
        row(WARN, "python module", "not importable", "only needed for the build path")
        py_ver = None

    hdr = DEPLOY_ROOT / "third_party" / "tensorrt" / "include" / "NvInferVersion.h"
    if not hdr.exists():
        for cand in ("/usr/include/aarch64-linux-gnu", "/usr/include/x86_64-linux-gnu", "/usr/include"):
            if (Path(cand) / "NvInferVersion.h").exists():
                hdr = Path(cand) / "NvInferVersion.h"
                break
    if hdr.exists():
        macros = {}
        for line in hdr.read_text().splitlines():
            for key in ("NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR", "NV_TENSORRT_PATCH"):
                if line.startswith(f"#define {key}"):
                    macros[key] = line.split()[2]
        hdr_ver = ".".join(macros.get(k, "?") for k in
                           ("NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR", "NV_TENSORRT_PATCH"))
        if py_ver and not py_ver.startswith(hdr_ver):
            row(BAD, "c++ headers", f"{hdr_ver}  ({hdr})", f"does not match runtime {py_ver}")
            problems += 1
        else:
            row(OK, "c++ headers", f"{hdr_ver}  ({hdr})")
    else:
        row(BAD, "c++ headers", "NvInfer.h not found", "run deploy/setup.sh")
        problems += 1

    print("\ncuda runtime")
    cudart = list((DEPLOY_ROOT / "third_party" / "cuda" / "lib").glob("libcudart.so.*"))
    if cudart:
        row(OK, "libcudart", Path(cudart[0]).resolve().name)
    elif Path("/usr/local/cuda/lib64/libcudart.so").exists():
        row(OK, "libcudart", "/usr/local/cuda/lib64/libcudart.so")
    else:
        row(BAD, "libcudart", "not found", "run deploy/setup.sh")
        problems += 1

    print("\nros 2")
    if Path("/opt/ros/humble").exists():
        row(OK, "distro", "humble")
    else:
        row(BAD, "distro", "humble not installed")
        problems += 1
    for tool in ("colcon", "cmake", "g++"):
        path = shutil.which(tool)
        row(OK if path else BAD, tool, path or "not found")
        problems += 0 if path else 1

    print("\nworkspace")
    ws = DEPLOY_ROOT / "ros2_ws" / "install" / "r1_policy_runner"
    row(OK if ws.exists() else WARN, "built", str(ws) if ws.exists() else "not built yet",
        "" if ws.exists() else "run: colcon build")
    for name, rel in (
        ("onnx", "artifacts/policy.onnx"),
        ("fixture", "artifacts/parity_fixture.bin"),
        ("engine", "artifacts/policy_fp32.plan"),
        ("interface", "interface/policy_interface.json"),
    ):
        p = DEPLOY_ROOT / rel
        row(OK if p.exists() else WARN, name, str(rel) if p.exists() else "missing")

    print()
    if problems:
        print(f"\033[31m{problems} blocking problem(s).\033[0m See deploy/README.md.\n")
    else:
        print("\033[32mtoolchain ok.\033[0m\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
