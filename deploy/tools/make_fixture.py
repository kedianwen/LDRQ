# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record PyTorch reference outputs so the TensorRT engine can be checked against them.

Runs the exported TorchScript policy on N observation vectors and writes both
sides to a binary fixture that `r1_parity_check` replays through the engine.

This is the step that turns "the node runs at 50 Hz" into "the node runs the
policy we trained". An engine that loads, runs fast, and computes something
subtly different -- wrong precision, a mis-fused layer, a bad ONNX opset
lowering -- looks perfectly healthy from the outside.

Needs only torch, so run it in the training environment:
    conda activate env_isaaclab
    python deploy/tools/make_fixture.py \
        --policy logs/rsl_rl/r1_flat/<run_id>/exported/policy.pt \
        --out deploy/artifacts/parity_fixture.bin

Fixture layout (little-endian):
    "R1FX" | u32 version=1 | u32 n | u32 in_dim | u32 out_dim
           | f32 inputs[n * in_dim] | f32 outputs[n * out_dim]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def build_samples(n: int, in_dim: int, generator: torch.Generator) -> torch.Tensor:
    """Observation vectors spanning the range the policy will actually see.

    A fixture of pure N(0, 1) noise would exercise a region of input space the
    policy never visits, where an FP16 discrepancy is neither representative nor
    actionable. The mix below covers the quiet case (near-zero, standing), the
    normal case, and deliberate outliers that stress the numerics.
    """
    n_quiet = n // 4
    n_outlier = n // 8
    n_normal = n - n_quiet - n_outlier

    quiet = 0.05 * torch.randn(n_quiet, in_dim, generator=generator)
    normal = 0.6 * torch.randn(n_normal, in_dim, generator=generator)
    outlier = 4.0 * torch.randn(n_outlier, in_dim, generator=generator)
    return torch.cat([quiet, normal, outlier], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Record PyTorch reference I/O for TensorRT parity checking.")
    ap.add_argument("--policy", required=True, help="Exported TorchScript policy (exported/policy.pt).")
    ap.add_argument("--out", required=True, help="Fixture path to write.")
    ap.add_argument("--n", type=int, default=512, help="Number of samples.")
    ap.add_argument("--in-dim", type=int, default=425, help="Observation dimension.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (the fixture must be reproducible).")
    args = ap.parse_args()

    policy = torch.jit.load(args.policy, map_location="cpu")
    policy.eval()

    gen = torch.Generator().manual_seed(args.seed)
    inputs = build_samples(args.n, args.in_dim, gen)

    with torch.inference_mode():
        outputs = policy(inputs)
    if outputs.dim() != 2 or outputs.shape[0] != args.n:
        raise SystemExit(f"unexpected policy output shape {tuple(outputs.shape)}")
    out_dim = outputs.shape[1]

    inputs = inputs.contiguous().float()
    outputs = outputs.contiguous().float()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        fh.write(b"R1FX")
        fh.write(struct.pack("<IIII", 1, args.n, args.in_dim, out_dim))
        fh.write(inputs.numpy().tobytes())
        fh.write(outputs.numpy().tobytes())

    print(f"[ok] {args.n} samples, {args.in_dim} -> {out_dim}")
    print(f"[ok] output range [{outputs.min():.4f}, {outputs.max():.4f}], "
          f"mean |a| = {outputs.abs().mean():.4f}")
    print(f"[ok] wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
