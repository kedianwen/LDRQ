#!/usr/bin/env python3
"""CPU forward pass for the R1 policy, straight from the ONNX weights.

Two jobs on the robot, both of which matter only if TensorRT misbehaves there
the way it did on the dev box:

1. **Cross-check** -- verify a parity fixture without TensorRT, PyTorch or a GPU,
   so "is the engine wrong?" can be answered on the spot with an independent
   implementation rather than by trusting the same library twice.
2. **Fallback sizing** -- measure how long this policy takes on the Orin's ARM
   cores. It is four GEMMs and three ELUs, ~90k multiply-accumulates. If that
   comfortably fits the 20 ms control budget, then "drop TensorRT entirely" is a
   real option rather than a hopeful one, and the deployment risk stops being a
   single point of failure.

Needs only numpy and onnx -- no CUDA, no TensorRT, no torch.

    python3 tools/cpu_reference.py --onnx artifacts/policy.onnx \
        --fixture artifacts/parity_fixture.bin --bench 2000
"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def load_weights(onnx_path: str):
    """Pull the Gemm weight/bias pairs out in graph order.

    Walking the graph rather than assuming the ``actor.N`` naming means a
    re-exported policy with different layer names still works.
    """
    model = onnx.load(onnx_path)
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            w = init[node.input[1]]
            b = init[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], np.float32)
            trans_b = any(a.name == "transB" and a.i for a in node.attribute)
            layers.append((np.ascontiguousarray(w if trans_b else w.T, np.float32),
                           np.ascontiguousarray(b, np.float32)))
        elif node.op_type not in ("Elu",):
            raise SystemExit(f"unexpected op '{node.op_type}' -- this reference only covers Gemm+Elu")
    return layers


def forward(x: np.ndarray, layers) -> np.ndarray:
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:  # ELU on every layer but the last
            x = np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1.0)
    return x


def read_fixture(path: str):
    with open(path, "rb") as fh:
        assert fh.read(4) == b"R1FX", "bad fixture magic"
        _, n, ind, outd = struct.unpack("<IIII", fh.read(16))
        ins = np.frombuffer(fh.read(n * ind * 4), np.float32).reshape(n, ind).copy()
        ref = np.frombuffer(fh.read(n * outd * 4), np.float32).reshape(n, outd).copy()
    return ins, ref


def main() -> int:
    ap = argparse.ArgumentParser(description="numpy-only forward pass for the R1 policy")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--fixture", help="verify against this fixture's PyTorch reference outputs")
    ap.add_argument("--bench", type=int, default=0, help="time this many single-sample forwards")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    layers = load_weights(args.onnx)
    shapes = " -> ".join([str(layers[0][0].shape[1])] + [str(w.shape[0]) for w, _ in layers])
    macs = sum(w.size for w, _ in layers)
    print(f"network {shapes}   ({macs/1000:.0f}k MAC/forward)", flush=True)

    code = 0
    if args.fixture:
        ins, ref = read_fixture(args.fixture)
        d = np.abs(forward(ins, layers) - ref)
        ok = d.max() <= args.tol
        print(f"vs fixture ({len(ins)} samples): max|err|={d.max():.3e}  mean={d.mean():.3e}  "
              f"{'PASS' if ok else 'FAIL'} (tol {args.tol:.0e})", flush=True)
        code = 0 if ok else 1

    if args.bench:
        x = (np.random.default_rng(0).standard_normal((1, layers[0][0].shape[1])) * 0.5).astype(np.float32)
        forward(x, layers)  # warm up numpy's BLAS threads
        times = np.empty(args.bench)
        for i in range(args.bench):
            t0 = time.perf_counter()
            forward(x, layers)
            times[i] = (time.perf_counter() - t0) * 1e6
        times.sort()
        p = lambda q: times[int(q * (len(times) - 1))]  # noqa: E731
        print(f"cpu latency (us over {args.bench}): p50={p(.5):.0f}  p95={p(.95):.0f}  "
              f"p99={p(.99):.0f}  max={times[-1]:.0f}", flush=True)
        print(f"budget at 50 Hz = 20000 us  ->  p99 uses {p(.99)/20000*100:.2f}%", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
