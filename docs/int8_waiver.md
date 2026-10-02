# INT8 quantisation (requirement FR-Q3): waived with evidence · precision comparison (FR-Q4): satisfied at two precisions

Decided 2026-09-26. INT8 was planned for W08 as post-training quantisation with a
calibration set, a behavioural acceptance, and a FP32/FP16/INT8 benchmark. It was cut
because three independent measurements say it cannot pay for itself **on this model** —
not because it was hard. Numbers below are from the robot's Orin NX (TensorRT 8.5.2.2,
clocks locked) unless marked otherwise.

## 1. It cannot buy time

The policy is a 425 → 24 MLP with 90,648 parameters, run at batch 1 and 50 Hz. One step
is ~180 kFLOP — a fraction of a microsecond at the Orin GPU's FP32 throughput — against
70 µs measured, so over 99 % of a step is kernel launch and synchronisation, not
arithmetic. Lower precision can only shorten the part that is already negligible.

| engine | where | parity vs PyTorch (max_abs) | latency p50 / p99 (µs, tight loop) |
|---|---|---|---|
| TensorRT FP32 | Orin | 1.335e-05 | 70.5 / 95.7 |
| TensorRT FP16 (`kFP16` permitted) | Orin | 1.144e-05 | 70.1 / 97.4 |
| TensorRT FP16, TF32 cleared (W07) | Orin | 1.717e-05 | 171 p50 — not faster than FP32 (different session, so no "slower" claim) |

FP16's error stays at FP32's order (true FP16 arithmetic would land near 1e-3) because
`kFP16`/`kINT8` *permit* reduced precision rather than require it: TensorRT picks each
layer's kernel by timing it at build, and for this network it keeps FP32 kernels. In the
real 50 Hz loop the whole inference is ~455 µs p50 against a 20,000 µs budget (2.3 %).
There is no latency problem for quantisation to solve.

## 2. It cannot buy space

| | FP32 engine | FP16 engine | change |
|---|---|---|---|
| Orin, TensorRT 8.5.2.2 | 966,887 B | 1,461,299 B | **+51.1 %** |

The reduced-precision build adds reformat layers and keeps weight copies in both
precisions; on a network this small that outweighs any saving. The builder also warned
that 4 weights are below FP16's subnormal limit — harmless here, but the weight
distribution already touches FP16's range.

## 3. The calibration set would not be defensible

The observation stacks five frames, and neighbouring stacks share 80 % of their content.
A 60 s single-speed straight walk is 3,000 samples but one operating point and ~72 gait
cycles. A calibration set that represents the policy's input distribution would need
real walking across the vx × wz command grid — many sessions on the robot for a
quantisation that, by §1 and §2, returns nothing.

## What is recorded

- **FR-Q3 (INT8 PTQ): waived after measurement.** `r1_build_engine --int8` and the
  calibration recorder (`deploy/tools/record_calib_obs.py`) are kept and still work, so
  the decision can be revisited — for example for a much larger policy, where the
  arithmetic would dominate.
- **FR-Q4 (precision tiers benchmarked): satisfied at FP32 and FP16**, with the
  table above as the benchmark.
- **The methodology INT8 was meant to serve is kept.** FR-R2 needed one controlled,
  physically real perturbation, swept in simulation and spot-checked on the robot. The
  knob became actuator-gain error (`kp_scale`) — see [stageA_kp_sweep.md](stageA_kp_sweep.md).

Rule followed: no change of metric to make INT8 look useful (batching, throughput).
The control loop is batch 1 at 50 Hz, and that is the only configuration that matters.
