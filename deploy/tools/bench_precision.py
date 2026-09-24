#!/usr/bin/env python3
"""FP32 / FP16 / INT8 benchmark table: latency, engine size, power.

FR-Q4's deliverable. Three things this tool is built around, all of them
measured rather than assumed:

1. LATENCY MUST BE REPORTED AT THE REAL DUTY CYCLE. A compact benchmark loop
   keeps the GPU clocked up; the control loop calls infer() once per 20 ms and
   the GPU falls to a low-power state in between. Measured on the dev box those
   differ by 8x (23.6 us vs 198 us). Only the paced number describes the robot,
   so this tool always passes --duty-hz and prints both.

2. LOCK THE CLOCKS FIRST, or the numbers are a lottery. The tool refuses to run
   until nvpmodel/jetson_clocks state has been checked.

3. THE EXPECTED RESULT IS "NO MATERIAL DIFFERENCE", and that is a finding, not a
   failure. This policy is 73k parameters at batch 1: ~150 kFLOP of arithmetic
   against ~900 us of wall time, so >99.9% of a step is launch overhead and
   memory traffic. FP16 already demonstrated this on the Orin (2026-09-24:
   error 1.717e-05, FP32's order, because the builder kept FP32 kernels).
   Do not go looking for a speed-up to report.

  python3 bench_precision.py --plans fp32=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan \\
      fp16=$R1_DEPLOY_ROOT/artifacts/policy_fp16.plan \\
      int8=$R1_DEPLOY_ROOT/artifacts/policy_int8.plan \\
      --fixture $R1_DEPLOY_ROOT/artifacts/parity_fixture.bin \\
      --out ~/orin_commissioning/precision_table.md
"""
import argparse
import glob
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------

def find_rails():
    """Locate the INA3221 power rails JetPack exposes through hwmon.

    Two shapes exist across JetPack versions: `power<N>_input` in microwatts, or
    `in<N>_input` (mV) plus `curr<N>_input` (mA) to be multiplied. Rail names come
    from `in<N>_label`, and VDD_IN is the board total -- the one to report.
    """
    rails = []
    for hw in glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/"):
        for lab in sorted(glob.glob(hw + "in*_label")):
            n = re.search(r"in(\d+)_label", lab)
            if not n:
                continue
            idx = n.group(1)
            try:
                name = pathlib.Path(lab).read_text().strip()
            except OSError:
                continue
            pw, v, c = f"{hw}power{idx}_input", f"{hw}in{idx}_input", f"{hw}curr{idx}_input"
            if os.path.exists(pw):
                rails.append((name, ("uW", pw)))
            elif os.path.exists(v) and os.path.exists(c):
                rails.append((name, ("VI", v, c)))
    return rails


def read_rail(spec):
    try:
        if spec[0] == "uW":
            return int(pathlib.Path(spec[1]).read_text().strip()) / 1e6
        mv = int(pathlib.Path(spec[1]).read_text().strip())
        ma = int(pathlib.Path(spec[2]).read_text().strip())
        return mv * ma / 1e6
    except (OSError, ValueError):
        return None


class PowerSampler(threading.Thread):
    """Samples every rail at 10 Hz in the background while a benchmark runs."""

    def __init__(self, rails, period=0.1):
        super().__init__(daemon=True)
        self.rails, self.period = rails, period
        self.samples = {n: [] for n, _ in rails}
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            for name, spec in self.rails:
                w = read_rail(spec)
                if w is not None:
                    self.samples[name].append(w)
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)

    def summary(self):
        out = {}
        for n, v in self.samples.items():
            if v:
                out[n] = (sum(v) / len(v), max(v))
        return out


# ---------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------

def clock_state():
    """What power model and clock policy is in force. Reported, never changed --
    nvpmodel and jetson_clocks need root, and a benchmark tool is the wrong place
    to silently reconfigure a robot."""
    out = []
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out.append(" / ".join(x.strip() for x in r.stdout.split("\n") if x.strip()))
    except (OSError, subprocess.SubprocessError):
        out.append("nvpmodel unavailable")
    gov = glob.glob("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        try:
            out.append("cpu0 governor=" + pathlib.Path(gov[0]).read_text().strip())
        except OSError:
            pass
    return "; ".join(out) if out else "unknown"


# ---------------------------------------------------------------------------
# one measurement
# ---------------------------------------------------------------------------

LAT = re.compile(r"latency (compact-loop|@duty) \(us over (\d+)\): "
                 r"p50=([\d.]+)\s+p95=([\d.]+)\s+p99=([\d.]+)\s+max=([\d.]+)")
PAR = re.compile(r"(?:parity|difference): max_abs=(\S+)\s+mean_abs=(\S+)")
FPR = re.compile(r"^plan (\d+)B fnv1a=(0x[0-9a-f]+)", re.M)


def bench_one(binary, plan, fixture, duty_hz, iters, rails, baseline=None, names=None):
    if not os.path.exists(plan):
        return {"error": f"missing {plan}"}
    cmd = [binary, "--plan", plan, "--fixture", fixture,
           "--duty-hz", str(duty_hz), "--latency-iters", str(iters)]
    if baseline:
        cmd += ["--baseline", baseline, "--per-dim"]
        if names:
            cmd += ["--names", names]

    idle = {}
    if rails:
        # Idle for a moment first: a power number with no baseline is unreadable,
        # because most of the board's draw is not the GPU.
        s = PowerSampler(rails)
        s.start()
        time.sleep(2.0)
        s.stop()
        idle = s.summary()

    sampler = PowerSampler(rails) if rails else None
    if sampler:
        sampler.start()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if sampler:
        sampler.stop()

    res = {"plan": plan, "stdout": r.stdout, "stderr": r.stderr, "rc": r.returncode,
           "bytes": os.path.getsize(plan), "cmd": " ".join(cmd)}
    for m in LAT.finditer(r.stdout):
        res["compact" if m.group(1) == "compact-loop" else "duty"] = {
            "iters": int(m.group(2)), "p50": float(m.group(3)),
            "p95": float(m.group(4)), "p99": float(m.group(5)), "max": float(m.group(6))}
    pm = PAR.search(r.stdout)
    if pm:
        res["max_abs"], res["mean_abs"] = pm.group(1), pm.group(2)
    fm = FPR.search(r.stdout)
    if fm:
        res["fingerprint"] = f"{fm.group(1)}B fnv1a={fm.group(2)}"
    res["idle_w"], res["run_w"] = idle, (sampler.summary() if sampler else {})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--binary", default=os.path.expandvars(
        "$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner/r1_parity_check"))
    ap.add_argument("--baseline-label", default="fp16",
                    help="which plan the others are compared against per-dimension")
    ap.add_argument("--names", default=str(pathlib.Path(__file__).resolve().parent
                                           / "probe_cpp" / "joints.tsv"))
    ap.add_argument("--duty-hz", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=500,
                    help="500 at 50 Hz = 10 s paced, and the parity+compact phases "
                         "together are under a second, so >90%% of the wall time "
                         "the power sampler sees is the paced phase")
    ap.add_argument("--out", help="write the markdown table here")
    ap.add_argument("--assume-clocks-locked", action="store_true",
                    help="skip the interactive clock confirmation (for scripting)")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"r1_parity_check not found at {args.binary}\n"
                 f"  source $R1_DEPLOY_ROOT/../deploy/env.sh, and check the execute bit:\n"
                 f"  chmod +x $R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/"
                 f"r1_policy_runner/*")

    plans = {}
    for p in args.plans:
        if "=" not in p:
            sys.exit(f"--plans wants LABEL=PATH, got {p!r}")
        k, v = p.split("=", 1)
        plans[k] = os.path.expandvars(os.path.expanduser(v))

    clocks = clock_state()
    print(f"clock state: {clocks}")
    rails = find_rails()
    if rails:
        print(f"power rails: {', '.join(n for n, _ in rails)}")
    else:
        print("power rails: NONE FOUND under /sys/bus/i2c/drivers/ina3221*.")
        print("  Power will be reported as n/a. Do not substitute a tegrastats number")
        print("  read by eye -- report n/a and say why. (If tegrastats does show")
        print("  VDD_IN, this kernel exposes the rails somewhere else; find the path")
        print("  and pass it on rather than guessing.)")

    if not args.assume_clocks_locked:
        print("\nBefore measuring, the clocks must be pinned, or these numbers are a")
        print("lottery -- W05 saw the tail tighten 11x from locking them alone:")
        print("  sudo nvpmodel -m 0        # max power model")
        print("  sudo jetson_clocks        # pin clocks to max")
        try:
            if input("\ndone, and the state above looks right? [y/N] ").strip().lower() != "y":
                sys.exit("aborted -- nothing measured")
        except EOFError:
            sys.exit("no tty; re-run with --assume-clocks-locked once the clocks are pinned")

    base_path = plans.get(args.baseline_label)
    results = {}
    for label, path in plans.items():
        print(f"\n--- {label}: {path}")
        baseline = base_path if (base_path and label != args.baseline_label) else None
        results[label] = bench_one(args.binary, path, args.fixture, args.duty_hz,
                                   args.iters, rails, baseline, args.names)
        r = results[label]
        if "error" in r:
            print(f"  SKIPPED: {r['error']}")
            continue
        if r["rc"] != 0 and not baseline:
            print(f"  r1_parity_check exited {r['rc']} (parity FAIL is exit 1 -- "
                  f"expected for an INT8 plan against the torch reference)")
        for k in ("compact", "duty"):
            if k in r:
                print(f"  {k:<8} p50={r[k]['p50']:.1f}us p99={r[k]['p99']:.1f}us")

    # -- table ---------------------------------------------------------------
    lines = ["# Precision benchmark", "",
             f"- host clock state: `{clocks}`",
             f"- duty cycle: {args.duty_hz:.0f} Hz ({args.iters} paced iterations)",
             f"- fixture: `{args.fixture}`",
             f"- per-dimension comparison baseline: `{args.baseline_label}`", "",
             "| precision | engine bytes | p50 @duty (us) | p99 @duty (us) | p50 compact (us) "
             "| max_abs | VDD_IN idle (W) | VDD_IN run (W) |",
             "|---|---|---|---|---|---|---|---|"]

    def rail_w(d):
        for name in ("VDD_IN", "VDD_SYS_5V0", "POM_5V_IN"):
            if name in d:
                return f"{d[name][0]:.2f}"
        return f"{next(iter(d.values()))[0]:.2f}" if d else "n/a"

    for label in plans:
        r = results[label]
        if "error" in r:
            lines.append(f"| {label} | MISSING | | | | | | |")
            continue
        d = r.get("duty", {})
        c = r.get("compact", {})
        lines.append(
            f"| {label} | {r['bytes']} | {d.get('p50', float('nan')):.1f} | "
            f"{d.get('p99', float('nan')):.1f} | {c.get('p50', float('nan')):.1f} | "
            f"{r.get('max_abs', '?')} | {rail_w(r['idle_w'])} | {rail_w(r['run_w'])} |")

    lines += ["", "## How to read this", "",
              "`max_abs` is against the trained policy for the baseline precision, and",
              "against the baseline ENGINE for the others (that is what `--baseline`",
              "reports). The two are not the same quantity; do not put them in one column",
              "in a slide without saying so.",
              "",
              "A row where all three precisions land within noise of each other is the",
              "expected result at batch 1 and 73k parameters, and it is the evidence that",
              "moves INT8 from 'performance optimisation' to 'controlled perturbation for",
              "the robustness study' (FR-R2/PG-5). It is not a failed measurement.",
              "",
              "Engine size is not guaranteed to fall either: on the Orin, FP16 came out",
              "51% LARGER than FP32 before the TF32 fix, because low precision adds",
              "reformat layers and extra weight copies that a 73k-parameter network cannot",
              "amortise. Report what the bytes column says.",
              "",
              "## Raw output", ""]
    for label in plans:
        r = results[label]
        lines += [f"### {label}", "", "```", r.get("cmd", ""), "",
                  r.get("stdout", r.get("error", "")).rstrip(), "```", ""]

    text = "\n".join(lines)
    if args.out:
        p = pathlib.Path(os.path.expanduser(args.out))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"\n[ok] {p}")
    else:
        print("\n" + text)


if __name__ == "__main__":
    main()
