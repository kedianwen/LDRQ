#!/usr/bin/env python3
"""Record real observations off the running robot, for INT8 calibration.

INT8 quantisation picks a scale per activation tensor, and it picks it by
watching activations while the network runs on a calibration set. So the
calibration set decides the scales, and a set drawn from the wrong distribution
produces an engine that is quantised for states the policy never visits.

That is why this tool exists instead of reusing `parity_fixture.bin`: the
fixture's inputs are Gaussian noise. Noise is the right thing for a parity
fixture (it stresses numerics) and the wrong thing for calibration (it drives
activations the policy never reaches while walking).

  # on the robot, WHILE IT IS WALKING, developer mode, stack running
  python3 record_calib_obs.py --seconds 60 --out ~/orin_commissioning/calib_obs.bin

  # or, better: don't run a second 50 Hz subscriber on the robot at all.
  # walk_metrics.py already records ~/obs, so convert its save instead --
  # one walk, one recording, both artefacts, and one less process competing
  # with the control loop for CPU.
  python3 record_calib_obs.py --from-walk ~/orin_commissioning/walk_pg2.json \
      --out ~/orin_commissioning/calib_obs.bin

  # later, anywhere, no ROS needed
  python3 record_calib_obs.py --stats ~/orin_commissioning/calib_obs.bin

Output layout (little-endian), the format r1_build_engine --calib reads:
    "R1CB" | u32 version=1 | u32 n | u32 dim | f32 obs[n * dim]
"""
import argparse
import json
import math
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPEC = HERE.parent / "interface" / "policy_interface.json"


def load_terms():
    """Term layout, so the coverage report can name what it is reporting on.

    Isaac Lab flattens the history PER TERM, not per frame: each term occupies
    history*width contiguous floats, oldest frame first. The newest frame of a
    term is therefore its LAST `width` floats, not a slice of some frame block.
    """
    spec = json.loads(SPEC.read_text())["observation"]
    hist = spec["history_length"]
    out, off = [], 0
    for t in spec["terms"]:
        w = t["width"]
        out.append({"name": t["name"], "width": w, "start": off, "newest": off + (hist - 1) * w})
        off += w * hist
    return out, spec["total_dim"]


def newest(sample, term):
    return sample[term["newest"]:term["newest"] + term["width"]]


def coverage(samples, terms):
    """Does this recording look like walking, or like a robot standing still?

    A calibration set recorded while standing is the classic way to get an INT8
    engine that is quantised perfectly for a state the robot is not in when it
    matters. It is also invisible afterwards -- the engine builds, the parity
    numbers look fine, and the failure only appears in motion.
    """
    print(f"\n=== calibration coverage ({len(samples)} samples) ===")
    flags = []
    for t in terms:
        vals = [v for s in samples for v in newest(s, t)]
        amax = max(abs(v) for v in vals)
        mean_abs = sum(abs(v) for v in vals) / len(vals)
        print(f"  {t['name']:<20} |max|={amax:8.4f}   mean|.|={mean_abs:8.4f}")
        if t["name"] == "velocity_commands" and amax < 0.05:
            flags.append("velocity_commands is ~0 for the whole recording: the robot was "
                         "commanded to stand, so this set does not cover walking.")
        if t["name"] == "joint_vel" and mean_abs < 0.05:
            flags.append(f"joint_vel mean|.|={mean_abs:.4f} is tiny: the joints barely moved. "
                         "Recorded while hanging idle?")
    # Duplicate frames mean the subscription stalled and the same message was
    # sampled repeatedly -- n looks fine, the distribution does not.
    uniq = len({tuple(s[-24:]) for s in samples})
    print(f"  distinct newest-action frames: {uniq}/{len(samples)}")
    if uniq < len(samples) * 0.5:
        flags.append(f"only {uniq} of {len(samples)} frames are distinct: the recording is "
                     "mostly repeats, so its effective size is much smaller than n.")
    return flags


def write_r1cb(path, samples, dim):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(b"R1CB")
        fh.write(struct.pack("<III", 1, len(samples), dim))
        for s in samples:
            fh.write(struct.pack(f"<{dim}f", *s))
    print(f"\n[ok] wrote {p} ({p.stat().st_size / 1024:.0f} KiB, {len(samples)} x {dim})")


def read_r1cb(path):
    raw = pathlib.Path(path).read_bytes()
    if raw[:4] != b"R1CB":
        sys.exit(f"{path}: magic is {raw[:4]!r}, not R1CB")
    version, n, dim = struct.unpack_from("<III", raw, 4)
    if version != 1:
        sys.exit(f"{path}: unsupported version {version}")
    body = struct.unpack_from(f"<{n * dim}f", raw, 16)
    return [list(body[i * dim:(i + 1) * dim]) for i in range(n)], dim


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", help="R1CB file to write")
    ap.add_argument("--topic", default="/r1_hw_bridge/obs")
    ap.add_argument("--max-samples", type=int, default=6000,
                    help="cap; 6000 = 120 s at 50 Hz. Entropy calibration at batch 1 "
                         "walks the whole set once, so a huge set only costs build time.")
    ap.add_argument("--stats", help="report on an existing R1CB file and exit (no ROS)")
    ap.add_argument("--from-walk", help="convert the obs stream out of a walk_metrics.py "
                                       "--save JSON instead of subscribing (no ROS needed)")
    args = ap.parse_args()

    terms, total_dim = load_terms()

    if args.stats:
        samples, dim = read_r1cb(args.stats)
        if dim != total_dim:
            print(f"[warn] file is {dim}-dim, interface says {total_dim}; term names below "
                  f"are probably misaligned")
        for f in coverage(samples, terms):
            print(f"  ! {f}")
        return

    if not args.out:
        sys.exit("--out is required when recording (or pass --stats to inspect a file)")

    if args.from_walk:
        blob = json.loads(pathlib.Path(args.from_walk).read_text())
        samples = [v for _, v in blob.get("obs", [])]
        if not samples:
            sys.exit(f"{args.from_walk} has no 'obs' entries -- was it saved by "
                     f"walk_metrics.py --save?")
        dim = len(samples[0])
        if dim != total_dim:
            sys.exit(f"observations are {dim}-dim, the interface says {total_dim}")
        print(f"converted {len(samples)} observations from {args.from_walk}")
        if blob.get("note"):
            print(f"  recording note: {blob['note']}")
        for f in coverage(samples, terms):
            print(f"  ! {f}")
        write_r1cb(args.out, samples, dim)
        return

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32MultiArray

    class Rec(Node):
        def __init__(self):
            super().__init__("r1_calib_recorder")
            self.samples = []
            self.create_subscription(
                Float32MultiArray, args.topic,
                lambda m: self.samples.append(list(m.data)), qos_profile_sensor_data)

    rclpy.init()
    node = Rec()
    print(f"recording {args.seconds:.0f}s from {args.topic}")
    print("WALK THE ROBOT during this window. A standing recording calibrates INT8")
    print("for standing, and nothing downstream will tell you that it did.")
    end = node.get_clock().now().nanoseconds + int(args.seconds * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(node.samples) >= args.max_samples:
            break
    samples = node.samples
    node.destroy_node()
    rclpy.shutdown()

    if not samples:
        # Same three false negatives this project has lost time to before.
        print("\nno samples. Before suspecting the bridge:")
        print("  echo $ROS_LOCALHOST_ONLY   # must be 1, and must match the stack's terminal")
        print("  ros2 daemon stop           # foxy caches a stale graph snapshot")
        print("  pgrep -f r1_hw_bridge      # -f is required; comm truncates to 15 chars")
        sys.exit(1)

    dim = len(samples[0])
    ragged = [len(s) for s in samples if len(s) != dim]
    if ragged:
        sys.exit(f"inconsistent sample widths: {dim} vs {set(ragged)}")
    if dim != total_dim:
        sys.exit(f"observations are {dim}-dim, the interface says {total_dim}: "
                 f"the bridge and the policy spec disagree, fix that before calibrating")

    rate = len(samples) / args.seconds
    print(f"\ngot {len(samples)} samples ({rate:.1f} Hz)")
    if rate < 40:
        print(f"  ! {rate:.1f} Hz is well under 50: frames were dropped, so this set is "
              f"a sparse sample of the trajectory rather than a dense one.")

    flags = coverage(samples, terms)
    for f in flags:
        print(f"  ! {f}")
    if flags:
        print("\nThe file is still written -- these are judgements, not errors. But an\n"
              "INT8 engine built on a flagged set must not be reported as 'calibrated on\n"
              "the deployment distribution', because it is not.")
    if len(samples) < 256:
        print(f"\n! only {len(samples)} samples; entropy calibration wants several hundred "
              f"at minimum (>= 1000 preferred = 20 s at 50 Hz).")

    write_r1cb(args.out, samples, dim)
    print("\nnext:")
    print("  r1_build_engine --onnx $R1_DEPLOY_ROOT/artifacts/policy.onnx \\")
    print(f"      --plan $R1_DEPLOY_ROOT/artifacts/policy_int8.plan --int8 --calib {args.out} \\")
    print("      --calib-cache $R1_DEPLOY_ROOT/artifacts/int8_calib.table")
    print("  (delete the .table first if you re-record -- an existing cache is REUSED")
    print("   and the new recording would be ignored silently.)")


if __name__ == "__main__":
    main()
