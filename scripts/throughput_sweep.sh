#!/usr/bin/env bash
# Week01 RK-7: sweep num_envs on the official H1 flat-velocity task until OOM,
# to find (a) the max num_envs this 2080Ti can hold, (b) steady-state fps at
# each level, (c) per-million-env-step wall time derived from fps.
#
# Usage: ./scripts/throughput_sweep.sh (from the project root, or anywhere)
# Output: docs/throughput_sweep.md (this script writes it incrementally so a
#         crash/OOM on the last level still leaves prior results on disk).

# NOTE: no `set -u` here -- it leaks into ./isaaclab.sh via bash's SHELLOPTS
# inheritance, and IsaacLab's setup_conda_env.sh references $ZSH_VERSION
# unguarded, which aborts immediately under nounset.
set -o pipefail

cd "$(dirname "$0")/.."
mkdir -p docs
REPORT=docs/throughput_sweep.md
ISAACLAB=~/IsaacLab
TASK=Isaac-Velocity-Flat-H1-v0
MAX_ITER=40
LEVELS=(64 256 1024 2048 4096 8192 16384)

echo "# H1 flat-velocity throughput sweep (2080Ti 11GB), RK-7" > "$REPORT"
echo "" >> "$REPORT"
echo "Task: $TASK, max_iterations=$MAX_ITER per level." >> "$REPORT"
echo "" >> "$REPORT"
echo "| num_envs | result | peak GPU mem (MiB) | steady total_fps | per-million-env-steps (s) |" >> "$REPORT"
echo "|---|---|---|---|---|" >> "$REPORT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

for N in "${LEVELS[@]}"; do
    echo "=== num_envs=$N ==="
    LOG="/tmp/sweep_${N}.log"
    START_EPOCH=$(date +%s)

    # background GPU memory sampler
    MEMLOG="/tmp/sweep_${N}_mem.log"
    : > "$MEMLOG"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$MEMLOG"; sleep 1; done ) &
    SAMPLER_PID=$!

    cd "$ISAACLAB"
    timeout 300 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task="$TASK" --headless --num_envs "$N" --max_iterations "$MAX_ITER" \
        > "$LOG" 2>&1
    STATUS=$?
    cd - > /dev/null

    kill "$SAMPLER_PID" 2>/dev/null
    PEAK_MEM=$(sort -n "$MEMLOG" | tail -1)
    PEAK_MEM=${PEAK_MEM:-NA}

    if [ $STATUS -ne 0 ] || grep -qiE "out of memory|CUDA error|illegal memory access" "$LOG"; then
        echo "[SWEEP] num_envs=$N FAILED (status=$STATUS) -- treating as OOM ceiling."
        echo "| $N | OOM/crash (exit $STATUS) | $PEAK_MEM | - | - |" >> "$REPORT"
        break
    fi

    # newest run dir created for this task after START_EPOCH
    RUNDIR=$(find "$ISAACLAB/logs/rsl_rl" -maxdepth 2 -type d -newermt "@$START_EPOCH" | sort | tail -1)
    FPS=$(python3 - "$RUNDIR" <<'EOF'
import sys, glob
from tensorboard.backend.event_processing import event_accumulator
rundir = sys.argv[1]
files = glob.glob(rundir + "/events.out.tfevents.*")
if not files:
    print("NA"); sys.exit()
ea = event_accumulator.EventAccumulator(files[0], size_guidance={"scalars": 0})
ea.Reload()
vals = ea.Scalars("Perf/total_fps")
last = vals[-min(10, len(vals)):]
print(f"{sum(v.value for v in last) / len(last):.0f}")
EOF
)
    if [ "$FPS" = "NA" ] || [ -z "$FPS" ]; then
        PER_M="NA"
    else
        PER_M=$(python3 -c "print(f'{1e6/${FPS}:.1f}')")
    fi
    echo "[SWEEP] num_envs=$N OK fps=$FPS peak_mem=${PEAK_MEM}MiB per_million_steps=${PER_M}s"
    echo "| $N | OK | $PEAK_MEM | $FPS | $PER_M |" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "Done. See per-level raw logs in /tmp/sweep_<N>.log" >> "$REPORT"
echo "[SWEEP] Report written to $REPORT"
