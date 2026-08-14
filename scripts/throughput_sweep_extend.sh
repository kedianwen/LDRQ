#!/usr/bin/env bash
# One-off continuation of throughput_sweep.sh: appends more num_envs levels to
# the existing docs/throughput_sweep.md instead of re-running from scratch.
# Usage: ./scripts/throughput_sweep_extend.sh 24576 32768 ...

set -o pipefail

cd "$(dirname "$0")/.."
REPORT=docs/throughput_sweep.md
ISAACLAB=~/IsaacLab
TASK=Isaac-Velocity-Flat-H1-v0
MAX_ITER=40

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

for N in "$@"; do
    echo "=== num_envs=$N ==="
    LOG="/tmp/sweep_${N}.log"
    START_EPOCH=$(date +%s)

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
        echo "[SWEEP] num_envs=$N FAILED (status=$STATUS) -- OOM ceiling."
        echo "| $N | OOM/crash (exit $STATUS) | $PEAK_MEM | - | - |" >> "$REPORT"
        break
    fi

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

echo "[SWEEP] Extend done."
