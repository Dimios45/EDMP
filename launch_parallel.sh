#!/bin/bash
# Parallel benchmark: GPD + EDMP run simultaneously.
#
# GPD:  1 worker  → ~45 min  (GPU-native, T=64)
# EDMP: 1 worker  → ~3-4 hr  (GPU-native, T=255)
# Wall time: ~3-4 hr  (down from 12 hr sequential)
#
# Optional: --mini for 50-scene quick test

cd /home/sra/EDMP
source .venv/bin/activate

FULL_FLAG="--full"
if [[ "$1" == "--mini" ]]; then
    FULL_FLAG=""
    echo "Running MINI benchmark (50 scenes per method)"
else
    echo "Running FULL benchmark (1800 scenes per method)"
fi

mkdir -p results logs

echo "========================================"
echo "START: $(date)"
echo "========================================"

# Launch GPD worker in background
echo ">>> Starting GPD worker..."
python run_worker.py --method gpd --worker_id 0 --num_workers 1 \
    $FULL_FLAG --out results/gpd_w0.json \
    > logs/gpd_w0.log 2>&1 &
GPD_PID=$!
echo "GPD PID: $GPD_PID"

# Launch EDMP worker in background
echo ">>> Starting EDMP worker..."
python run_worker.py --method edmp --worker_id 0 --num_workers 1 \
    $FULL_FLAG --out results/edmp_w0.json \
    > logs/edmp_w0.log 2>&1 &
EDMP_PID=$!
echo "EDMP PID: $EDMP_PID"

echo ""
echo "Both workers launched. Monitoring..."
echo "  GPD  log: logs/gpd_w0.log"
echo "  EDMP log: logs/edmp_w0.log"
echo ""
echo "Monitor with:"
echo "  tail -f logs/gpd_w0.log"
echo "  tail -f logs/edmp_w0.log"
echo "  ./check_progress.sh"
echo ""

# Wait for both to finish
wait $GPD_PID
GPD_EXIT=$?
echo "GPD  done at $(date) (exit=$GPD_EXIT)"

wait $EDMP_PID
EDMP_EXIT=$?
echo "EDMP done at $(date) (exit=$EDMP_EXIT)"

echo ""
echo "========================================"
echo "ALL DONE: $(date)"
echo "========================================"

# Merge and print final summary
python merge_results.py --method both --num_workers 1 \
    --results_dir results --out results/compare_final.json
