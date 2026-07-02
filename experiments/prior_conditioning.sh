#!/bin/bash
# Scene-conditioned (CFG) GPD vs unconditional baseline (71.8%).
# Sweep CFG weight; GPD-stitched config (1 guide + faithful-PyBullet RRT stitch).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
COMMON="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --conditional --model_dir ./models/"
for w in 1.0 2.0 3.0; do
    tag=${w%%.*}; rdir=results/conditioning_cfgw$tag; mkdir -p "$rdir" logs
    echo "===== $(date) :: conditional CFG w=$w ====="
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $COMMON --cfg_weight $w --worker_id $k --out "$rdir/gpd_w$k.json" \
            > "logs/d3_${tag}_w$k.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json"
done
echo "===== CONDITIONING-SWEEP DONE $(date) ====="
