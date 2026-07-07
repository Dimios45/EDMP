#!/bin/bash
# Polynomial-capacity ablation: n=16 vs n=8, identical GPD-stitched config, 100 scenes/type hybrid.
# Single-variable comparison (only n_control differs; same 20k budget).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
NW=8
COMMON="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW"

run_model () {   # $1=tag  $2=n_control  $3=model_dir
    tag=$1; nc=$2; mdir=$3
    rdir=results/capacity_$tag
    mkdir -p "$rdir" logs
    echo "===== $(date) :: launching $tag (n_control=$nc, model_dir=$mdir) ====="
    pids=()
    for w in $(seq 0 $((NW-1))); do
        $PY run_worker.py $COMMON --n_control $nc --model_dir "$mdir" \
            --worker_id $w --out "$rdir/gpd_w$w.json" \
            > "logs/d4_${tag}_w$w.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    echo "===== $(date) :: $tag done — summary ====="
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" \
        --out "$rdir/summary.json"
}

run_model n8_20k  8  models/_n8_20k/
run_model n16_20k 16 ./models/

echo "===== ALL DONE $(date) ====="
