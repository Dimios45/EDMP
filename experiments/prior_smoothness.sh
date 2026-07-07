#!/bin/bash
# Smoothness-penalty sweep: control-point curvature penalty on N16@20k.
# Tests "compression is a regularizer": does re-imposing smoothness on the
# high-capacity (n=16) model recover / exceed N8 (71.8%)?
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
NW=6
COMMON="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 60 --dataset hybrid --num_workers $NW --n_control 16 --model_dir models/_n16_20k/"

for lam in 0.0 0.01 0.03 0.06 0.12; do
    tag=$(echo $lam | tr '.' 'p')
    rdir=results/smoothness_lam$tag
    mkdir -p "$rdir" logs
    echo "===== $(date) :: smoothness_weight=$lam ====="
    pids=()
    for w in $(seq 0 $((NW-1))); do
        $PY run_worker.py $COMMON --smoothness_weight $lam \
            --worker_id $w --out "$rdir/gpd_w$w.json" \
            > "logs/smooth_${tag}_w$w.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    echo "----- lambda=$lam summary -----"
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" \
        --out "$rdir/summary.json"
done
echo "===== SMOOTH SWEEP DONE $(date) ====="
