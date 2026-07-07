#!/bin/bash
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/"
run () {  # $1=tag  $2..=extra flags
    tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir" logs
    echo "===== $(date) :: $tag :: $* ====="
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json" \
            > "logs/${tag}_w$k.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json"
}
# C: conditioned, NO cost-gradient guidance (CFG-only prior)
run conditioning_cfg_only --conditional --cfg_weight 2.0 --guidance_scale 0.0
# D: unconditioned, NO guidance (prior-only control)
run conditioning_prior_only --guidance_scale 0.0
echo "===== CONDITIONING-ABLATION DONE $(date) ====="
