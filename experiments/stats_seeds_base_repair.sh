#!/bin/bash
# Multi-seed CIs on the headline base-vs-D6 comparison.
# 5 seeds x {plain GPD, GPD+trajopt}, GPDS config, hybrid balanced 400, 8 workers.
# Establishes baseline SR variance (sigma) + a CI on the +10pp trajopt delta.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
SEEDS="0 1 2 3 4"
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/"
LOG=logs/bench_seeds_main.log
mkdir -p logs results
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
    echo "===== $(date) :: START $tag :: $* =====" | tee -a "$LOG"
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json" > "logs/${tag}_w$k.log" 2>&1 &
        pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1 | tee -a "$LOG"
    echo "===== $(date) :: DONE $tag =====" | tee -a "$LOG"
}
for s in $SEEDS; do
    run seed_base_s$s            --seed $((1000*(s+1)))
    run seed_d6_s$s   --trajopt --trajopt_iters 60 --seed $((1000*(s+1)))
done
echo "===== ALL SEED RUNS DONE $(date) =====" | tee -a "$LOG"
