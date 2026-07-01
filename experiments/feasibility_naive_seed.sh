#!/bin/bash
# Control #1: does the diffusion prior matter as an INITIALIZER?
# Replace the diffusion seed with a straight-line joint-space seed, run the
# IDENTICAL downstream (RRT stitch [+ trajopt]). 3 seeds (RRT has RNG variance).
# Compare vs diffusion: base 72.9+/-1.6, d6 81.8+/-1.3 (results/seed_stats.json).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
SEEDS="0 1 2"
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --naive_seed linear"
LOG=logs/bench_naive_main.log
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
    run naive_base_s$s            --seed $((1000*(s+1)))
    run naive_d6_s$s   --trajopt --trajopt_iters 60 --seed $((1000*(s+1)))
done
echo "===== ALL NAIVE RUNS DONE $(date) =====" | tee -a "$LOG"
