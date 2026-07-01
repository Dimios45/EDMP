#!/bin/bash
# #6: per-intervention seeds for the null results, so D3/D8 get their own n=5 CIs
# (rather than only being bounded by the baseline sigma).
#   D3 = scene-conditioned denoiser + guidance (cfg_w=1)
#   D8 = 1M-step long prior (plain GPD)
# hybrid balanced 400, 5 seeds each. Compare vs base 72.9+/-1.6 (seed_stats.json).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
SEEDS="0 1 2 3 4"
COMMON="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8"
LOG=logs/bench_seeds2_main.log
mkdir -p logs results
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
    echo "===== $(date) :: START $tag :: $* =====" | tee -a "$LOG"
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $COMMON "$@" --worker_id $k --out "$rdir/gpd_w$k.json" > "logs/${tag}_w$k.log" 2>&1 &
        pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1 | tee -a "$LOG"
    echo "===== $(date) :: DONE $tag =====" | tee -a "$LOG"
}
for s in $SEEDS; do
    run seed_d3_s$s   --model_dir ./models/      --conditional --cfg_weight 1 --seed $((1000*(s+1)))
    run seed_d8_s$s   --model_dir ./models_long/                              --seed $((1000*(s+1)))
done
echo "===== ALL SEEDS2 DONE $(date) =====" | tee -a "$LOG"
