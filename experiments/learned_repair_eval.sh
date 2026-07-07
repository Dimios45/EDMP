#!/bin/bash
# C3: evaluate the learned one-shot repair operator vs iterative trajopt & no-repair.
# learned-only and learned+trajopt-polish (warm-start hybrid), 5 seeds, hybrid 400.
# Compare vs existing seed_baseline_* (no repair ~72.9) and seed_repair_* (trajopt ~81.8).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/"
LOG=logs/learned_repair_eval_main.log; mkdir -p logs results
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
    echo "===== $(date) :: START $tag :: $* =====" | tee -a "$LOG"
    pids=(); for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json" > "logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1 | tee -a "$LOG"
    echo "===== $(date) :: DONE $tag =====" | tee -a "$LOG"; }
for s in 0 1 2 3 4; do
    run learned_repair_s$s        --learned_repair                          --seed $((1000*(s+1)))
    run learned_polish_s$s        --learned_repair --trajopt --trajopt_iters 20 --seed $((1000*(s+1)))
done
echo "===== ALL LEARNED-REPAIR-EVAL DONE $(date) =====" | tee -a "$LOG"
