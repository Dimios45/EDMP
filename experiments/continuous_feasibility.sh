#!/bin/bash
# Continuous-feasibility study E1: repair-objective ablation over edge_samples S
# (S=1 = original single-midpoint; S>1 = dense continuous term). GPD-stitched +
# trajopt, hybrid balanced 400, 8 workers. 5 seeds for the S=1 vs S=4 paired
# comparison (CIs); S=2, S=8 single-seed for the trend.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --trajopt --trajopt_iters 60"
LOG=logs/continuous_feasibility_main.log
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
for s in 0 1 2 3 4; do
    run continuous_edge1_s$s  --repair_edge_samples 1  --seed $((1000*(s+1)))
    run continuous_edge4_s$s  --repair_edge_samples 4  --seed $((1000*(s+1)))
done
run continuous_edge2_s0  --repair_edge_samples 2  --seed 1000
run continuous_edge8_s0  --repair_edge_samples 8  --seed 1000
echo "===== ALL CONTINUOUS-FEASIBILITY RUNS DONE $(date) =====" | tee -a "$LOG"
