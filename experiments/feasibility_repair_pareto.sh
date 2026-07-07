#!/bin/bash
# Trajopt hyperparameter ablation -> time/accuracy Pareto.
# One-factor-at-a-time around the default anchor (iters=60, densify=3, mid_w=1.0).
# hybrid, 100/type (400 scenes), 8 workers. summary.json carries avg plan_time.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --trajopt"
LOG=logs/bench_d7_main.log
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

# Anchor (iters=60, densify=3, mid_w=1.0)
run repair_pareto_default      --trajopt_iters 60  --trajopt_densify 3 --trajopt_mid_w 1.0

# iters sweep (densify=3, mid_w=1.0); iters=0 == densify-only floor
run repair_pareto_iters0      --trajopt_iters 0   --trajopt_densify 3 --trajopt_mid_w 1.0
run repair_pareto_iters20     --trajopt_iters 20  --trajopt_densify 3 --trajopt_mid_w 1.0
run repair_pareto_iters40     --trajopt_iters 40  --trajopt_densify 3 --trajopt_mid_w 1.0
run repair_pareto_iters100    --trajopt_iters 100 --trajopt_densify 3 --trajopt_mid_w 1.0

# densify sweep (iters=60, mid_w=1.0); densify=1 == no output densification
run repair_pareto_dens1       --trajopt_iters 60  --trajopt_densify 1 --trajopt_mid_w 1.0
run repair_pareto_dens2       --trajopt_iters 60  --trajopt_densify 2 --trajopt_mid_w 1.0
run repair_pareto_dens5       --trajopt_iters 60  --trajopt_densify 5 --trajopt_mid_w 1.0

# midpoint-weight sweep (iters=60, densify=3); mid_w=0 == waypoints-only objective
run repair_pareto_midw0        --trajopt_iters 60  --trajopt_densify 3 --trajopt_mid_w 0.0
run repair_pareto_midw0p5      --trajopt_iters 60  --trajopt_densify 3 --trajopt_mid_w 0.5
run repair_pareto_midw2        --trajopt_iters 60  --trajopt_densify 3 --trajopt_mid_w 2.0

echo "===== REPAIR-PARETO DONE $(date) =====" | tee -a "$LOG"
