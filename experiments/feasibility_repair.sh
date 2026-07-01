#!/bin/bash
# D6: continuous-feasibility repair. base vs densify-only vs trajopt+densify.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/"
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir" logs
    echo "===== $(date) :: $tag :: $* ====="
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json" > "logs/${tag}_w$k.log" 2>&1 &
        pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json"
}
run d6_base
run d6_dense   --trajopt --trajopt_iters 0
run d6_full    --trajopt --trajopt_iters 60
echo "===== D6 DONE $(date) ====="
