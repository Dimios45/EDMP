#!/bin/bash
# Batch #2 (does repair generalize beyond GPD? -> EDMP base vs EDMP+repair)
# and #3 (failure-mode shift after repair on GPD). Sequential (avoid contention).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
LOG=logs/bench_batch23_main.log
mkdir -p logs results
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
    echo "===== $(date) :: START $tag :: $* =====" | tee -a "$LOG"
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py --method edmp --dataset hybrid --per_type 100 --num_workers $NW \
            "$@" --worker_id $k --out "$rdir/edmp_w$k.json" > "logs/${tag}_w$k.log" 2>&1 &
        pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method edmp --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1 | tee -a "$LOG"
    echo "===== $(date) :: DONE $tag =====" | tee -a "$LOG"
}

# ---- #2: EDMP base vs EDMP + same trajopt repair (balanced 400 hybrid) ----
run edmp_baseline
run edmp_repair   --pybullet_collision --trajopt --trajopt_iters 60

# ---- #3: failure-mode shift after repair (GPD, 50/type = 200 scenes) ----
echo "===== $(date) :: START diag_d6 =====" | tee -a "$LOG"
$PY diag_failures.py 50 repair 2>&1 | tee -a "$LOG"
echo "===== $(date) :: DONE diag_d6 =====" | tee -a "$LOG"

echo "===== ALL BATCH23 DONE $(date) =====" | tee -a "$LOG"
