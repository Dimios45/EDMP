#!/bin/bash
# B3: compute-optimal allocation. Grid over candidate budget K, repair iterations, and
# oracle-select on/off; per-component timers (t_diffuse/t_select/t_repair) logged in each
# summary. Build the SR-vs-time Pareto + an allocation rule. hybrid 200 (per_type 50), 1 seed.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --pybullet_collision --per_type 50 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --seed 1000"
LOG=logs/compute_allocation_main.log; mkdir -p logs results
run () { tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
    echo "===== $(date) :: START $tag :: $* =====" | tee -a "$LOG"
    pids=(); for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json" > "logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1 | tee -a "$LOG"
    echo "===== $(date) :: DONE $tag =====" | tee -a "$LOG"; }
for K in 32 128; do
  for IT in 0 20 60; do
    for ORC in 0 1; do
      flags="--batch_per $K"; [ "$IT" -gt 0 ] && flags="$flags --trajopt --trajopt_iters $IT"
      [ "$ORC" -eq 1 ] && flags="$flags --oracle_select"
      run alloc_k${K}_it${IT}_orc${ORC} $flags
    done
  done
done
echo "===== ALL COMPUTE-ALLOCATION DONE $(date) =====" | tee -a "$LOG"
