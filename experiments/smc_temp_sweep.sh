set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --seed 1000"
LOG=logs/smc_temp_main.log; mkdir -p logs results
run(){ tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
  echo "===== $(date) $tag $* ====="|tee -a "$LOG"; pids=()
  for k in $(seq 0 $((NW-1))); do $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json">"logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done
  $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1|tee -a "$LOG"; }
for T in 2 8 30; do run smc_temp${T} --smc_every 8 --smc_temp $T; done
echo "SMC TEMP SWEEP DONE $(date)"|tee -a "$LOG"
