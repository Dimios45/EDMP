#!/bin/bash
# Part A rigor CIs: (1) AABB-proxy stitch (5 seeds) -- faithful side reuses
# seed_baseline_* (=faithful stitch, no repair, 72.9+-1.6); (2) EDMP base vs
# EDMP+repair (3 seeds). hybrid 400. Backs "+14pp stitching" and "+10.3pp EDMP".
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; NW=8; LOG=logs/rigor_ci_main.log; mkdir -p logs results
run(){ tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
  echo "===== $(date) START $tag :: $* ====="|tee -a "$LOG"; pids=()
  for k in $(seq 0 $((NW-1))); do $PY run_worker.py "$@" --num_workers $NW --per_type 100 --dataset hybrid --worker_id $k --out "$rdir/${MTH}_w$k.json">"logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || true; done
  $PY merge_results.py --method $MTH --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1|tee -a "$LOG" || true
  echo "===== $(date) DONE $tag ====="|tee -a "$LOG"; }
# (1) AABB-proxy RRT stitch, no repair, 5 seeds (GPD)
MTH=gpd
for s in 0 1 2 3 4; do
  run stitch_proxy_s$s --method gpd --guides 1 --batch_per 32 --n_control 8 --model_dir ./models/ --seed $((1000*(s+1)))
done
# (2) EDMP base vs EDMP+repair, 3 seeds
for s in 0 1 2; do
  MTH=edmp; run edmp_base_s$s   --method edmp --seed $((1000*(s+1)))
  MTH=edmp; run edmp_rep_s$s    --method edmp --pybullet_collision --trajopt --trajopt_iters 60 --seed $((1000*(s+1)))
done
echo "===== RIGOR CI DONE $(date) ====="|tee -a "$LOG"
