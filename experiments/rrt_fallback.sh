#!/bin/bash
# Last shot at SOTA: learned (1-guide + trajopt repair) + RRT-Connect completeness
# fallback on the residual failures. Complete planner on solvable scenes -> aims 90%+.
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; NW=8; LOG=logs/rrt_fb_main.log; mkdir -p logs results
BASE="--method gpd --guides 1 --batch_per 32 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/ --trajopt --trajopt_iters 60"
run(){ tag=$1; shift; rdir=results/$tag; mkdir -p "$rdir"
  echo "===== $(date) START $tag :: $* ====="|tee -a "$LOG"; pids=()
  for k in $(seq 0 $((NW-1))); do $PY run_worker.py $BASE "$@" --worker_id $k --out "$rdir/gpd_w$k.json">"logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || true; done
  $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1|tee -a "$LOG" || true
  echo "===== $(date) DONE $tag ====="|tee -a "$LOG"; }
for s in 0 1 2; do run rrtfb_s$s --rrt_fallback --rrt_fb_iters 2000 --seed $((1000*(s+1))); done
run rrtfb_hi_s0 --rrt_fallback --rrt_fb_iters 6000 --seed 1000
echo "===== RRT FALLBACK DONE $(date) ====="|tee -a "$LOG"
