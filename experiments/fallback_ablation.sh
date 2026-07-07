#!/bin/bash
# Honest ablation of the RRT-Connect completeness fallback. De-cheated: default trigger
# is the plan-time static gate (--fb_trigger static), never the execution grader.
# Parts: A honesty (static vs exec cheat, R_TRIG sweep), B attribution (RRT-only, +/-prior,
# +/-repair), C budget/time, D generalization (global/both/EDMP).
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; NW=8; LOG=logs/fb_ablation_main.log; mkdir -p logs results
COMMON="--pybullet_collision --num_workers $NW --n_control 8 --model_dir ./models/"
FB="--rrt_fallback --rrt_fb_iters 6000 --rrt_fb_time 10 --rrt_fb_trigger_r 8"
run(){ tag=$1; mth=$2; shift 2; rdir=results/$tag; mkdir -p "$rdir"
  echo "===== $(date) START $tag [$mth] :: $* ====="|tee -a "$LOG"; pids=()
  for k in $(seq 0 $((NW-1))); do $PY run_worker.py --method $mth $COMMON "$@" --worker_id $k --out "$rdir/${mth}_w$k.json">"logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || true; done
  $PY merge_results.py --method $mth --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1|tee -a "$LOG" || true
  echo "===== $(date) DONE $tag ====="|tee -a "$LOG"; }
G="--guides 1 --batch_per 32 --per_type 100 --dataset hybrid --trajopt --trajopt_iters 60"

# ---- Part A: honesty (5-seed static) + R_TRIG sweep + exec-cheat comparison ----
for s in 0 1 2 3 4; do run fbA_static_s$s gpd $G $FB --fb_trigger static --seed $((1000*(s+1))); done
run fbA_rtrig1_s0  gpd $G $FB --fb_trigger static --rrt_fb_trigger_r 1  --seed 1000
run fbA_rtrig32_s0 gpd $G $FB --fb_trigger static --rrt_fb_trigger_r 32 --seed 1000
for s in 0 1 2; do run fbA_exec_s$s gpd $G $FB --fb_trigger exec --seed $((1000*(s+1))); done

# ---- Part B: attribution ladder (3 seeds; full stack = fbA_static) ----
for s in 0 1 2; do
  run fbB_rrtonly_s$s gpd --guides 1 --batch_per 32 --per_type 100 --dataset hybrid --naive_seed linear --no_stitch $FB --fb_trigger static --seed $((1000*(s+1)))
  run fbB_linrep_s$s  gpd --guides 1 --batch_per 32 --per_type 100 --dataset hybrid --naive_seed linear --trajopt --trajopt_iters 60 $FB --fb_trigger static --seed $((1000*(s+1)))
  run fbB_difffb_s$s  gpd --guides 1 --batch_per 32 --per_type 100 --dataset hybrid $FB --fb_trigger static --seed $((1000*(s+1)))
done

# ---- Part C: budget sweep (1 seed; iters 6000 = fbA_static_s0) ----
run fbC_it500_s0  gpd $G --rrt_fallback --rrt_fb_iters 500  --rrt_fb_time 10 --rrt_fb_trigger_r 8 --fb_trigger static --seed 1000
run fbC_it2000_s0 gpd $G --rrt_fallback --rrt_fb_iters 2000 --rrt_fb_time 10 --rrt_fb_trigger_r 8 --fb_trigger static --seed 1000

# ---- Part D: generalization ----
for ds in global both; do for s in 0 1 2; do
  run fbD_${ds}_s$s gpd --guides 1 --batch_per 32 --per_type 100 --dataset $ds --trajopt --trajopt_iters 60 $FB --fb_trigger static --seed $((1000*(s+1)))
done; done
run fbD_edmp_s0 edmp --per_type 50 --dataset hybrid --trajopt --trajopt_iters 60 $FB --fb_trigger static --seed 1000

echo "===== FB ABLATION DONE $(date) ====="|tee -a "$LOG"
