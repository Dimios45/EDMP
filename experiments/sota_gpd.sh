#!/bin/bash
# Repair-Augmented GPD: stack the SOTA config (12-guide ensemble + K + scale2 +
# faithful stitch) and measure where repair takes us vs the published 92.8%.
# Phase 1: component ladder (1 seed). Phase 2: 5-seed headline (ensemble base vs +repair).
# Phase 3: cross-dataset top config (global/both, 3 seeds).
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; NW=8; ENS="1,2,3,4,5,10,11,13,14,16,18,21"
LOG=logs/sota_gpd_main.log; mkdir -p logs results
run(){ tag=$1; ds=$2; shift 2; rdir=results/$tag; mkdir -p "$rdir"
  echo "===== $(date) START $tag ($ds) :: $* ====="|tee -a "$LOG"; pids=()
  for k in $(seq 0 $((NW-1))); do $PY run_worker.py --method gpd --pybullet_collision \
      --per_type 100 --dataset $ds --num_workers $NW --n_control 8 --model_dir ./models/ \
      "$@" --worker_id $k --out "$rdir/gpd_w$k.json" >"logs/${tag}_w$k.log" 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || true; done
  $PY merge_results.py --method gpd --num_workers $NW --results_dir "$rdir" --out "$rdir/summary.json" 2>&1|tee -a "$LOG" || true
  echo "===== $(date) DONE $tag ====="|tee -a "$LOG"; }

# ---- Phase 1: component ladder (1 seed, hybrid) ----
run sota_L1_ens        hybrid --guides "$ENS" --batch_per 10 --guidance_scale 1.0 --seed 1000
run sota_L2_ens_sc2    hybrid --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --seed 1000
run sota_L3_ens_rep    hybrid --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --trajopt --trajopt_iters 60 --seed 1000
run sota_L4_ensK192_rep hybrid --guides "$ENS" --batch_per 16 --guidance_scale 2.0 --trajopt --trajopt_iters 60 --seed 1000

# ---- Phase 2: 5-seed headline (hybrid): ensemble base vs +repair ----
for s in 0 1 2 3 4; do
  run sota_ensbase_s$s hybrid --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --seed $((1000*(s+1)))
  run sota_ensrep_s$s  hybrid --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --trajopt --trajopt_iters 60 --seed $((1000*(s+1)))
done

# ---- Phase 3: cross-dataset top config (3 seeds, global/both) ----
for ds in global both; do for s in 0 1 2; do
  run sota_${ds}_base_s$s $ds --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --seed $((1000*(s+1)))
  run sota_${ds}_rep_s$s  $ds --guides "$ENS" --batch_per 10 --guidance_scale 2.0 --trajopt --trajopt_iters 60 --seed $((1000*(s+1)))
done; done
echo "===== SOTA GPD DONE $(date) ====="|tee -a "$LOG"
