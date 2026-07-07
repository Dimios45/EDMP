#!/bin/bash
# Reproduction study: how far does in-system guidance tuning close the ~21pp gap to
# the GPD paper's absolute GPD-stitched number (hybrid 92.8%)? Sweeps candidate count
# K (--batch_per) and guidance scale on the GPD-stitched config (single guide +
# faithful RRT stitch), hybrid balanced 400, 8 workers. Keeps the delta thesis; this
# only bounds the recoverable fraction and attributes the residual.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
BASE="--method gpd --guides 1 --pybullet_collision --per_type 100 --dataset hybrid --num_workers $NW --n_control 8 --model_dir ./models/"
LOG=logs/reproduction_main.log
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
# candidate-count K sweep (scale 1x)
run reproduction_k32_scale1   --batch_per 32  --guidance_scale 1
run reproduction_k64_scale1   --batch_per 64  --guidance_scale 1
run reproduction_k128_scale1  --batch_per 128 --guidance_scale 1
# guidance-scale sweep (K=32)
run reproduction_k32_scale2   --batch_per 32  --guidance_scale 2
run reproduction_k32_scale3   --batch_per 32  --guidance_scale 3
# best-combo candidates
run reproduction_k128_scale2  --batch_per 128 --guidance_scale 2
run reproduction_k128_scale3  --batch_per 128 --guidance_scale 3
echo "===== ALL REPRODUCTION RUNS DONE $(date) =====" | tee -a "$LOG"
