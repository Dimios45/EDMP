#!/bin/bash
# C1: generate (stitched-seed -> trajopt-repaired) pairs for learned-repair training.
# Runs the repair pipeline with --save_repair_pairs on all 3 datasets (per_type 150 =
# 600 scenes/dataset) and stashes per-worker jsonl into per-dataset dirs.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
LOG=logs/repair_datagen_main.log; mkdir -p logs results
for ds in hybrid global both; do
    echo "===== $(date) :: datagen $ds =====" | tee -a "$LOG"
    rm -rf results/repair_pairs; mkdir -p results/repair_pairs
    pids=(); for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py --method gpd --guides 1 --batch_per 32 --pybullet_collision \
            --per_type 150 --dataset $ds --num_workers $NW --n_control 8 --model_dir ./models/ \
            --trajopt --trajopt_iters 60 --save_repair_pairs \
            --worker_id $k --out "results/_datagen_${ds}_w$k.json" > "logs/datagen_${ds}_w$k.log" 2>&1 & pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    mv results/repair_pairs "results/repair_pairs_${ds}"
    echo "  pairs: $(cat results/repair_pairs_${ds}/*.jsonl | wc -l)" | tee -a "$LOG"
done
echo "===== ALL DATAGEN DONE $(date) total pairs: $(cat results/repair_pairs_*/*.jsonl | wc -l) =====" | tee -a "$LOG"
