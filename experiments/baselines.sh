#!/bin/bash
# Matched-split baseline table: GPD baseline, GPD+repair, EDMP across
# global/hybrid/both solvable, natural-weighted 600/dataset (200/100/100/200).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python; NW=8
CAPS="200,100,100,200"
runp () {  # $1=tag  $2..=run_worker flags
    tag=$1; shift; rdir=results/baseline_$tag; mkdir -p "$rdir" logs
    echo "===== $(date) :: START $tag :: $* =====" | tee -a logs/bench_baselines_main.log
    pids=()
    for k in $(seq 0 $((NW-1))); do
        $PY run_worker.py "$@" --num_workers $NW --caps $CAPS --worker_id $k \
            --out "$rdir/${MTH}_w$k.json" > "logs/base_${tag}_w$k.log" 2>&1 &
        pids+=($!); done
    for p in "${pids[@]}"; do wait $p; done
    $PY merge_results.py --method $MTH --num_workers $NW --results_dir "$rdir" \
        --out "$rdir/summary.json" | tee -a logs/bench_baselines_main.log
    echo "===== $(date) :: DONE $tag =====" | tee -a logs/bench_baselines_main.log
}
for ds in hybrid global both; do
    MTH=gpd; runp gpd_${ds}  --method gpd --guides 1 --batch_per 32 --pybullet_collision --dataset $ds
    MTH=gpd; runp repair_${ds} --method gpd --guides 1 --batch_per 32 --pybullet_collision --trajopt --trajopt_iters 60 --dataset $ds
    MTH=edmp; runp edmp_${ds} --method edmp --dataset $ds
done
echo "===== ALL BASELINES DONE $(date) =====" | tee -a logs/bench_baselines_main.log
