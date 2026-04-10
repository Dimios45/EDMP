#!/bin/bash
# Sequential GPD → EDMP full benchmark
# Logs everything to benchmark_run.log

cd /home/sra/EDMP
source .venv/bin/activate

echo "========================================" | tee -a benchmark_run.log
echo "START: $(date)" | tee -a benchmark_run.log
echo "========================================" | tee -a benchmark_run.log

echo "" | tee -a benchmark_run.log
echo ">>> PHASE 1: GPD (started $(date))" | tee -a benchmark_run.log
echo "" | tee -a benchmark_run.log

python compare.py --method gpd --full --out compare_results.json 2>&1 | tee -a benchmark_run.log

echo "" | tee -a benchmark_run.log
echo ">>> GPD COMPLETE: $(date)" | tee -a benchmark_run.log
echo "" | tee -a benchmark_run.log
echo ">>> PHASE 2: EDMP (started $(date))" | tee -a benchmark_run.log
echo "" | tee -a benchmark_run.log

python compare.py --method edmp --full --out compare_results.json 2>&1 | tee -a benchmark_run.log

echo "" | tee -a benchmark_run.log
echo "========================================" | tee -a benchmark_run.log
echo "ALL DONE: $(date)" | tee -a benchmark_run.log
echo "========================================" | tee -a benchmark_run.log
