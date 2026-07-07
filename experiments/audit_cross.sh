#!/bin/bash
# Workstream A: continuous-feasibility audit across datasets (GPD) + cross-planner (EDMP).
# Single-process (audit is not sharded); runs sequentially. Outputs
# results/continuous_audit_<method>_<dataset>.json.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/bin/python
LOG=logs/audit_cross.log; mkdir -p logs results
echo "===== AUDIT CROSS START $(date) =====" | tee -a "$LOG"
for ds in global hybrid both; do
    echo "----- gpd/$ds $(date) -----" | tee -a "$LOG"
    $PY continuous_audit.py --method gpd --dataset $ds --per_type 25 2>&1 | tee -a "$LOG"
done
echo "----- edmp/hybrid $(date) -----" | tee -a "$LOG"
$PY continuous_audit.py --method edmp --dataset hybrid --per_type 10 2>&1 | tee -a "$LOG"
echo "===== AUDIT CROSS DONE $(date) =====" | tee -a "$LOG"
