cd "$(dirname "$(readlink -f "$0")")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; LOG=logs/audit_multiseed_main.log; mkdir -p logs results
echo "===== AUDIT MULTISEED START $(date) =====" | tee -a "$LOG"
for s in 0 1 2 3 4; do
  $PY continuous_audit.py --method gpd --dataset hybrid --per_type 25 --seed $s 2>&1 | tee -a "$LOG"
done
for ds in global both; do for s in 0 1 2; do
  $PY continuous_audit.py --method gpd --dataset $ds --per_type 25 --seed $s 2>&1 | tee -a "$LOG"
done; done
for s in 0 1 2; do
  $PY continuous_audit.py --method edmp --dataset hybrid --per_type 10 --seed $s 2>&1 | tee -a "$LOG"
done
echo "===== AUDIT MULTISEED DONE $(date) =====" | tee -a "$LOG"
