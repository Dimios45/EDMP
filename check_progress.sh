#!/bin/bash
# Quick progress check for parallel benchmark workers
cd /home/sra/EDMP

echo "========================================"
echo "Benchmark Progress - $(date)"
echo "========================================"

for METHOD in gpd edmp; do
    LOG="logs/${METHOD}_w0.log"
    if [[ ! -f "$LOG" ]]; then
        echo "[$METHOD] log not found"
        continue
    fi

    SCENES=$(grep -c "success=" "$LOG" 2>/dev/null || echo 0)
    LAST=$(grep "success=" "$LOG" 2>/dev/null | tail -1)
    RUNNING=$(pgrep -f "run_worker.py.*method $METHOD" | head -1)

    echo ""
    echo "[$METHOD] scenes=$SCENES | running=$([ -n "$RUNNING" ] && echo YES pid=$RUNNING || echo NO)"
    [ -n "$LAST" ] && echo "  Last: $LAST"

    # Per-scene-type breakdown
    for ST in tabletop cubby merged_cubby dresser; do
        COUNT=$(grep "\[$ST:" "$LOG" 2>/dev/null | wc -l)
        SR=$(grep "\[$ST:" "$LOG" 2>/dev/null | grep "success=1" | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  $ST: $SR/$COUNT"
        fi
    done

    # Average time from last 10 scenes
    AVG=$(grep "t=[0-9]" "$LOG" 2>/dev/null | tail -10 | \
          grep -oP 't=\K[0-9.]+' | \
          awk '{s+=$1;c++} END {if(c>0) printf "%.2f", s/c; else print "?"}')
    [ -n "$AVG" ] && echo "  avg t= ${AVG}s (last 10 scenes)"
done

echo ""
echo "========================================"
