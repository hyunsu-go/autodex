#!/usr/bin/env bash
# Manage snapshot_daemon across capture1-3, 5, 6 (capture4 out).
# Mirrors init_daemons.sh — same env (gotrack_cu128), same restart semantics.
#
# Usage:
#     bash scripts/snapshot_daemons.sh start
#     bash scripts/snapshot_daemons.sh stop
#     bash scripts/snapshot_daemons.sh status
#     bash scripts/snapshot_daemons.sh log capture1
set -euo pipefail

PCS=(capture1 capture2 capture3 capture5 capture6)
PY='$HOME/anaconda3/envs/gotrack_cu128/bin/python'
DAEMON='$HOME/AutoDex/src/execution/daemon/snapshot_daemon.py'
LOG=/tmp/snapshot_daemon.log

ACTION="${1:-status}"

case "$ACTION" in
    start)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f snapshot_daemon 2>/dev/null || true" &
        done
        wait
        sleep 2
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "bash -c 'nohup $PY $DAEMON > $LOG 2>&1 &'"
        done
        sleep 3
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*snapshot_daemon'" 2>/dev/null || echo 0)
            echo "  $pc: $n daemon(s)"
        done
        ;;
    stop)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f snapshot_daemon 2>/dev/null && echo killed || true" &
        done
        wait
        ;;
    status)
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*snapshot_daemon'" 2>/dev/null || echo "?")
            echo "  $pc: $n"
        done
        ;;
    log)
        pc="${2:-capture1}"
        ssh "$pc" "tail -50 $LOG"
        ;;
    *)
        echo "usage: $0 {start|stop|status|log [pc_name]}"
        exit 1
        ;;
esac
