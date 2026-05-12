#!/usr/bin/env bash
# Manage gotrack_daemon across capture1-6.
#
# Usage:
#     bash scripts/gotrack_daemons.sh start
#     bash scripts/gotrack_daemons.sh stop
#     bash scripts/gotrack_daemons.sh status
#     bash scripts/gotrack_daemons.sh log capture1   # tail one PC's log
set -euo pipefail

PCS=(capture1 capture2 capture3 capture5 capture6)  # capture4 out
# Resolved on the REMOTE side via $HOME — do NOT expand ~ locally.
PY='$HOME/anaconda3/envs/gotrack_cu128/bin/python'
DAEMON='$HOME/AutoDex/src/execution/daemon/gotrack_daemon.py'
LOG=/tmp/gotrack_daemon.log
ROBOT_IP="${ROBOT_IP:-192.168.0.2}"

ACTION="${1:-status}"

case "$ACTION" in
    start)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f gotrack_daemon 2>/dev/null || true" &
        done
        wait
        sleep 2
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "bash -c 'nohup $PY $DAEMON --robot-ip $ROBOT_IP > $LOG 2>&1 &'"
        done
        sleep 3
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*gotrack_daemon'" 2>/dev/null || echo 0)
            echo "  $pc: $n daemon(s)"
        done
        ;;
    stop)
        for pc in "${PCS[@]}"; do
            ssh -o ConnectTimeout=3 "$pc" "pkill -9 -f gotrack_daemon 2>/dev/null && echo killed || true" &
        done
        wait
        ;;
    status)
        for pc in "${PCS[@]}"; do
            n=$(ssh -o ConnectTimeout=3 "$pc" "pgrep -fc 'python.*gotrack_daemon'" 2>/dev/null || echo "?")
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
