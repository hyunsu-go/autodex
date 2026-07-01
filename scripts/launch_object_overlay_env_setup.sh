#!/usr/bin/env bash
# Launch overlay environment setup on capture PCs.
#
# This opens at most one SSH session per PC. The setup itself keeps running on
# the capture PC and writes logs to the shared log root.
set -euo pipefail

PCS_DEFAULT=(capture1 capture2 capture3 capture5 capture6)
if [[ -n "${PCS:-}" ]]; then
    read -r -a PCS_LIST <<< "$PCS"
else
    PCS_LIST=("${PCS_DEFAULT[@]}")
fi

REMOTE_REPO="${REMOTE_REPO:-\$HOME/AutoDex}"
REMOTE_SHARED="${REMOTE_SHARED:-\$HOME/shared_data}"
LOCAL_SHARED="${LOCAL_SHARED:-$HOME/shared_data}"
BRANCH="${BRANCH:-tracking-session-progress}"
REMOTE_URL="${REMOTE_URL:-https://github.com/hyunsu-go/AutoDex.git}"
LOG_ID="${LOG_ID:-overlay_env_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT_REL="AutoDex/object_tracking/env_setup/$LOG_ID"
LOCAL_LOG_ROOT="$LOCAL_SHARED/$LOG_ROOT_REL"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
SETUP_ARGS="${SETUP_ARGS:-}"
ACTION="${1:-launch}"

usage() {
    cat <<EOF
usage: $0 {launch|status}

Environment overrides:
  PCS="capture1 capture2 ..."   Target PCs. Default: capture1 capture2 capture3 capture5 capture6
  BRANCH                        AutoDex branch to fetch. Default: tracking-session-progress
  REMOTE_URL                    Git URL to fetch from. Default: hyunsu-go fork
  LOG_ID                        Stable log id. Default: overlay_env_<timestamp>
  STAGGER_SECONDS               Delay between SSH launches. Default: 20
  SETUP_ARGS                    Extra args for setup script

Logs:
  $LOCAL_LOG_ROOT
EOF
}

status() {
    echo "[overlay-env] log root: $LOCAL_LOG_ROOT"
    mkdir -p "$LOCAL_LOG_ROOT"
    for pc in "${PCS_LIST[@]}"; do
        log="$LOCAL_LOG_ROOT/$pc.setup.log"
        launch_log="$LOCAL_LOG_ROOT/$pc.launch.log"
        if [[ -f "$log" ]]; then
            now=$(date +%s)
            mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
            age=$((now - mtime))
            if grep -q "\[overlay-env\] done" "$log"; then
                state=done
            elif grep -q "overlay_imports_ok" "$log"; then
                state=verified
            elif [[ "$age" -lt 180 ]]; then
                state=running
            elif grep -qiE "error|failed|traceback|moduleNotFound|no module named" "$log"; then
                state=failed
            else
                state=running
            fi
            last=$(tail -n 1 "$log" || true)
            echo "$pc  $state  age=${age}s  $last"
        elif [[ -f "$launch_log" ]]; then
            echo "$pc  launch-only  $(tail -n 1 "$launch_log" || true)"
        else
            echo "$pc  no-log"
        fi
    done
}

launch() {
    mkdir -p "$LOCAL_LOG_ROOT"
    echo "[overlay-env] launching setup on: ${PCS_LIST[*]}"
    echo "[overlay-env] local log root: $LOCAL_LOG_ROOT"
    local last_index=$((${#PCS_LIST[@]} - 1))
    for i in "${!PCS_LIST[@]}"; do
        pc="${PCS_LIST[$i]}"
        echo "[overlay-env] launching $pc"
        remote_cmd=$(cat <<EOF
set -eu
REPO=$REMOTE_REPO
LOG_ROOT=$REMOTE_SHARED/$LOG_ROOT_REL
mkdir -p "\$LOG_ROOT"
{
    echo "[launch] host=\$(hostname)"
    echo "[launch] repo=\$REPO"
    echo "[launch] log_root=\$LOG_ROOT"
    echo "[launch] setup_args=$SETUP_ARGS"
    cd "\$REPO"
    git fetch "$REMOTE_URL" "$BRANCH"
    git checkout -B "$BRANCH" FETCH_HEAD
    chmod +x scripts/setup_object_overlay_env.sh
    nohup bash scripts/setup_object_overlay_env.sh $SETUP_ARGS > "\$LOG_ROOT/$pc.setup.log" 2>&1 &
    echo \$! > "\$LOG_ROOT/$pc.setup.pid"
    echo "[launch] pid=\$(cat "\$LOG_ROOT/$pc.setup.pid")"
} > "\$LOG_ROOT/$pc.launch.log" 2>&1
EOF
)
        if ssh -o BatchMode=yes -o ConnectTimeout=8 "$pc" "$remote_cmd"; then
            echo "$pc launch submitted"
        else
            echo "$pc launch failed; see SSH output above" | tee "$LOCAL_LOG_ROOT/$pc.launch_failed"
        fi
        if [[ "$i" -ne "$last_index" ]]; then
            sleep "$STAGGER_SECONDS"
        fi
    done
    status
}

case "$ACTION" in
    launch)
        launch
        ;;
    status)
        status
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "unknown action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
