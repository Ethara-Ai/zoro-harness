#!/usr/bin/env bash
set -euo pipefail

_resolve_script_dir() {
  local src="${BASH_SOURCE[0]}"
  if command -v readlink >/dev/null 2>&1 && readlink -f "$src" >/dev/null 2>&1; then
    dirname "$(readlink -f "$src")"
  elif command -v perl >/dev/null 2>&1; then
    dirname "$(perl -MCwd -e 'print Cwd::abs_path(shift)' "$src")"
  else
    cd "$(dirname "$src")" && pwd
  fi
}

SCRIPT_DIR="$(_resolve_script_dir)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="${ZORO_CC_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

HOST="${ZORO_CC_BRIDGE_HOST:-127.0.0.1}"
PORT="${ZORO_CC_BRIDGE_PORT:-8738}"
PID_FILE="${REPO_ROOT}/.claude_bridge.pid"
MONITOR_PID_FILE="${REPO_ROOT}/.claude_bridge_monitor.pid"
LOG_FILE="${ZORO_CC_BRIDGE_LOG:-${REPO_ROOT}/logs/claude_bridge.log}"
MONITOR_LOG_FILE="${ZORO_CC_BRIDGE_MONITOR_LOG:-${REPO_ROOT}/logs/claude_bridge_monitor.log}"
MONITOR_POLL_SECONDS="${ZORO_CC_MONITOR_POLL:-30}"
MONITOR_FAIL_THRESHOLD="${ZORO_CC_MONITOR_FAILS:-3}"

_print_exports() {
  local api_key="${ZORO_CC_BRIDGE_SECRET:-zoro-cc-stub}"
  echo "export ZORO_LLM_BASE_URL=http://${HOST}:${PORT}/v1"
  echo "export ZORO_LLM_API_KEY=${api_key}"
}

_start_bridge_process() {
  mkdir -p "$(dirname "$LOG_FILE")"
  cd "$REPO_ROOT"
  nohup "$PY" -m claude_bridge --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
}

_wait_for_healthz() {
  local ok=0 i
  for i in $(seq 1 60); do
    if curl -fsS --max-time 5 "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 0.5
  done
  if [[ "$ok" != "1" ]]; then
    echo "[bridge] FAILED to come up; tail of $LOG_FILE:" >&2
    tail -20 "$LOG_FILE" >&2 || true
    return 1
  fi
  return 0
}

_start_monitor_process() {
  if [[ -f "$MONITOR_PID_FILE" ]] && kill -0 "$(cat "$MONITOR_PID_FILE")" 2>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$MONITOR_LOG_FILE")"
  nohup bash "${BASH_SOURCE[0]}" monitor \
    >>"$MONITOR_LOG_FILE" 2>&1 &
  echo $! >"$MONITOR_PID_FILE"
}

action="${1:-start}"

case "$action" in
  start)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[bridge] already running (PID $(cat "$PID_FILE"))" >&2
    else
      _start_bridge_process
    fi
    if ! _wait_for_healthz; then
      exit 1
    fi
    if [[ "${ZORO_CC_DISABLE_MONITOR:-0}" != "1" ]]; then
      _start_monitor_process
      echo "[bridge] monitor up (PID $(cat "$MONITOR_PID_FILE"))" >&2
    fi
    echo "[bridge] up on http://${HOST}:${PORT} (PID $(cat "$PID_FILE"))" >&2
    _print_exports
    ;;
  stop)
    if [[ -f "$MONITOR_PID_FILE" ]]; then
      mpid=$(cat "$MONITOR_PID_FILE")
      if kill -0 "$mpid" 2>/dev/null; then
        kill "$mpid" 2>/dev/null || true
        sleep 0.3
        kill -0 "$mpid" 2>/dev/null && kill -9 "$mpid" 2>/dev/null || true
        echo "[bridge] monitor stopped PID $mpid" >&2
      fi
      rm -f "$MONITOR_PID_FILE"
    fi
    if [[ -f "$PID_FILE" ]]; then
      pid=$(cat "$PID_FILE")
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 0.5
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        echo "[bridge] stopped PID $pid" >&2
      fi
      rm -f "$PID_FILE"
    else
      echo "[bridge] no PID file at $PID_FILE" >&2
    fi
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "[bridge] running (PID $(cat "$PID_FILE")) on http://${HOST}:${PORT}"
      curl -fsS --max-time 5 "http://${HOST}:${PORT}/healthz" || true
      echo
      if [[ -f "$MONITOR_PID_FILE" ]] && kill -0 "$(cat "$MONITOR_PID_FILE")" 2>/dev/null; then
        echo "[monitor] running (PID $(cat "$MONITOR_PID_FILE"))"
      else
        echo "[monitor] not running"
      fi
    else
      echo "[bridge] not running"
      exit 1
    fi
    ;;
  logs)
    exec tail -f "$LOG_FILE"
    ;;
  monitor)
    echo "[monitor] started PID=$$ poll=${MONITOR_POLL_SECONDS}s fail_threshold=${MONITOR_FAIL_THRESHOLD}"
    consecutive_failures=0
    while true; do
      if curl -fsS --max-time 5 "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1; then
        if [[ "$consecutive_failures" -gt 0 ]]; then
          echo "[monitor] $(date '+%Y-%m-%d %H:%M:%S') bridge recovered"
        fi
        consecutive_failures=0
      else
        consecutive_failures=$((consecutive_failures + 1))
        echo "[monitor] $(date '+%Y-%m-%d %H:%M:%S') healthz failed (${consecutive_failures}/${MONITOR_FAIL_THRESHOLD})"
        if [[ "$consecutive_failures" -ge "$MONITOR_FAIL_THRESHOLD" ]]; then
          echo "[monitor] $(date '+%Y-%m-%d %H:%M:%S') restarting bridge"
          if [[ -f "$PID_FILE" ]]; then
            kill "$(cat "$PID_FILE")" 2>/dev/null || true
            sleep 0.5
            kill -9 "$(cat "$PID_FILE")" 2>/dev/null || true
            rm -f "$PID_FILE"
          fi
          _start_bridge_process
          consecutive_failures=0
          sleep 2
        fi
      fi
      sleep "$MONITOR_POLL_SECONDS"
    done
    ;;
  *)
    echo "usage: $0 {start|stop|status|logs|monitor}" >&2
    exit 2
    ;;
esac
