#!/usr/bin/env bash
# Usage: scripts/notify.sh "<handoff phrase>"
# Desktop notification (if available) plus a terminal banner, so the operator sees the phrase.
set -u
msg="${1:?usage: notify.sh \"<phrase>\"}"
if command -v notify-send >/dev/null 2>&1; then
    notify-send "xdna-relay" "$msg" 2>/dev/null || true
fi
line=$(printf '=%.0s' $(seq 1 72))
printf '\n%s\n  xdna-relay handoff:\n\n  %s\n%s\n\n' "$line" "$msg" "$line"
