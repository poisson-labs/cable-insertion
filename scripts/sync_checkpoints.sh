#!/usr/bin/env bash
# Pull RunPod-side checkpoints to local on a fixed cadence.
# wandb handles live metrics; this script only mirrors the .zip files so
# the local copy of the repo can run validate_checkpoint.sh against them
# without an extra rsync each time.
#
# Usage:
#   ./scripts/sync_checkpoints.sh <pod_ssh_user@pod_host> [pod_workdir] [interval_sec]
#
# Defaults:
#   pod_workdir  : /workspace/cable-insertion
#   interval_sec : 600  (10 min)
#
# Local destination: cable-insertion/models_runpod/
# Stops when killed; safe to ctrl-c.

set -euo pipefail

POD_HOST="${1:?usage: sync_checkpoints.sh <user@host> [pod_workdir] [interval_sec]}"
POD_WORKDIR="${2:-/workspace/cable-insertion}"
INTERVAL="${3:-600}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DEST="${REPO_ROOT}/models_runpod"
mkdir -p "${LOCAL_DEST}"

echo "[sync] pod    : ${POD_HOST}:${POD_WORKDIR}/models/"
echo "[sync] local  : ${LOCAL_DEST}/"
echo "[sync] period : ${INTERVAL}s (Ctrl-c to stop)"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if rsync -av --partial --info=progress2 \
      "${POD_HOST}:${POD_WORKDIR}/models/" "${LOCAL_DEST}/" 2>&1 | tail -3; then
    echo "[sync] ${ts} OK"
  else
    echo "[sync] ${ts} FAILED (will retry)" >&2
  fi
  sleep "${INTERVAL}"
done
