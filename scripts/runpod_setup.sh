#!/usr/bin/env bash
# Workstream-A Session 4 — RunPod 4090 bringup for cable_ppo restart training.
#
# Run this on the pod after SSH-ing in. Assumes a CUDA-enabled image (any of
# the standard PyTorch templates work). Will:
#   1. Install MuJoCo's GL deps (commonly missing on minimal images)
#   2. Clone the repo (override REPO_URL env var if you fork it)
#   3. Build a fresh venv with project requirements + wandb
#   4. Smoke-test env construction + 5K-step training to catch breakage early
#   5. Print the kickoff command (you run it manually inside tmux)
#
# Required env var:
#   WANDB_API_KEY  (set in RunPod dashboard or `export WANDB_API_KEY=...` before)
#
# Optional env vars:
#   REPO_URL       (default: github SSH; flip to https if no SSH key on pod)
#   REPO_BRANCH    (default: main)
#   WORKDIR        (default: /workspace/cable-insertion)
#   WANDB_RUN_NAME (default: train.py picks restart_session4_<utc_iso8601>)
#
# Usage on pod:
#   curl -sSL <url-or-scp-this-file> > setup.sh && bash setup.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/poisson-labs/cable--insertion.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace/cable-insertion}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Set it via RunPod dashboard env vars" >&2
  echo "       or 'export WANDB_API_KEY=...' before running this script." >&2
  exit 1
fi

echo "[setup] Stage 1/5 — installing MuJoCo GL deps"
apt-get update -qq
apt-get install -y -qq \
  libgl1-mesa-glx \
  libegl1 \
  libglu1-mesa \
  libglib2.0-0 \
  git \
  rsync \
  tmux \
  >/dev/null

echo "[setup] Stage 2/5 — cloning repo to ${WORKDIR}"
mkdir -p "$(dirname "${WORKDIR}")"
if [[ -d "${WORKDIR}/.git" ]]; then
  echo "[setup] repo already present, fetching latest"
  git -C "${WORKDIR}" fetch --quiet origin "${REPO_BRANCH}"
  git -C "${WORKDIR}" checkout --quiet "${REPO_BRANCH}"
  git -C "${WORKDIR}" reset --hard --quiet "origin/${REPO_BRANCH}"
else
  git clone --quiet --branch "${REPO_BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"

echo "[setup] Stage 3/5 — building venv"
python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements.txt

echo "[setup] Stage 4a/5 — verifying core imports"
python -c "
import mujoco, gymnasium, stable_baselines3, wandb
import torch
print(f'  mujoco           : {mujoco.__version__}')
print(f'  gymnasium        : {gymnasium.__version__}')
print(f'  stable_baselines3: {stable_baselines3.__version__}')
print(f'  wandb            : {wandb.__version__}')
print(f'  torch            : {torch.__version__}')
print(f'  CUDA available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device      : {torch.cuda.get_device_name(0)}')
"

echo "[setup] Stage 4b/5 — env smoke test"
python -c "
import sys; sys.path.insert(0, '.')
from envs.cable_env import CableInsertionEnv
env = CableInsertionEnv(randomize=True)
obs, _ = env.reset(seed=0)
print(f'  obs shape       : {obs.shape}')
print(f'  action_space    : {env.action_space}')
for _ in range(10):
    obs, r, done, trunc, info = env.step(env.action_space.sample())
print(f'  step ok, dist   : {info[\"distance\"]:.3f}m')
env.close()
"

echo "[setup] Stage 4c/5 — 5K-timestep training smoke (no wandb)"
WANDB_MODE=disabled python -c "
import sys; sys.path.insert(0, '.')
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.cable_env import CableInsertionEnv
env = Monitor(CableInsertionEnv(randomize=True))
model = PPO('MlpPolicy', env, verbose=0, learning_rate=1e-4, ent_coef=0.01,
            n_steps=2048, batch_size=64, n_epochs=10, seed=42)
t0 = time.time()
model.learn(total_timesteps=5000)
dt = time.time() - t0
print(f'  5K steps in {dt:.1f}s ({5000/dt:.0f} fps) — full run ETA {(4_000_000/(5000/dt))/3600:.1f}h')
"

echo "[setup] Stage 5/5 — DONE. To kick off real training:"
cat <<'EOF'

  # 1. Open a tmux session that survives SSH disconnect:
  tmux new -s cable_train

  # 2. Inside tmux, activate venv and start training:
  cd ${WORKDIR:-/workspace/cable-insertion}
  source venv/bin/activate
  export WANDB_API_KEY=...    # if not already exported via dashboard
  mkdir -p logs models
  python -m training.train 2>&1 | tee logs/train_session4.log

  # 3. Detach with Ctrl-b d. Reattach with: tmux attach -t cable_train
  # 4. wandb URL is printed at the top of the run.
EOF
