#!/usr/bin/env bash
# Stage 0 of the 3x2 pilot (research/pilot_3x2_liberoplus.md, section 8): arms N and R,
# seeds 41-42, conditions ID / scene blackout / robot_init 0.2, plus the released-
# checkpoint reference and the pi05_base zero-shot exposure probe.
#
# Prerequisites (openpi README + examples/libero/README.md):
#   - openpi checkout at $OPENPI_ROOT with `uv sync` done (server/training env, >70 GB GPU)
#   - LIBERO client venv at $OPENPI_ROOT/examples/libero/.venv (Python 3.8)
#   - MUJOCO_GL=egl (or glx) works on this machine
#
# Usage:  OPENPI_ROOT=~/openpi bash stage0.sh [step ...]
#   steps: norm_stats train eval reference probe analyze   (default: all, in order)
set -euo pipefail

: "${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}"
PILOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$PILOT/../results/stage0}"
SEEDS="${SEEDS:-41 42}"
ARMS="${ARMS:-N R}"
CONDITIONS="${CONDITIONS:-id,scene_blackout,robot_init:0.2}"
SUITES="libero_spatial libero_object libero_goal libero_10"
PORT="${PORT:-8000}"
WANDB_FLAG="${WANDB_FLAG:---no-wandb-enabled}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUT/logs"
cd "$OPENPI_ROOT"

srv() { # srv <config> <checkpoint dir> [extra serve.py args...]
  uv run python "$PILOT/serve.py" --config "$1" --dir "$2" --port "$PORT" "${@:3}" \
    >"$OUT/logs/serve_$(basename "$1")_$(echo "$2" | tr '/:' '__').log" 2>&1 &
  SERVER_PID=$!
}

client() { # client <arm> <seed label> <conditions> [init range]; one process per suite, in parallel
  local pids=()
  for s in $SUITES; do
    examples/libero/.venv/bin/python "$PILOT/eval_libero.py" --arm "$1" --seed-label "$2" \
      --port "$PORT" --suites "$s" --conditions "$3" --init-states "${4:-0-9}" \
      --out "$OUT/${1}_s${2}.jsonl.$s" >"$OUT/logs/client_${1}_s${2}_$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  cat "$OUT/${1}_s${2}.jsonl."* >"$OUT/${1}_s${2}.jsonl"
}

step_norm_stats() {
  for arm in $ARMS; do uv run python "$PILOT/run_openpi.py" norm_stats --config-name "pilot_$arm"; done
  # N and R read the same data, so their state/action stats must be identical.
  cmp "assets/pilot_N/physical-intelligence/libero/norm_stats.json" \
      "assets/pilot_R/physical-intelligence/libero/norm_stats.json" && echo "norm stats identical across arms"
}

step_train() {
  uv run python "$PILOT/run_openpi.py" describe pilot_N pilot_R | tee "$OUT/configs.jsonl"
  for arm in $ARMS; do for seed in $SEEDS; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python "$PILOT/run_openpi.py" train "pilot_$arm" \
      --exp-name "s$seed" --seed "$seed" --overwrite $WANDB_FLAG 2>&1 | tee "$OUT/logs/train_${arm}_s$seed.log"
  done; done
}

step_eval() {
  export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"
  for arm in $ARMS; do for seed in $SEEDS; do
    # openpi saves the final step as num_train_steps - 1; params are the EMA weights.
    srv "pilot_$arm" "checkpoints/pilot_$arm/s$seed/9999"
    client "$arm" "$seed" "$CONDITIONS"
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  done; done
}

step_reference() { # released pi05_libero = stock config, no state (reference for arm N)
  export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"
  srv pi05_libero gs://openpi-assets/checkpoints/pi05_libero
  client released released "$CONDITIONS"
  kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
}

step_probe() { # pi05_base zero-shot on LIBERO: 10 tasks x 5 init states per suite, ID only
  export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_ROOT/third_party/libero"
  srv pilot_base_probe gs://openpi-assets/checkpoints/pi05_base \
      --norm-stats-from gs://openpi-assets/checkpoints/pi05_libero
  client base_probe base id 0-4
  kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
}

step_analyze() {
  uv run python "$PILOT/analyze_stage0.py" "$OUT"/*.jsonl \
    --norm-stats "assets/pilot_R/physical-intelligence/libero/norm_stats.json" \
    --out "$OUT/summary.json" | tail -n 60
}

for step in "${@:-norm_stats train eval reference probe analyze}"; do
  for s in $step; do "step_$s"; done
done
