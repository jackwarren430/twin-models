#!/usr/bin/env bash
# Build the v8 frontier bank, smoke pass first.
#
# The real build spends hours in stage 3 (calibrate), which is the LAST stage
# and therefore the one a crash is most expensive to discover. The smoke pass
# runs the identical code path end to end at toy scale — generate, dedup, vet
# with the real judge, calibrate, select, write both files — in a few minutes.
# If it exits non-zero the real build never starts.
#
# The smoke pass writes to its own --report and --output so it can never be
# mistaken for the bank, and so its checkpoint cannot be resumed into the real
# one (the signature gate would reject it anyway: different rollout count and
# episodes-per-candidate).
#
# Usage:
#   BANK_MODEL=<hf-id-or-path> scripts/run_bank_build.sh [EXTRA_ARGS...]
#
#   BANK_MODEL overrides the config's player model — the bake-off winner — and
#   applies to BOTH passes, since a smoke pass on a different model would not
#   be exercising the path the real build takes (chat template, stop tokens and
#   thinking suppression all differ per model). Omit to use whatever the config
#   points at.
#
#   BANK_ADAPTER is a solver LoRA checkpoint to calibrate against (bank-v2 is
#   measured against the trained policy, not the base model). It also applies
#   to BOTH passes, for the same reason: adapter loading and per-turn adapter
#   switching is a CODE PATH, not a scale knob, and it is the newest and least
#   exercised part of the build. Smoke-testing the base path and then running
#   hours of the adapter path would defeat the gate precisely where it matters.
#
#   EXTRA_ARGS go to the real build only, e.g. --rollouts-per-category 250. The
#   smoke pass keeps fixed toy args so it stays fast whatever the real build is
#   asked to do — scale knobs only, never a change of code path.
set -u
cd /home/jackwarren430/Documents/repos/twin-models

CONFIG=configs/twentyq-v8-bank-solver.yaml
PY=.venv/bin/python

MODEL_ARGS=()
if [ -n "${BANK_MODEL:-}" ]; then
  MODEL_ARGS=(--model "$BANK_MODEL")
  echo "player model override: $BANK_MODEL" >&2
fi
if [ -n "${BANK_ADAPTER:-}" ]; then
  [ -f "$BANK_ADAPTER" ] || { echo "BANK_ADAPTER not found: $BANK_ADAPTER" >&2; exit 1; }
  MODEL_ARGS+=(--adapter "$BANK_ADAPTER" --adapter-name "${BANK_ADAPTER_NAME:-B}")
  echo "calibrating against adapter: $BANK_ADAPTER" >&2
fi

echo "=== smoke pass (toy scale, real judge, full pipeline) ===" >&2
if ! $PY -u scripts/build_twentyq_bank.py \
      --config "$CONFIG" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" \
      --rollouts-per-category 4 \
      --targets 0.95 0.9 \
      --max-candidates 3 \
      --episodes-per-candidate 2 \
      --band 0.0 1.0 \
      --no-resume \
      --report tq-runs/bank-smoke-calibration.json \
      --output data/twentyq-bank-smoke.json; then
  echo "SMOKE FAILED — real build not started" >&2
  exit 1
fi
echo "=== smoke pass OK ===" >&2

echo "=== real build ===" >&2
# --resume is on by default: if this is a restart after a kill, the signature
# gate reuses the checkpoint and measures only what is missing.
$PY -u scripts/build_twentyq_bank.py \
    --config "$CONFIG" "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" "$@" \
  || { echo "BUILD FAILED" >&2; exit 1; }

echo "BANK BUILD COMPLETE" >&2
