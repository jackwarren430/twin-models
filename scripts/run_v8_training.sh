#!/usr/bin/env bash
# Launch the v8 solver run under a memory-capped transient unit.
#
# Both Spark guards apply (see the unified-memory postmortem):
#   1. compute.grad_checkpointing: true — set in the config. This is the guard
#      that actually works; it is what kept peak flat at ~46GB across a 30-iter
#      run. Unified memory means an overflowing GPU alloc exhausts SYSTEM RAM
#      and trips the GLOBAL kernel OOM killer, taking the desktop with it.
#   2. A cgroup cap so a runaway kills only the run. NOTE: on this build the
#      cgroup counter has been observed NOT to reflect CUDA/unified allocations
#      (MemoryCurrent read ~12GB against a true ~46GB footprint), so the cap may
#      never fire. Do not rely on it — the real safety metrics are system-wide
#      `free -g` and the per-iteration peak_mem_gb in the run JSONL.
#
# Preconditions are checked here rather than discovered an hour in: the bank
# must exist and be big enough to be worth training on, and no other GPU job may
# be running (two resident models on a shared pool is how the box goes down).
set -u
REPO=/home/jackwarren430/Documents/repos/twin-models
cd "$REPO"

CONFIG=configs/twentyq-v8-bank-solver.yaml
UNIT=${UNIT:-q-v8-bank}
RUN_NAME=${RUN_NAME:-q-v8-bank-solver}
LOG="tq-runs/${RUN_NAME}.console.log"
MIN_BANK=${MIN_BANK:-12}

BANK=$(.venv/bin/python - "$CONFIG" <<'PY'
import sys; sys.path.insert(0, "src")
from twin.config import Config
print(Config.from_yaml(sys.argv[1]).twentyq.bank_path)
PY
)

if [ ! -f "$BANK" ]; then
  echo "REFUSING: bank $BANK does not exist — run scripts/run_bank_build.sh first" >&2
  exit 1
fi

N=$(.venv/bin/python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['secrets']))" "$BANK")
if [ "$N" -lt "$MIN_BANK" ]; then
  echo "REFUSING: bank $BANK has only $N secrets (min $MIN_BANK)." >&2
  echo "  A bank this thin invites memorizing entities rather than learning to" >&2
  echo "  probe, which shows up as training win rate rising while validation" >&2
  echo "  stays flat. Generate more candidates before spending a night on it." >&2
  echo "  Override with MIN_BANK=<n> if that is genuinely what you want." >&2
  exit 1
fi

for other in q-probe-batch2 q-bank-build; do
  if systemctl --user is-active --quiet "$other"; then
    echo "REFUSING: $other is still running — two resident models on a shared" >&2
    echo "  memory pool is how this box goes down. Wait for it to finish." >&2
    exit 1
  fi
done

if systemctl --user is-active --quiet "$UNIT"; then
  echo "REFUSING: unit $UNIT is already active." >&2
  exit 1
fi

echo "bank: $BANK ($N secrets)"
echo "unit: $UNIT   log: $LOG"

exec systemd-run --user --unit="$UNIT" \
  -p MemoryMax=100G -p MemorySwapMax=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="$REPO" \
  bash -c ".venv/bin/python -u scripts/train_twentyq.py \
             --config $CONFIG \
             --run-name $RUN_NAME \
             --checkpoints-dir checkpoints/twentyq-v8-bank \
             $* > $LOG 2>&1"
