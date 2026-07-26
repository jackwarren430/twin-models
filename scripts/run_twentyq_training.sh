#!/usr/bin/env bash
# Launch a twentyq bank-solver run under a memory-capped transient unit.
#
# Generalised from run_v8_training.sh, which this replaces. v8 was launched as:
#
#   scripts/run_twentyq_training.sh configs/twentyq-v8-bank-solver.yaml \
#     UNIT=q-v8-bank RUN_NAME=q-v8-bank-solver CKPT_DIR=checkpoints/twentyq-v8-bank
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
# must exist and be big enough to be worth training on, it must be disjoint
# from every set the run will ever be scored against, and no other GPU job may
# be running (two resident models on a shared pool is how the box goes down).
#
# Usage:
#   scripts/run_twentyq_training.sh CONFIG [VAR=VAL ...] [-- EXTRA_TRAIN_ARGS]
#
#   UNIT            systemd unit name          (default: derived from RUN_NAME)
#   RUN_NAME        log/checkpoint stem        (default: config basename)
#   CKPT_DIR        checkpoint directory       (default: checkpoints/<RUN_NAME>)
#   EXTRA_HOLDOUT   additional secret sets the bank must not overlap, space
#                   separated. REQUIRED whenever the run's checkpoints will be
#                   replayed against a set other than the live one: a set is
#                   contaminated by training on it whether the scoring happens
#                   during the run or after it. v9 must pass validation-v2 here.
#   MIN_BANK        minimum bank size          (default: 12)
set -u
REPO=/home/jackwarren430/Documents/repos/twin-models
cd "$REPO"

CONFIG=${1:-}
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
  echo "usage: $0 CONFIG [VAR=VAL ...] [-- EXTRA_TRAIN_ARGS]" >&2
  echo "  config not found: '${CONFIG}'" >&2
  exit 1
fi
shift

# VAR=VAL pairs before an optional `--`; everything after goes to the trainer.
TRAIN_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; TRAIN_ARGS=("$@"); break ;;
    *=*) export "${1?}" ;;
    *) echo "unexpected argument '$1' (use -- before trainer args)" >&2; exit 1 ;;
  esac
  shift
done

RUN_NAME=${RUN_NAME:-$(basename "$CONFIG" .yaml)}
UNIT=${UNIT:-$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-checkpoints/$RUN_NAME}
LOG="tq-runs/${RUN_NAME}.console.log"
MIN_BANK=${MIN_BANK:-12}
EXTRA_HOLDOUT=${EXTRA_HOLDOUT:-}

read -r BANK HOLDOUT <<EOF
$(.venv/bin/python - "$CONFIG" <<'PY'
import sys; sys.path.insert(0, "src")
from twin.config import Config
q = Config.from_yaml(sys.argv[1]).twentyq
print(q.bank_path, q.validation_secret_set)
PY
)
EOF

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

# shellcheck disable=SC2086
if ! .venv/bin/python scripts/check_bank_disjoint.py "$BANK" \
       --holdout "$HOLDOUT" $EXTRA_HOLDOUT; then
  echo "REFUSING: bank overlaps a set this run will be scored against — the" >&2
  echo "  result would be uninterpretable. Rebuild the bank." >&2
  exit 1
fi

# Any other GPU job, not a hardcoded list: the failure mode is two resident
# models on one shared pool, and it does not care what the other unit is named.
for other in $(systemctl --user list-units --type=service --state=running \
                 --no-legend --plain | awk '{print $1}' | grep '^q-' || true); do
  if [ "$other" != "${UNIT}.service" ]; then
    echo "REFUSING: $other is still running — two resident models on a shared" >&2
    echo "  memory pool is how this box goes down. Wait for it to finish." >&2
    exit 1
  fi
done

if systemctl --user is-active --quiet "$UNIT"; then
  echo "REFUSING: unit $UNIT is already active." >&2
  exit 1
fi

echo "config: $CONFIG"
echo "bank:   $BANK ($N secrets)"
echo "scored against: $HOLDOUT ${EXTRA_HOLDOUT:+(+ replay: $EXTRA_HOLDOUT)}"
echo "unit:   $UNIT   ckpt: $CKPT_DIR   log: $LOG"

exec systemd-run --user --unit="$UNIT" \
  -p MemoryMax=100G -p MemorySwapMax=0 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --setenv=PYTHONUNBUFFERED=1 \
  --working-directory="$REPO" \
  bash -c ".venv/bin/python -u scripts/train_twentyq.py \
             --config $CONFIG \
             --run-name $RUN_NAME \
             --checkpoints-dir $CKPT_DIR \
             ${TRAIN_ARGS[*]+${TRAIN_ARGS[*]}} > $LOG 2>&1"
