#!/usr/bin/env bash
# Build validation-v3: a band-selected evaluation set with real dynamic range.
#
# Why. Through v8, validation was data/twentyq-validation-v2.json, and 15 of
# its 24 secrets are 0/8 for the trained adapter at EVERY checkpoint while cow
# sits at 8/8 — the metric's whole dynamic range is about seven secrets. A
# secret no checkpoint ever wins carries no information about whether the
# policy improved, exactly as an all-loss GRPO group carries no gradient.
# Raising K fixed the sample size and could not fix this, because the limit is
# WHICH SECRETS ARE IN THE SET.
#
# So build the instrument the same way the training bank was built: generate,
# vet, PLAY, and keep only what the base policy sometimes wins and sometimes
# loses. Selection uses the BASE model alone and never consults a trained
# adapter, which is what keeps this from being test-set contamination.
#
# This set is a SECONDARY instrument. v8's pre-registered result stands or
# falls on validation-v2; v3 exists to interpret that result, not to replace it
# after the fact.
#
# Two deliberate differences from the bank build:
#   * K=16 rather than 8. Band membership is decided on a noisy measurement,
#     so entries near an edge are partly lucky and regress when replayed. For
#     a training bank that is tolerable; for a measuring instrument, selection
#     error is the dominant risk, and K is the one knob that reduces it.
#   * 400 rollouts per category rather than 250. 85 names are already spoken
#     for by the bank and validation-v2, and this creator only produced 118
#     distinct secrets in 750 draws, so diversity — not calibration yield — is
#     what bounds the set size.
#
# Everything else is held identical to the bank build on purpose: same model,
# same categories, same band, same turn budget. Exactly one thing is changing
# about the instrument, and it is composition.
set -u
cd /home/jackwarren430/Documents/repos/twin-models

PY=.venv/bin/python
OUT=data/twentyq-validation-v3.json
REPORT=tq-runs/validation-v3-calibration.json

# Unified memory: a second model loaded next to a live run is not merely slow,
# it can OOM the box and kill the run. The build itself is resumable; the run
# it would take down is not.
BUSY=$(systemctl --user list-units --type=service --state=running --no-legend --plain \
       | awk '{print $1}' | grep '^q-.*\.service$' || true)
if [ -n "$BUSY" ]; then
  echo "REFUSING: $BUSY is running. Wait for it to finish." >&2
  exit 1
fi

for f in data/twentyq-validation-v2.json data/twentyq-bank-v1.json; do
  [ -f "$f" ] || { echo "missing holdout: $f" >&2; exit 1; }
done

# Selection MUST use the base policy alone. BANK_ADAPTER exported for a bank-v2
# build and still set in the environment here would silently select validation
# items against the very adapter under test — test-set contamination that no
# downstream check could detect, because the resulting file looks perfectly
# well-formed. Unset rather than trusted to be absent.
unset BANK_ADAPTER BANK_ADAPTER_NAME

scripts/run_bank_build.sh \
    --rollouts-per-category 400 \
    --episodes-per-candidate 16 \
    --band 0.125 0.875 \
    --holdout data/twentyq-validation-v2.json data/twentyq-bank-v1.json \
    --set-version 3 \
    --output "$OUT" \
    --report "$REPORT" \
  || { echo "BUILD FAILED" >&2; exit 1; }

# The build already holds these out; this checks the artifact rather than the
# intent. Contamination here would make "the solver improved" and "the solver
# memorised the test" indistinguishable after the fact, and nothing downstream
# would flag it.
$PY scripts/check_bank_disjoint.py "$OUT" \
    --holdout data/twentyq-validation-v2.json data/twentyq-bank-v1.json \
  || { echo "DISJOINTNESS CHECK FAILED — do not use $OUT" >&2; exit 1; }

# Size gate. Derived from v8's measured paired per-secret sd of 0.187 (see
# scripts/validation_power.py): at 80% power and p<0.05 a paired comparison
# needs ~27 all-live secrets to resolve +0.10 and ~56 to resolve +0.07. Below
# ~30 this set would reproduce v8's failure in a new costume — a null that
# cannot distinguish "no transfer" from "could not have seen it" — after
# spending hours to build it and hours more to score against it.
MIN_V3=${MIN_V3:-30}
$PY - "$OUT" "$MIN_V3" <<'EOF'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
s = d["secrets"]
print(f"\n{d['name']}: {len(s)} secrets")
for c, n in sorted(Counter(x["category"] for x in s).items()):
    print(f"  {c:<20} {n}")
rates = [x["measured_guess_rate"] for x in s]
print(f"  measured win rate: min {min(rates):.3f} "
      f"mean {sum(rates) / len(rates):.3f} max {max(rates):.3f}")
sd = 0.187
z = 1.96 + 0.8416
print(f"  minimum detectable effect (paired, 80% power, sd={sd}): "
      f"{z * sd / len(s) ** 0.5:+.4f}")
need = int(sys.argv[2])
if len(s) < need:
    print(f"\nTOO SMALL: {len(s)} secrets < {need}. This set could not resolve "
          f"the effect\nsizes in play; raise --rollouts-per-category and "
          f"rebuild, or widen --band.", file=sys.stderr)
    raise SystemExit(1)
EOF
[ $? -eq 0 ] || { echo "V3 REJECTED — do not use $OUT" >&2; exit 1; }

echo "VALIDATION-V3 BUILD COMPLETE" >&2
