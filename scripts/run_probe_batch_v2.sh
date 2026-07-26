#!/usr/bin/env bash
# GPU batch 2, run sequentially so only one model is resident at a time.
#
#   1. gemma-4-E2B under the FIXED parser, baseline + dedup. Diffed against
#      probe-headroom-v1.json (identical arms, pre-fix parser) this isolates
#      what recovering 88% of format failures is worth, and — the number that
#      actually gates training — what it does to usable_group_rate.
#   2. The sub-4B bake-off. Run after the parser fix on purpose: models differ
#      in how much they decorate output, so a pre-fix comparison would have
#      scored markdown habits rather than 21-questions ability.
set -u
cd /home/jackwarren430/Documents/repos/twin-models

run () {
  echo "=== $* ===" >&2
  .venv/bin/python scripts/probe_twentyq_headroom.py "$@" || echo "FAILED: $*" >&2
}

run --arms baseline dedup --output tq-runs/probe-headroom-v2-gemma.json

for spec in \
  "HuggingFaceTB/SmolLM3-3B|smollm3" \
  "unsloth/Llama-3.2-3B-Instruct|llama32"
do
  model="${spec%%|*}"; tag="${spec##*|}"
  run --arms baseline dedup --model "$model" \
      --output "tq-runs/probe-headroom-v2-${tag}.json"
done

echo "BATCH COMPLETE" >&2
