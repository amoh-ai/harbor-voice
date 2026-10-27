#!/bin/bash
# The task's verifier, baked into the tau2 image at /tests/test.sh: grades the
# call with tau2's own evaluator plus tau2's hallucination check, and writes
# /logs/verifier/{reward.txt,result.json}.
set -euo pipefail
mkdir -p /logs/verifier
python /app/evaluate.py \
  --state /logs/agent/tau_voice_state.json \
  --reward /logs/verifier/reward.txt \
  --result /logs/verifier/result.json \
  --review-model "${TAU_VOICE_REVIEW_MODEL}"
