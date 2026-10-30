#!/bin/bash
# The task's verifier, run in main as harbor runs every verifier. EVA's grade
# needs EVA, so the eva service runs it (POST /grade) on the record it holds;
# the driver graded already, and EVA reuses those metrics, so no judge is
# called twice. Writes /logs/verifier/{result.json,reward.json}.
set -euo pipefail
mkdir -p /logs/verifier
exec python3 "$(dirname "$0")/grade.py" --reward "__REWARD__" \
    --sidecar http://eva:8000 --out /logs/verifier
