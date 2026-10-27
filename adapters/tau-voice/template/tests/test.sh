#!/bin/bash
# The task's verifier, run in main as harbor runs every verifier. The grade is
# tau2's own evaluator plus tau2's hallucination check, which need tau2 — so
# the tau2 service runs them (POST /grade) on the call state it holds. Writes
# /logs/verifier/{reward.txt,result.json}.
set -euo pipefail
mkdir -p /logs/verifier
python3 - <<'PY'
import json
import urllib.request

request = urllib.request.Request(
    "http://tau2:8000/grade",
    data=json.dumps({"hallucination_check": True}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=840) as response:
    result = json.load(response)
with open("/logs/verifier/result.json", "w", encoding="utf-8") as out:
    json.dump(result, out, indent=2)
with open("/logs/verifier/reward.txt", "w", encoding="utf-8") as out:
    out.write(f"{result['reward']}\n")
PY
