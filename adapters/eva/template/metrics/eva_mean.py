# /// script
# requires-python = ">=3.10"
# ///
"""The mean EVA reward over scored trials, as a harbor ``uv-script`` metric.

``eva_mean`` is the mean of ``reward`` (the chosen composite); each of EVA's
composites in the rewards (``EVA-A_pass``, ``EVA-X_pass``,
``EVA-overall_pass``) gets its own mean, over the scored trials that have it.

A trial is unscored (``null`` here) when no attempt passed EVA's validation,
when EVA left the composite unset, or when the trial errored. EVA's metrics
cover validated records only, so these are left out of the mean, and
counted. EVA's own run-level numbers (pass@k, pass^k, per-domain means,
bootstrap CIs) come from ``sidecar/aggregate.py`` over the trials'
``metrics.json`` files.
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True)
parser.add_argument("-o", required=True)
args = parser.parse_args()

scored: list[float] = []
named: dict[str, list[float]] = {}
unscored = 0
with open(args.i, encoding="utf-8") as lines:
    for line in lines:
        if not line.strip():
            continue
        reward = json.loads(line)
        if not reward:
            unscored += 1
        else:
            scored.append(float(reward.get("reward", next(iter(reward.values())))))
            for key, value in reward.items():
                if key != "reward":
                    named.setdefault(key, []).append(float(value))

with open(args.o, "w", encoding="utf-8") as out:
    json.dump(
        {
            "eva_mean": sum(scored) / len(scored) if scored else 0.0,
            "scored": len(scored),
            "unscored": unscored,
            **{key: sum(v) / len(v) for key, v in sorted(named.items())},
        },
        out,
    )
