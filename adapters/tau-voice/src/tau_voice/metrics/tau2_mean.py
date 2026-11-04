# /// script
# requires-python = ">=3.10"
# ///
"""tau2's average reward, as a harbor ``uv-script`` metric.

Harbor scores a trial that errored (no verifier result) as 0; tau2 leaves an
infrastructure error out of the mean after its retries. Harbor passes errored
trials as ``null`` lines, so this drops them and reports how many it dropped.
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True)
parser.add_argument("-o", required=True)
args = parser.parse_args()

graded: list[float] = []
errored = 0
with open(args.i, encoding="utf-8") as lines:
    for line in lines:
        reward = json.loads(line)
        if reward is None:
            errored += 1
        else:
            graded.append(float(next(iter(reward.values()), 0)))

with open(args.o, "w", encoding="utf-8") as out:
    json.dump(
        {
            "tau2_mean": sum(graded) / len(graded) if graded else 0.0,
            "graded": len(graded),
            "errored": errored,
        },
        out,
    )
