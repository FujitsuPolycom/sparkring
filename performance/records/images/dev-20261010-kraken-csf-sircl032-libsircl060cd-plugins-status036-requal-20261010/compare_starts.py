"""Compare CSF TP4 starts by decode engine steps/s, from each start directory's tables.txt (the serving A/B
runner's report).

python compare_starts.py --references DIR DIR --starts DIR [DIR...]
Prints, per cell (context x streams), the mean of the starts against the mean of the references, then the median
and mean of those differences, the cells below every reference start and the references' own median start-to-start
difference.
"""
import argparse
import re
import statistics
from pathlib import Path


def cells(directory):
    text = (Path(directory) / "tables.txt").read_text().split("Decode aggregate")[0]
    return {(m.group(1), int(m.group(2))): float(m.group(3))
            for m in re.finditer(r"\| (\d+k) \| (\d+) \| ([\d.]+) ", text)}


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--references", nargs=2, required=True)
parser.add_argument("--starts", nargs="+", required=True)
args = parser.parse_args()
refs, starts = [cells(item) for item in args.references], [cells(item) for item in args.starts]
keys = sorted(refs[0], key=lambda key: (int(key[0][:-1]), key[1]))
diffs, below = [], 0
for key in keys:
    reference = statistics.mean(item[key] for item in refs)
    value = statistics.mean(item[key] for item in starts)
    diffs.append(value / reference - 1)
    below += value < min(item[key] for item in refs)
    print(f"{key[0]:>4} x{key[1]}: references {reference:.2f}, starts {value:.2f}, {diffs[-1] * 100:+.1f}%")
spread = statistics.median(abs(refs[0][key] / refs[1][key] - 1) for key in keys)
print(f"median {statistics.median(diffs) * 100:+.1f}%, mean {statistics.mean(diffs) * 100:+.1f}%, {below} of "
      f"{len(keys)} cells below every reference start; references' start-to-start median {spread * 100:.1f}%")
