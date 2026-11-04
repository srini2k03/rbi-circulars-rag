#!/usr/bin/env bash
# Print the judge's measured speed from this job's status files (Markdown, for the step summary).
python3 - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob("results/generation/status-*.json")):
    try:
        d = json.load(open(f))
    except (OSError, ValueError):
        continue
    if d.get("judge_verdicts"):
        rows.append((f.split("/")[-1], d["judge_model"], d["judge_verdicts"], d["seconds_per_verdict"]))
print("## Judge speed\n")
if not rows:
    print("No verdicts in this job.\n")
else:
    print("| status file | judge | verdicts | s / verdict |\n|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()
PY
