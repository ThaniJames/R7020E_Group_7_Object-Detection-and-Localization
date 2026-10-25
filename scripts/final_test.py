"""The final check: run the frozen system once on the test recording.
Steps: YOLO + 3D -> filter (frozen thresholds) -> evaluation vs baseline -> tracking extension -> speed -> figures.
"""
import os, sys, subprocess
from pathlib import Path

env = {**os.environ, "FINAL_TEST": "1"}
PY = sys.executable
FROZEN = Path("outputs/models/real_decoy_filter.json")
if not FROZEN.exists():
    raise SystemExit(f"missing frozen filter {FROZEN}: run the eval (development) steps first")

steps = [
    ([PY, "scripts/detect.py", "test"], "."),
    ([PY, "scripts/extract_test_depth_cues.py"], "."),
    ([PY, "scripts/apply_real_decoy_filter.py", "test"], "."),
    ([PY, "scripts/evaluate_test_filter.py"], "."),
    ([PY, "scripts/evaluate.py", "test"], "."),
    ([PY, "scripts/track_vote.py", "test"], "."),
    ([PY, "scripts/evaluate_tracking.py", "test"], "."),
    ([PY, "scripts/timing.py", "test"], "."),
    ([PY, "scripts/figures_3d.py", "test"], "."),
    ([PY, "scripts/figures_eval.py", "test"], "."),
]
for cmd, cwd in steps:
    print(f"\n### {' '.join(c for c in cmd[1:])}  (in {cwd})", flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if not l.endswith(" boxes")]
    print("\n".join(out[-12:]))
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit(f"step failed: {cmd}")
