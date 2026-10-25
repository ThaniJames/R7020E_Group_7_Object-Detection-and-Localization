"""Run real/decoy development or the frozen test pipeline."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "outputs" / "models" / "real_decoy_filter.json"

STAGES = {
    "development": (
        ("Check inputs", "check_filter_inputs.py", ()),
        ("Extract depth cues", "extract_depth_cues.py", ()),
        ("Evaluate baseline", "evaluate_filter.py", ()),
        ("Fit and validate filter", "fit_real_decoy_filter.py", ()),
    ),
    "test": (
        ("Extract depth cues", "extract_test_depth_cues.py", ()),
        ("Apply frozen filter", "apply_real_decoy_filter.py", ("test",)),
        ("Evaluate baseline and filter", "evaluate_test_filter.py", ()),
    ),
}


def run_script(filename, args):
    script = SCRIPTS / filename
    if not script.is_file():
        raise FileNotFoundError(script)
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()


def main(phase="test"):
    if phase == "all":
        main("development")
        main("test")
        return
    if phase not in STAGES:
        raise ValueError("phase must be 'all', 'development', or 'test'")
    if phase == "test" and not MODEL.is_file():
        raise FileNotFoundError(f"Frozen filter missing: {MODEL}")

    stages = STAGES[phase]
    print(f"REAL VS DECOY - {phase.upper()}")
    for index, (title, script, args) in enumerate(stages, 1):
        print(f"\n[{index}/{len(stages)}] {title}")
        run_script(script, args)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("Usage: real_vs_decoy_main.py [all|development|test]")
    main(sys.argv[1] if len(sys.argv) == 2 else "test")
