"""Copy the paired colour frames (the ones the pipeline uses) into labels/to_upload/{split} for labeling."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, "src")
from rgbd import pair_frames

for split in ["test", "eval"]:
    out = Path(f"labels/to_upload/{split}")
    out.mkdir(parents=True, exist_ok=True)
    pairs = pair_frames(split)
    for p in pairs:
        shutil.copy2(p.color_path, out / p.color_path.name)  # keep the filename: it holds the timestamp
    print(split, len(pairs), "frames copied to", out)
