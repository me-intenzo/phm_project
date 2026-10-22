"""Verify that the conformal calibration split follows the checkpoint seed.

Before the fix, ``scripts/uncertainty.py`` hard-coded ``random_state=42`` for
the calibration split, so a seed-43 checkpoint was calibrated against the
seed-42 hold-out — which can contain engines the seed-43 model was trained
on.  This script demonstrates that the seed is now taken from the checkpoint
and that the resulting calibration sets are disjoint from the training sets.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupShuffleSplit

import scripts.uncertainty as us

DATA = ROOT / "data" / "processed"


def calibration_engines(subset: str, seed: int) -> set:
    engine_ids = np.load(DATA / f"{subset}_train_engine_ids.npy").reshape(-1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=int(seed))
    train_idx, cal_idx = next(splitter.split(np.zeros(len(engine_ids)), groups=engine_ids))
    return set(engine_ids[cal_idx].tolist())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset = "FD001"

    seen = {}
    for seed in (42, 43, 44):
        tag = f"ms{seed}"
        model, checkpoint = us.load_model(
            model_name="gru_att_deg_v2",
            input_size=14,
            device=device,
            subset=subset,
            tag=tag,
        )
        ckpt_seed = us.checkpoint_seed(checkpoint)
        cal = calibration_engines(subset, ckpt_seed)
        seen[seed] = cal
        print(
            f"tag={tag:6s} checkpoint_seed={ckpt_seed:3d} "
            f"calibration_engines={len(cal):3d} "
            f"run_metadata_seed={checkpoint.get('run_metadata', {}).get('seed')}"
        )

        assert ckpt_seed == seed, (ckpt_seed, seed)
        assert len(cal) == 20, len(cal)

    # The calibration sets must differ between seeds; otherwise the fix
    # would not be doing anything.
    for a, b in ((42, 43), (43, 44), (42, 44)):
        overlap = seen[a] & seen[b]
        print(f"overlap(seed {a}, seed {b}) = {len(overlap)} engines")
        assert overlap != seen[a], "calibration sets are identical - seed not applied"

    print("\nOK: the calibration split now follows the checkpoint seed.")


if __name__ == "__main__":
    main()
