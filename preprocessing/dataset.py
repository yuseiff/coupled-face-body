"""
PyTorch Dataset for converted canonical sequences (output of convert_amass.py / convert_beat2.py).

Loads .npz files written by preprocessing/canonical_state.py's Sequence.to_npz(), and yields
fixed-length windows (window_length_frames from configs/default.yaml) as tensors ready for the
geometry generator, descriptor extraction, and coupling prior.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CanonicalSequenceDataset(Dataset):
    def __init__(
        self,
        converted_dir: str | None = None,
        window_length: int = 60,
        stride: int | None = None,
        files: list[str] | None = None,
    ):
        """
        converted_dir: folder of .npz files written by Sequence.to_npz() (e.g. data/beat2/converted/...)
        window_length: number of frames per training window (matches optimization.window_length_frames
                        in configs/default.yaml)
        stride: step between window start points; defaults to window_length (non-overlapping windows)
        files: optional explicit list of .npz paths to use INSTEAD of globbing converted_dir. Use this
               for train/val splits (see scripts/train_prior.py) so the split is done by subject on the
               file list BEFORE constructing the dataset, rather than by directory.
        """
        self.window_length = window_length
        self.stride = stride or window_length

        if files is not None:
            self.files = sorted(files)
        elif converted_dir is not None:
            self.files = sorted(glob.glob(os.path.join(converted_dir, "*.npz")))
        else:
            raise ValueError("Provide either converted_dir or files.")

        if not self.files:
            raise FileNotFoundError(f"No converted .npz files found (converted_dir={converted_dir}).")

        # Precompute (file_index, start_frame) for every valid window across all sequences,
        # so __getitem__ is O(1) rather than re-scanning files each call.
        self._index: list[tuple[int, int]] = []
        self._lengths: list[int] = []
        for fi, path in enumerate(self.files):
            with np.load(path, allow_pickle=True) as d:
                T = d["theta"].shape[0]
            self._lengths.append(T)
            if T < window_length:
                continue  # sequence too short for even one window; skipped, not padded
            for start in range(0, T - window_length + 1, self.stride):
                self._index.append((fi, start))

        if not self._index:
            raise ValueError(
                f"No sequence in {converted_dir} is long enough for window_length={window_length}. "
                f"Sequence lengths found: {sorted(set(self._lengths))[:10]}..."
            )

        # --- preload every file's arrays into RAM once, instead of reopening/decompressing the
        # .npz on every __getitem__ call. With windowing, the same file is opened many times
        # (e.g. a 1900-frame BEAT2 sequence yields ~30 windows) -- without caching this makes
        # training 20-30x+ slower than necessary (measured), since np.load decompresses the
        # full array on every open. Trade-off: higher one-time startup cost + RAM usage (for
        # the full BEAT2 dataset this is roughly 1-2 GB, not per-epoch, just once here).
        print(f"Preloading {len(self.files)} sequence file(s) into memory (one-time cost)...")
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        for fi, path in enumerate(self.files):
            with np.load(path, allow_pickle=True) as d:
                self._cache[fi] = {
                    "beta": d["beta"], "alpha": d["alpha"], "theta": d["theta"],
                    "psi": d["psi"], "gaze": d["gaze"], "cam": d["cam"],
                    "subject_id": str(d["subject_id"]), "source_dataset": str(d["source_dataset"]),
                }
        print("Preload complete.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, start = self._index[idx]
        end = start + self.window_length
        c = self._cache[fi]

        return {
            "beta": torch.from_numpy(c["beta"]).float(),
            "alpha": torch.from_numpy(c["alpha"]).float(),
            "theta": torch.from_numpy(c["theta"][start:end]).float(),
            "psi": torch.from_numpy(c["psi"][start:end]).float(),
            "gaze": torch.from_numpy(c["gaze"][start:end]).float(),
            "cam": torch.from_numpy(c["cam"][start:end]).float(),
            "subject_id": c["subject_id"],
            "source_dataset": c["source_dataset"],
        }


if __name__ == "__main__":
    # Smoke test: point this at a real converted_dir once you have one, e.g.:
    #   python preprocessing/dataset.py --converted_dir data/beat2/converted/beat_english_v2.0.0
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--window_length", type=int, default=60)
    args = parser.parse_args()

    ds = CanonicalSequenceDataset(args.converted_dir, window_length=args.window_length)
    print(f"OK: dataset has {len(ds)} windows across {len(ds.files)} sequences")

    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
