from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NativeFaultEpisode:
    """Container for one native-fault trajectory."""

    process_vars: np.ndarray  # [T, p]
    controls: np.ndarray  # [T, m]
    outputs: np.ndarray  # [T, 1]
    regime_id: int
    active_entries: Optional[np.ndarray] = None  # [T, 1]


class TEPNativeFaultDataset(Dataset):
    """TEP native-fault episodes with a unified industrial batch schema."""

    def __init__(self, data: Dict[str, np.ndarray]):
        required = ["process_vars", "controls", "outputs", "regime_id"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing required keys for TEP dataset: {missing}")

        self.process_vars = np.asarray(data["process_vars"], dtype=np.float32)
        self.controls = np.asarray(data["controls"], dtype=np.float32)
        self.outputs = np.asarray(data["outputs"], dtype=np.float32)
        self.regime_id = np.asarray(data["regime_id"], dtype=np.int64)

        if "active_entries" in data and data["active_entries"] is not None:
            self.active_entries = np.asarray(data["active_entries"], dtype=np.float32)
        else:
            self.active_entries = np.ones_like(self.outputs, dtype=np.float32)

        n = self.process_vars.shape[0]
        if not (self.controls.shape[0] == self.outputs.shape[0] == self.regime_id.shape[0] == n):
            raise ValueError("All leading dimensions must match for TEP native-fault episodes.")

    def __len__(self) -> int:
        return self.process_vars.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "process_vars": torch.tensor(self.process_vars[idx], dtype=torch.float32),
            "controls": torch.tensor(self.controls[idx], dtype=torch.float32),
            "outputs": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "regime_id": torch.as_tensor(self.regime_id[idx], dtype=torch.long),
            "active_entries": torch.tensor(self.active_entries[idx], dtype=torch.float32),
        }
