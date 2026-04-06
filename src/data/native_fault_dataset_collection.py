from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

REQUIRED_KEYS = ("process_vars", "controls", "outputs", "regime_id")
OPTIONAL_KEYS = (
    "active_entries",
    "is_fault",
    "regime_seen_in_train",
    "intervention_mask_s",
    "episode_id",
    "matched_normal_reference",
)


class NativeFaultEpisodeDataset(Dataset):
    """Unified native-fault episode dataset.

    Each item contains:
      - process_vars: [T, p]
      - controls: [T, m]
      - outputs: [T, 1]
      - regime_id: scalar long
      - active_entries: [T, 1]
    """

    def __init__(
        self,
        process_vars: np.ndarray,
        controls: np.ndarray,
        outputs: np.ndarray,
        regime_id: np.ndarray,
        active_entries: Optional[np.ndarray] = None,
        is_fault: Optional[np.ndarray] = None,
        regime_seen_in_train: Optional[np.ndarray] = None,
        intervention_mask_s: Optional[np.ndarray] = None,
        episode_id: Optional[np.ndarray] = None,
        matched_normal_reference: Optional[np.ndarray] = None,
    ):
        self.process_vars = np.asarray(process_vars, dtype=np.float32)
        self.controls = np.asarray(controls, dtype=np.float32)
        self.outputs = np.asarray(outputs, dtype=np.float32)
        self.regime_id = np.asarray(regime_id, dtype=np.int64)
        self.active_entries = (
            np.asarray(active_entries, dtype=np.float32)
            if active_entries is not None
            else np.ones_like(self.outputs, dtype=np.float32)
        )
        self.is_fault = (
            np.asarray(is_fault, dtype=np.float32)
            if is_fault is not None
            else (self.regime_id != 0).astype(np.float32)
        )
        self.regime_seen_in_train = (
            np.asarray(regime_seen_in_train, dtype=np.float32)
            if regime_seen_in_train is not None
            else np.ones_like(self.regime_id, dtype=np.float32)
        )
        self.intervention_mask_s = (
            np.asarray(intervention_mask_s, dtype=np.float32)
            if intervention_mask_s is not None
            else np.ones((self.outputs.shape[0], self.outputs.shape[1], 1), dtype=np.float32)
        )
        self.episode_id = (
            np.asarray(episode_id, dtype=np.int64)
            if episode_id is not None
            else np.arange(self.outputs.shape[0], dtype=np.int64)
        )
        self.matched_normal_reference = (
            np.asarray(matched_normal_reference, dtype=np.float32)
            if matched_normal_reference is not None
            else None
        )
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = self.process_vars.shape[0]
        if self.controls.shape[0] != n or self.outputs.shape[0] != n or self.regime_id.shape[0] != n:
            raise ValueError("All episode tensors must share the same leading episode dimension N.")
        if self.outputs.ndim != 3:
            raise ValueError("`outputs` must have shape [N, T, 1] (or [N, T, d_y]).")
        if self.process_vars.ndim != 3 or self.controls.ndim != 3:
            raise ValueError("`process_vars` and `controls` must both be rank-3 arrays: [N, T, d].")
        if self.active_entries.shape[:2] != self.outputs.shape[:2]:
            raise ValueError("`active_entries` must align with the first two dims of `outputs`: [N, T, 1].")
        if self.intervention_mask_s.shape[:2] != self.outputs.shape[:2]:
            raise ValueError(
                "`intervention_mask_s` must align with outputs [N, T, 1] and be broadcastable to s [N,T,D]."
            )
        if self.is_fault.shape[0] != n or self.regime_seen_in_train.shape[0] != n:
            raise ValueError("`is_fault` and `regime_seen_in_train` must be episode-level vectors of length N.")
        if self.episode_id.shape[0] != n:
            raise ValueError("`episode_id` must be an episode-level vector of length N.")

    @classmethod
    def from_dict(cls, payload: Dict[str, np.ndarray]) -> "NativeFaultEpisodeDataset":
        missing = [k for k in REQUIRED_KEYS if k not in payload]
        if missing:
            raise ValueError(f"Missing required keys for NativeFaultEpisodeDataset: {missing}")
        return cls(
            process_vars=payload["process_vars"],
            controls=payload["controls"],
            outputs=payload["outputs"],
            regime_id=payload["regime_id"],
            active_entries=payload.get("active_entries"),
            is_fault=payload.get("is_fault"),
            regime_seen_in_train=payload.get("regime_seen_in_train"),
            intervention_mask_s=payload.get("intervention_mask_s"),
            episode_id=payload.get("episode_id"),
            matched_normal_reference=payload.get("matched_normal_reference"),
        )

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "process_vars": self.process_vars,
            "controls": self.controls,
            "outputs": self.outputs,
            "regime_id": self.regime_id,
            "active_entries": self.active_entries,
            "is_fault": self.is_fault,
            "regime_seen_in_train": self.regime_seen_in_train,
            "intervention_mask_s": self.intervention_mask_s,
            "episode_id": self.episode_id,
            "matched_normal_reference": self.matched_normal_reference,
        }

    def normalize_(
        self,
        stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
        eps: float = 1e-6,
    ) -> "NativeFaultEpisodeDataset":
        for name in ("process_vars", "controls", "outputs"):
            mean, std = stats[name]
            arr = getattr(self, name)
            setattr(self, name, (arr - mean) / (std + eps))
        return self

    def __len__(self) -> int:
        return self.process_vars.shape[0]

    def __getitem__(self, idx: int):
        return {
            "process_vars": torch.tensor(self.process_vars[idx], dtype=torch.float32),
            "controls": torch.tensor(self.controls[idx], dtype=torch.float32),
            "outputs": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "x": torch.tensor(self.process_vars[idx], dtype=torch.float32),
            "y": torch.tensor(self.outputs[idx], dtype=torch.float32),
            "regime_id": torch.as_tensor(self.regime_id[idx], dtype=torch.long),
            "active_entries": torch.tensor(self.active_entries[idx], dtype=torch.float32),
            "is_fault": torch.as_tensor(self.is_fault[idx], dtype=torch.float32),
            "regime_seen_in_train": torch.as_tensor(
                self.regime_seen_in_train[idx], dtype=torch.float32
            ),
            "intervention_mask_s": torch.tensor(
                self.intervention_mask_s[idx], dtype=torch.float32
            ),
            "episode_id": torch.as_tensor(self.episode_id[idx], dtype=torch.long),
            "matched_normal_reference": (
                torch.tensor(self.matched_normal_reference[idx], dtype=torch.float32)
                if self.matched_normal_reference is not None
                else torch.zeros_like(torch.tensor(self.outputs[idx], dtype=torch.float32))
            ),
        }


@dataclass
class NativeFaultSplit:
    train: NativeFaultEpisodeDataset
    val: NativeFaultEpisodeDataset
    test: NativeFaultEpisodeDataset


class NativeFaultDatasetCollection:
    """Loads `.npz` native-fault episodes and creates seen/held-out splits."""

    def __init__(
        self,
        path: str,
        train_regimes: Sequence[int],
        test_regimes: Sequence[int],
        val_ratio: float = 0.1,
        seed: int = 0,
        normalize: bool = True,
        eps: float = 1e-6,
        val_batch_size: int = 256,
        name: str = "native_fault",
        max_seq_length: Optional[int] = None,
        projection_horizon: int = 1,
        autoregressive: bool = True,
        treatment_mode: str = "multilabel",
    ):
        self.path = Path(path)
        self.train_regimes = sorted({int(r) for r in train_regimes})
        self.test_regimes = sorted({int(r) for r in test_regimes})
        self.val_ratio = val_ratio
        self.seed = seed
        self.normalize = normalize
        self.eps = eps
        self.val_batch_size = val_batch_size
        self.name = name
        self.max_seq_length = max_seq_length
        self.projection_horizon = projection_horizon
        self.autoregressive = autoregressive
        self.treatment_mode = treatment_mode

        self.processed_data_multi = False
        self.train_f: Optional[NativeFaultEpisodeDataset] = None
        self.val_f: Optional[NativeFaultEpisodeDataset] = None
        self.test_f: Optional[NativeFaultEpisodeDataset] = None
        self.train_scaling_params: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

        self._build()

    def _build(self) -> None:
        payload = self._load_npz(self.path)
        payload = self._augment_counterfactual_fields(payload)
        split = self._split_seen_and_heldout(payload)
        self.train_f, self.val_f, self.test_f = split.train, split.val, split.test
        self.test_cf_one_step = self.test_f
        self.test_cf_treatment_seq = self.test_f

    @staticmethod
    def _load_npz(path: Path) -> Dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Native-fault data file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}

        missing = [k for k in REQUIRED_KEYS if k not in payload]
        if missing:
            raise ValueError(f"`{path}` is missing required keys: {missing}")
        return payload

    def _split_seen_and_heldout(self, payload: Dict[str, np.ndarray]) -> NativeFaultSplit:
        regime = np.asarray(payload["regime_id"], dtype=np.int64)

        train_mask = np.isin(regime, self.train_regimes)
        test_mask = np.isin(regime, self.test_regimes)

        if not np.any(train_mask):
            raise ValueError("No episodes match `train_regimes`.")
        if not np.any(test_mask):
            raise ValueError("No episodes match `test_regimes`.")

        overlap = set(self.train_regimes).intersection(self.test_regimes)
        if overlap:
            raise ValueError(f"train_regimes and test_regimes overlap: {sorted(overlap)}")

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=self.val_ratio,
            random_state=self.seed,
            shuffle=True,
        )

        train_ds = NativeFaultEpisodeDataset.from_dict(self._slice_payload(payload, train_idx))
        val_ds = NativeFaultEpisodeDataset.from_dict(self._slice_payload(payload, val_idx))
        test_ds = NativeFaultEpisodeDataset.from_dict(self._slice_payload(payload, test_indices))

        if self.normalize:
            stats = self._fit_normalization(train_ds)
            train_ds.normalize_(stats, eps=self.eps)
            val_ds.normalize_(stats, eps=self.eps)
            test_ds.normalize_(stats, eps=self.eps)
            self.train_scaling_params = stats

        return NativeFaultSplit(train=train_ds, val=val_ds, test=test_ds)

    def _augment_counterfactual_fields(self, payload: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        payload = dict(payload)
        regime = np.asarray(payload["regime_id"], dtype=np.int64)
        n, t = payload["outputs"].shape[:2]
        payload.setdefault("is_fault", (regime != 0).astype(np.float32))
        payload.setdefault(
            "regime_seen_in_train", np.isin(regime, self.train_regimes).astype(np.float32)
        )
        payload.setdefault("intervention_mask_s", np.ones((n, t, 1), dtype=np.float32))
        payload.setdefault("episode_id", np.arange(n, dtype=np.int64))
        return payload

    @staticmethod
    def _slice_payload(payload: Dict[str, np.ndarray], indices: Sequence[int]) -> Dict[str, np.ndarray]:
        keep = set(REQUIRED_KEYS).union(OPTIONAL_KEYS)
        return {k: np.asarray(v)[indices] for k, v in payload.items() if k in keep}

    @staticmethod
    def _fit_normalization(
        dataset: NativeFaultEpisodeDataset,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in ("process_vars", "controls", "outputs"):
            arr = getattr(dataset, name)
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True)
            stats[name] = (mean.astype(np.float32), std.astype(np.float32))
        return stats

    @staticmethod
    def make_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def process_data_multi(self, *args, **kwargs):
        _ = args, kwargs
        self.processed_data_multi = True

    def available_regimes(self) -> List[int]:
        regimes = np.concatenate(
            [
                self.train_f.regime_id if self.train_f is not None else np.array([], dtype=np.int64),
                self.val_f.regime_id if self.val_f is not None else np.array([], dtype=np.int64),
                self.test_f.regime_id if self.test_f is not None else np.array([], dtype=np.int64),
            ]
        )
        if regimes.size == 0:
            return []
        return sorted(np.unique(regimes).astype(int).tolist())
