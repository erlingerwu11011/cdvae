import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.data.dataset_collection import RealDatasetCollection

logger = logging.getLogger(__name__)


@dataclass
class NativeFaultArrayBundle:
    process_vars: np.ndarray
    controls: np.ndarray
    outputs: np.ndarray
    regime_id: np.ndarray
    active_entries: np.ndarray


class NativeFaultEpisodeDataset(torch.utils.data.Dataset):
    """Episode-style dataset for native-fault industrial soft-sensing.

    Each sample is a full trajectory with canonical keys:
    - process_vars: [T, p]
    - controls: [T, m]
    - outputs: [T, 1]
    - regime_id: scalar regime label (0 = normal)
    - active_entries: [T, 1] mask
    """

    def __init__(
        self,
        bundle: NativeFaultArrayBundle,
        subset_name: str,
        dtype: torch.dtype = torch.double,
    ):
        self.data = {
            "process_vars": bundle.process_vars,
            "controls": bundle.controls,
            "outputs": bundle.outputs,
            "regime_id": bundle.regime_id,
            "active_entries": bundle.active_entries,
        }
        self.subset_name = subset_name
        self.dtype = dtype

        self.scaling_params = None

    def __len__(self) -> int:
        return self.data["process_vars"].shape[0]

    def __getitem__(self, index: int):
        item = {
            "process_vars": torch.as_tensor(self.data["process_vars"][index]).to(self.dtype),
            "controls": torch.as_tensor(self.data["controls"][index]).to(self.dtype),
            "outputs": torch.as_tensor(self.data["outputs"][index]).to(self.dtype),
            "active_entries": torch.as_tensor(self.data["active_entries"][index]).to(self.dtype),
            "regime_id": torch.as_tensor(self.data["regime_id"][index]).long(),
        }

        # Convenience alias for future model migration
        item["quality"] = item["outputs"]
        return item

    def get_scaling_params(self):
        active_mask = self.data["active_entries"].astype(bool)
        active_outputs = self.data["outputs"][active_mask]

        if active_outputs.size == 0:
            raise ValueError("No active entries found when computing scaling params.")

        mean = float(np.mean(active_outputs))
        std = float(np.std(active_outputs))
        if std == 0:
            std = 1.0

        return {
            "output_means": mean,
            "output_stds": std,
        }

    def process_data(self, scaling_params, rep_static=None):
        _ = rep_static
        mean = scaling_params["output_means"]
        std = scaling_params["output_stds"]

        outputs = self.data["outputs"]
        self.data["unscaled_outputs"] = outputs.copy()
        self.data["outputs"] = (outputs - mean) / std
        self.scaling_params = scaling_params


class NativeFaultDatasetCollection(RealDatasetCollection):
    """Dataset collection for native-fault industrial episodes.

    Expects an `.npz` file with keys:
      - process_vars: [N, T, p]
      - controls: [N, T, m]
      - outputs: [N, T, 1] or [N, T]
      - regime_id: [N]
    Optional:
      - active_entries: [N, T, 1] or [N, T]

    Split protocol:
      - train/val: regimes in `train_regimes` (val obtained by ratio split)
      - test: regimes in `test_regimes`
    """

    def __init__(
        self,
        data_path: str,
        train_regimes: Sequence[int],
        test_regimes: Sequence[int],
        val_ratio: float = 0.1,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seed = int(seed)
        self.train_regimes = set(int(r) for r in train_regimes)
        self.test_regimes = set(int(r) for r in test_regimes)
        self.val_ratio = float(val_ratio)

        self.autoregressive = True
        self.has_vitals = True

        arrays = self._load_npz(data_path)
        train_bundle, val_bundle, test_bundle = self._split_by_regime(arrays)

        self.train_f = NativeFaultEpisodeDataset(train_bundle, subset_name="train")
        self.val_f = NativeFaultEpisodeDataset(val_bundle, subset_name="val")
        self.test_f = NativeFaultEpisodeDataset(test_bundle, subset_name="test")

        self.train_scaling_params = self.train_f.get_scaling_params()

    @staticmethod
    def _normalize_outputs_shape(outputs: np.ndarray) -> np.ndarray:
        if outputs.ndim == 2:
            return outputs[..., None]
        if outputs.ndim == 3:
            return outputs
        raise ValueError(f"Unsupported outputs shape: {outputs.shape}")

    @staticmethod
    def _normalize_mask_shape(mask: np.ndarray, target_outputs: np.ndarray) -> np.ndarray:
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape != target_outputs.shape:
            if mask.shape[-1] == 1 and target_outputs.shape[-1] == 1:
                return mask
            raise ValueError(
                f"active_entries shape {mask.shape} is incompatible with outputs shape {target_outputs.shape}."
            )
        return mask

    def _load_npz(self, data_path: str) -> NativeFaultArrayBundle:
        logger.info("Loading native-fault dataset from %s", data_path)
        raw = np.load(data_path)

        required = {"process_vars", "controls", "outputs", "regime_id"}
        missing = required.difference(set(raw.files))
        if missing:
            raise KeyError(f"Missing required keys in {data_path}: {sorted(missing)}")

        process_vars = raw["process_vars"]
        controls = raw["controls"]
        outputs = self._normalize_outputs_shape(raw["outputs"])
        regime_id = raw["regime_id"].astype(np.int64)

        if "active_entries" in raw.files:
            active_entries = self._normalize_mask_shape(raw["active_entries"], outputs)
        else:
            active_entries = np.ones_like(outputs, dtype=np.float64)

        for name, arr in {
            "process_vars": process_vars,
            "controls": controls,
            "outputs": outputs,
            "regime_id": regime_id,
            "active_entries": active_entries,
        }.items():
            if arr.shape[0] != process_vars.shape[0]:
                raise ValueError(f"{name} first dimension mismatch: {arr.shape[0]} != {process_vars.shape[0]}")

        return NativeFaultArrayBundle(
            process_vars=process_vars,
            controls=controls,
            outputs=outputs,
            regime_id=regime_id,
            active_entries=active_entries,
        )

    @staticmethod
    def _subset(bundle: NativeFaultArrayBundle, indices: np.ndarray) -> NativeFaultArrayBundle:
        return NativeFaultArrayBundle(
            process_vars=bundle.process_vars[indices],
            controls=bundle.controls[indices],
            outputs=bundle.outputs[indices],
            regime_id=bundle.regime_id[indices],
            active_entries=bundle.active_entries[indices],
        )

    def _split_by_regime(self, bundle: NativeFaultArrayBundle):
        regime = bundle.regime_id

        train_pool = np.where(np.isin(regime, list(self.train_regimes)))[0]
        test_idx = np.where(np.isin(regime, list(self.test_regimes)))[0]

        if train_pool.size == 0:
            raise ValueError("No samples matched train_regimes.")
        if test_idx.size == 0:
            raise ValueError("No samples matched test_regimes.")

        rng = np.random.default_rng(self.seed)
        shuffled = train_pool.copy()
        rng.shuffle(shuffled)

        n_val = max(1, int(round(self.val_ratio * shuffled.size)))
        n_val = min(n_val, shuffled.size - 1) if shuffled.size > 1 else 0

        val_idx = shuffled[:n_val] if n_val > 0 else np.array([], dtype=np.int64)
        train_idx = shuffled[n_val:] if n_val > 0 else shuffled

        if train_idx.size == 0:
            raise ValueError("Train split is empty after applying val_ratio.")

        logger.info(
            "Native-fault split done: train=%d, val=%d, test=%d",
            train_idx.size,
            val_idx.size,
            test_idx.size,
        )

        train_bundle = self._subset(bundle, train_idx)
        val_bundle = self._subset(bundle, val_idx if val_idx.size > 0 else train_idx[:1])
        test_bundle = self._subset(bundle, test_idx)

        return train_bundle, val_bundle, test_bundle

    def process_data_encoder(self):
        self.train_f.process_data(self.train_scaling_params)
        self.val_f.process_data(self.train_scaling_params)
        self.test_f.process_data(self.train_scaling_params)
        self.processed_data_encoder = True
