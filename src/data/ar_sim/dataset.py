import collections
import copy
import gzip
import logging
import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch

from src.data.ar_sim.ar_simulation import (
    CounterFactualDataSimulator,
    CounterFactualDataSimulator_seq,
    FactualDataSimulator,
)
from src.data.dataset_collection import SyntheticDatasetCollection

DefaultDict = collections.defaultdict
deepcopy = copy.deepcopy
Dataset = torch.utils.data.Dataset

logger = logging.getLogger(__name__)


class ARSimulationDataset(Dataset):
    """Pytorch-style Dataset of Tumor Growth Simulator datasets."""

    def __init__(
        self,
        d_vitals: int,
        d_random_effects: int,
        treat_imb: float,
        p: int,
        cov_rho: float,
        cov_var: float,
        re_centers: int,
        re_cluster_std,
        num_patients: int,
        window_size: int,
        seq_length: int,
        subset_name: str,
        mode: str = "factual",
        projection_horizon: int = None,
        seed: int = None,
        lag: int = 0,
        num_treatments: int = 1,
        coeff: float = 1.2,
        cf_seq_mode: str = "sliding_treatment",
        treatment_mode: str = "multiclass",
        type_static_feat: str = None,
    ):

        if seed is not None:
            np.random.seed(seed)

        self.subset_name = subset_name

        self.d_vitals = d_vitals

        self.norm_const = 1.0

        self.type_static_feat = type_static_feat

        if mode == "factual":
            self.data = FactualDataSimulator(
                num_patients=num_patients,
                d_vitals=d_vitals,
                d_random_effects=d_random_effects,
                seq_length=seq_length,
                treat_imb=treat_imb,
                p=p,
                cov_rho=cov_rho,
                cov_var=cov_var,
                re_centers=re_centers,
                re_cluster_std=re_cluster_std,
                coeff=coeff,
            ).get_simulation_output()

        elif mode == "counterfactual_one_step":
            self.data = CounterFactualDataSimulator(
                num_patients=num_patients,
                d_vitals=d_vitals,
                d_random_effects=d_random_effects,
                seq_length=seq_length,
                treat_imb=treat_imb,
                p=p,
                cov_rho=cov_rho,
                cov_var=cov_var,
                re_centers=re_centers,
                re_cluster_std=re_cluster_std,
                num_treatments=num_treatments,
                coeff=coeff,
            ).get_simulation_output()

        elif mode == "counterfactual_treatment_seq":
            assert projection_horizon is not None
            self.data = CounterFactualDataSimulator_seq(
                num_patients=num_patients,
                d_vitals=d_vitals,
                d_random_effects=d_random_effects,
                seq_length=seq_length,
                treat_imb=treat_imb,
                p=p,
                cov_rho=cov_rho,
                cov_var=cov_var,
                re_centers=re_centers,
                re_cluster_std=re_cluster_std,
                projection_horizon=projection_horizon,
                cf_seq_mode=cf_seq_mode,
                coeff=coeff,
            ).get_simulation_output()

        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.treatment_mode = treatment_mode
        self.exploded = False

    def reinitialize_processing(self):
        # self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False

    def __getitem__(self, index):
        result = {}

        if "augmented_lengths_csum" in self.data:
            real_index = (
                np.searchsorted(self.data["augmented_lengths_csum"], index, side="right") - 1
            )
            future_past_split = index - self.data["augmented_lengths_csum"][real_index] + 1
            result["future_past_split"] = torch.tensor(future_past_split).long()
            index = real_index
            result.update(
                {
                    k: torch.as_tensor(v[index]).to(torch.get_default_dtype())
                    for k, v in self.data.items()
                    if hasattr(v, "__len__") and len(v) == self.data["vitals"].shape[0]
                }
            )
        else:
            result = {
                k: torch.as_tensor(v[index]).to(torch.get_default_dtype())
                for k, v in self.data.items()
                if hasattr(v, "__len__") and len(v) == len(self)
            }
        if hasattr(self, "encoder_r"):
            if "original_index" in self.data:
                result.update(
                    {
                        "encoder_r": torch.as_tensor(
                            self.encoder_r[int(result["original_index"])]
                        ).to(torch.get_default_dtype())
                    }
                )
            else:
                result.update(
                    {
                        "encoder_r": torch.as_tensor(self.encoder_r[index]).to(
                            torch.get_default_dtype()
                        )
                    }
                )
        return result

    def __len__(self):
        if "augmented_lengths_csum" in self.data:
            return self.data["augmented_lengths_csum"][-1]
        else:
            return self.data["vitals"].shape[0]

    def get_scaling_params(self):

        real_idx = ["outcome"]

        means = {}
        stds = {}
        seq_lengths = self.data["sequence_lengths"]
        for k in real_idx:
            active_values = []
            for i in range(seq_lengths.shape[0]):
                end = int(seq_lengths[i])
                active_values += list(self.data[k][i, :end])

            means[k] = np.mean(active_values)
            stds[k] = np.std(active_values)

        return pd.Series(means), pd.Series(stds)

    def process_data(self, scaling_params, rep_static=None):
        """Pre-process dataset for one-step-ahead prediction.

        Args:
            scaling_params: dict of standard normalization parameters (calculated
            with train subset)

        Returns:
            dict self.data
        """

        if not self.processed:
            logger.info("%s", f"Processing {self.subset_name} dataset before training")

            if self.type_static_feat == "true":
                static_features = self.data["random_effects"]
                print("static_features", static_features.shape)
                self.data["static_features"] = static_features.reshape(
                    -1, static_features.shape[1]
                )

            self._encode_treatments()
            self._init_prev_treatments()
            self._process_vitals()
            self._scale_outputs(scaling_params)
            self._init_prev_outputs()
            self._add_active_entries()

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info("%s", f"Shape of processed {self.subset_name} data: {data_shapes}")

            self.processed = True
        else:
            self._handle_already_processed(rep_static)

        return self.data

    def _encode_treatments(self):
        treatments = self.data["treatment"][:, :, np.newaxis]

        if self.treatment_mode == "multiclass":
            unique_treatments = np.unique(treatments.reshape(-1, treatments.shape[-1]), axis=0)
            num_treatments = unique_treatments.shape[0]
            one_hot_treatments = np.zeros(
                (treatments.shape[0], treatments.shape[1], num_treatments)
            )

            for patient_id in range(treatments.shape[0]):
                for timestep in range(treatments.shape[1]):
                    treatment_index = np.where(
                        (unique_treatments == treatments[patient_id][timestep]).all(axis=1)
                    )[0][0]
                    one_hot_treatments[patient_id][timestep][treatment_index] = 1

            self.data["prev_treatments"] = one_hot_treatments[:, :-1, :]
            self.data["current_treatments"] = one_hot_treatments

        elif self.treatment_mode == "multilabel":
            self.data["prev_treatments"] = treatments[:, :-1, :]
            self.data["current_treatments"] = treatments

    def _init_prev_treatments(self):
        zero_init_treatment = np.zeros(
            shape=[
                self.data["treatment"].shape[0],
                1,
                self.data["prev_treatments"].shape[-1],
            ]
        )
        self.data["prev_treatments"] = np.concatenate(
            [zero_init_treatment, self.data["prev_treatments"]], axis=1
        )

    def _process_vitals(self):
        vitals = np.transpose(self.data["vitals"], axes=(0, 2, 1))
        self.data["vitals"] = vitals

        zero_init_vitals = np.zeros(
            shape=[
                vitals.shape[0],
                1,
                vitals.shape[-1],
            ]
        )

        self.data["prev_vitals"] = np.concatenate(
            [zero_init_vitals, self.data["vitals"][:, :-1, :]], axis=1
        )

        next_vitals = np.transpose(self.data["next_vitals"], axes=(0, 2, 1))
        self.data["next_vitals"] = next_vitals

    def _scale_outputs(self, scaling_params):
        mean, std = scaling_params

        input_means = mean[
            [
                "outcome",
            ]
        ].values.flatten()
        input_stds = std[
            [
                "outcome",
            ]
        ].values.flatten()

        outcome = (self.data["outcome"] - mean["outcome"]) / std["outcome"]
        outputs = outcome[:, :, np.newaxis]
        self.data["outputs"] = outputs

        output_means = mean[["outcome"]].values.flatten()[0]
        output_stds = std[["outcome"]].values.flatten()[0]

        self.data["unscaled_outputs"] = outputs * std["outcome"] + mean["outcome"]

        self.scaling_params = {
            "input_means": input_means,
            "inputs_stds": input_stds,
            "output_means": output_means,
            "output_stds": output_stds,
        }

    def _init_prev_outputs(self):
        outcome = (
            self.data["outcome"] - self.scaling_params["output_means"]
        ) / self.scaling_params["output_stds"]
        self.data["prev_outputs"] = outcome[:, :-1, np.newaxis]
        zero_init_output = np.zeros(
            shape=[
                self.data["treatment"].shape[0],
                1,
                self.data["prev_outputs"].shape[-1],
            ]
        )
        self.data["prev_outputs"] = np.concatenate(
            [zero_init_output, self.data["prev_outputs"]], axis=1
        )

    def _add_active_entries(self):
        sequence_lengths = self.data["sequence_lengths"]
        active_entries = np.zeros(self.data["outputs"].shape)
        for i in range(sequence_lengths.shape[0]):
            sequence_length = int(sequence_lengths[i])
            active_entries[i, :sequence_length, :] = 1

        self.data["active_entries"] = active_entries

    def _handle_already_processed(self, rep_static):

        print("self.type_static_feat", self.type_static_feat)

        if self.type_static_feat == "rep":
            if rep_static is not None:
                self.data["static_features"] = rep_static
                logger.info("%s", f"rep_static are added to {self.subset_name} Dataset")

        x_keys = [
            "vitals",
            "prev_outputs",
            "current_treatments",
        ]
        x_keys = [x for x in x_keys if x in self.data]

        for x_key in x_keys:
            self.data[x_key] = np.nan_to_num(self.data[x_key])

        logger.info("%s", f"{self.subset_name} Dataset already processed")

    def explode_trajectories(self, projection_horizon):
        assert self.processed

        logger.info(
            "%s",
            f"Exploding {self.subset_name} dataset before testing (multiple" " sequences)",
        )

        outputs = self.data["outputs"]
        prev_outputs = self.data["prev_outputs"]
        sequence_lengths = self.data["sequence_lengths"]
        vitals = self.data["vitals"]
        next_vitals = self.data["next_vitals"]
        active_entries = self.data["active_entries"]
        current_treatments = self.data["current_treatments"]
        previous_treatments = self.data["prev_treatments"]

        static_features = (
            self.data["static_features"] if self.type_static_feat in ["rep", "true"] else None
        )

        if "stabilized_weights" in self.data:
            stabilized_weights = self.data["stabilized_weights"]
        else:
            stabilized_weights = None

        num_patients, max_seq_length, _ = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1])
        )
        seq2seq_current_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, current_treatments.shape[-1])
        )
        seq2seq_static_features = (
            np.zeros((num_seq2seq_rows, static_features.shape[-1]))
            if static_features is not None
            else None
        )
        seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length, next_vitals.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        seq2seq_active_entries = np.zeros(
            (num_seq2seq_rows, max_seq_length, active_entries.shape[-1])
        )
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        if "stabilized_weights" in self.data:
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))
        else:
            seq2seq_stabilized_weights = None

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])

            for t in range(projection_horizon, sequence_length):  # shift outputs back by 1
                seq2seq_active_entries[total_seq2seq_rows, : (t + 1), :] = active_entries[
                    i, : (t + 1), :
                ]
                if "stabilized_weights" in self.data:
                    seq2seq_stabilized_weights[total_seq2seq_rows, : (t + 1)] = stabilized_weights[
                        i, : (t + 1)
                    ]

                seq2seq_vitals[total_seq2seq_rows, : (t + 1), :] = vitals[i, : (t + 1), :]
                seq2seq_next_vitals[total_seq2seq_rows, : (t + 1), :] = next_vitals[
                    i, : (t + 1), :
                ]
                seq2seq_previous_treatments[total_seq2seq_rows, : (t + 1), :] = (
                    previous_treatments[i, : (t + 1), :]
                )
                seq2seq_current_treatments[total_seq2seq_rows, : (t + 1), :] = current_treatments[
                    i, : (t + 1), :
                ]
                seq2seq_outputs[total_seq2seq_rows, : (t + 1), :] = outputs[i, : (t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, : (t + 1), :] = prev_outputs[
                    i, : (t + 1), :
                ]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1

                if static_features is not None:
                    seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]

        if static_features is not None:
            seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]

        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        if "stabilized_weights" in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            "prev_treatments": seq2seq_previous_treatments,
            "current_treatments": seq2seq_current_treatments,
            "prev_outputs": seq2seq_prev_outputs,
            "outputs": seq2seq_outputs,
            "vitals": seq2seq_vitals,
            "next_vitals": seq2seq_next_vitals,
            "unscaled_outputs": (
                seq2seq_outputs * self.scaling_params["output_stds"]
                + self.scaling_params["output_means"]
            ),
            "sequence_lengths": seq2seq_sequence_lengths,
            "active_entries": seq2seq_active_entries,
        }

        if static_features is not None:
            new_data["static_features"] = seq2seq_static_features

        if "stabilized_weights" in self.data:
            new_data["stabilized_weights"] = seq2seq_stabilized_weights

        self.data = new_data
        self.exploded = True

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info("%s", f"Shape of processed {self.subset_name} data: {data_shapes}")

    def process_sequential(self, encoder_r, projection_horizon, save_encoder_r=False):
        """Pre-process dataset for multiple-step-ahead prediction.

        explodes dataset to a larger one with rolling origin

        Args:
            encoder_r: Representations of encoder
            projection_horizon: Projection horizon
            save_encoder_r: Save all encoder representations (for cross-attention of
              EDCT)

        Returns:
            exploded dataset
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(
                "%s",
                f"Processing {self.subset_name} dataset before training (multiple" " sequences)",
            )

            outputs = self.data["outputs"]
            sequence_lengths = self.data["sequence_lengths"]
            active_entries = self.data["active_entries"]
            current_treatments = self.data["current_treatments"]
            previous_treatments = self.data["prev_treatments"][
                :, 1:, :
            ]  # Without zero_init_treatment
            vitals = self.data["vitals"]

            static_features = (
                self.data["static_features"] if self.type_static_feat in ["rep", "true"] else None
            )

            stabilized_weights = (
                self.data["stabilized_weights"] if "stabilized_weights" in self.data else None
            )

            num_patients, seq_length, _ = outputs.shape

            num_seq2seq_rows = num_patients * seq_length

            seq2seq_static_features = (
                np.zeros((num_seq2seq_rows, static_features.shape[-1]))
                if static_features is not None
                else None
            )

            seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, seq_length))
            seq2seq_original_index = np.zeros((num_seq2seq_rows,))
            seq2seq_previous_treatments = np.zeros(
                (num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1])
            )
            seq2seq_current_treatments = np.zeros(
                (num_seq2seq_rows, projection_horizon, current_treatments.shape[-1])
            )
            seq2seq_vitals = np.zeros((num_seq2seq_rows, projection_horizon, vitals.shape[-1]))
            seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros(
                (num_seq2seq_rows, projection_horizon, active_entries.shape[-1])
            )
            seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
            seq2seq_stabilized_weights = (
                np.zeros((num_seq2seq_rows, projection_horizon + 1))
                if stabilized_weights is not None
                else None
            )

            total_seq2seq_rows = 0  # we use this to shorten any trajectories later

            for i in range(num_patients):
                sequence_length = int(sequence_lengths[i])

                for t in range(1, sequence_length - projection_horizon):  # shift outputs back by 1
                    seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[
                        i, t - 1, :
                    ]  # previous state output
                    seq2seq_original_index[total_seq2seq_rows] = i
                    seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                    max_projection = min(projection_horizon, sequence_length - t)

                    seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = (
                        active_entries[i, t : t + max_projection, :]
                    )
                    seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = (
                        previous_treatments[i, t - 1 : t + max_projection - 1, :]
                    )
                    seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = (
                        current_treatments[i, t : t + max_projection, :]
                    )
                    seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[
                        i, t : t + max_projection, :
                    ]
                    seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                    seq2seq_vitals[total_seq2seq_rows, :max_projection, :] = vitals[
                        i, t : t + max_projection, :
                    ]

                    if (
                        seq2seq_stabilized_weights is not None
                    ):  # Also including SW of one-step-ahead prediction
                        seq2seq_stabilized_weights[total_seq2seq_rows, :] = stabilized_weights[
                            i, t - 1 : t + max_projection
                        ]

                    if static_features is not None:
                        seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                    total_seq2seq_rows += 1

            # Filter everything shorter
            seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
            seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]

            if static_features is not None:
                seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]

            seq2seq_active_encoder_r = seq2seq_active_encoder_r[:total_seq2seq_rows, :]
            seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
            seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
            seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
            seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
            seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]
            if seq2seq_stabilized_weights is not None:
                seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

            # Package outputs
            seq2seq_data = {
                "init_state": seq2seq_state_inits,
                "original_index": seq2seq_original_index,
                "active_encoder_r": seq2seq_active_encoder_r,
                "prev_treatments": seq2seq_previous_treatments,
                "current_treatments": seq2seq_current_treatments,
                "vitals": seq2seq_vitals,
                "prev_outputs": seq2seq_vitals[:, :, :1],
                "outputs": seq2seq_outputs,
                "sequence_lengths": seq2seq_sequence_lengths,
                "active_entries": seq2seq_active_entries,
                "unscaled_outputs": (
                    seq2seq_outputs * self.scaling_params["output_stds"]
                    + self.scaling_params["output_means"]
                ),
            }

            if self.type_static_feat in ["rep", "true"]:
                seq2seq_data["static_features"] = seq2seq_static_features

            if seq2seq_stabilized_weights is not None:
                seq2seq_data["stabilized_weights"] = seq2seq_stabilized_weights

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info("%s", f"Shape of processed {self.subset_name} data: {data_shapes}")

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :seq_length, :]

            self.processed_sequential = True
            self.exploded = True

        else:
            logger.info(
                "%s",
                f"{self.subset_name} Dataset already processed (multiple sequences)",
            )

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        """Pre-process test dataset for multiple-step-ahead prediction.

        takes the last n-steps according to the projection horizon

        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of
              EDCT)

        Returns:
            processed sequential test data
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(
                "%s",
                f"Processing {self.subset_name} dataset before testing (multiple" " sequences)",
            )

            sequence_lengths = self.data["sequence_lengths"]
            outputs = self.data["outputs"]
            current_treatments = self.data["current_treatments"]
            previous_treatments = self.data["prev_treatments"][
                :, 1:, :
            ]  # Without zero_init_treatment
            vitals = self.data["vitals"]

            num_patient_points, max_seq_length, _ = outputs.shape

            if self.type_static_feat in ["rep", "true"]:
                print("self.data", self.data.keys())
                static_features = self.data["static_features"]
            else:
                static_features = None

            seq2seq_static_features = (
                np.zeros((num_patient_points, static_features.shape[-1]))
                if static_features is not None
                else None
            )

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
            else:
                seq2seq_state_inits = None
            seq2seq_active_encoder_r = np.zeros(
                (num_patient_points, max_seq_length - projection_horizon)
            )
            seq2seq_previous_treatments = np.zeros(
                (
                    num_patient_points,
                    projection_horizon,
                    previous_treatments.shape[-1],
                )
            )
            seq2seq_current_treatments = np.zeros(
                (num_patient_points, projection_horizon, current_treatments.shape[-1])
            )
            seq2seq_vitals = np.zeros((num_patient_points, projection_horizon, vitals.shape[-1]))
            seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
            seq2seq_sequence_lengths = np.zeros(num_patient_points)

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                if fact_length >= 1:
                    seq2seq_previous_treatments[i] = previous_treatments[
                        i, fact_length - 1 : fact_length + projection_horizon - 1, :
                    ]
                seq2seq_current_treatments[i] = current_treatments[
                    i, fact_length : fact_length + projection_horizon, :
                ]
                seq2seq_outputs[i] = outputs[i, fact_length : fact_length + projection_horizon, :]
                seq2seq_sequence_lengths[i] = projection_horizon
                # Disabled teacher forcing for test dataset
                if fact_length >= 1:
                    seq2seq_vitals[i] = np.repeat(
                        [vitals[i, fact_length - 1]],
                        projection_horizon,
                        axis=0,
                    )
                if static_features is not None:
                    seq2seq_static_features[i] = static_features[i]

            # Package outputs
            seq2seq_data = {
                "active_encoder_r": seq2seq_active_encoder_r,
                "prev_treatments": seq2seq_previous_treatments,
                "current_treatments": seq2seq_current_treatments,
                "vitals": seq2seq_vitals,
                "prev_outputs": seq2seq_vitals[:, :, :1],
                "outputs": seq2seq_outputs,
                "sequence_lengths": seq2seq_sequence_lengths,
                "active_entries": seq2seq_active_entries,
                "unscaled_outputs": (
                    seq2seq_outputs * self.scaling_params["output_stds"]
                    + self.scaling_params["output_means"]
                ),
                "patient_ids_all_trajectories": (
                    self.data["patient_ids_all_trajectories"]
                    if "patient_ids_all_trajectories" in self.data
                    else None
                ),
                "patient_current_t": (
                    self.data["patient_current_t"] if "patient_curent_t" in self.data else None
                ),
            }
            if self.type_static_feat in ["rep", "true"]:
                seq2seq_data["static_features"] = seq2seq_static_features

            if encoder_r is not None:
                seq2seq_data["init_state"] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data = seq2seq_data
            data_shapes = {k: v.shape for k, v in self.data.items() if v is not None}
            logger.info("%s", f"Shape of processed {self.subset_name} data: {data_shapes}")

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, : max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(
                "%s",
                f"{self.subset_name} Dataset already processed (multiple sequences)",
            )

        return self.data

    def process_autoregressive_test(
        self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False
    ):
        """Pre-process test dataset for multiple-step-ahead prediction.

        axillary dataset placeholder for autoregressive prediction

        Args:
            encoder_r: Representations of encoder
            encoder_outputs: encoder outputs
            projection_horizon: Projection horizon
            save_encoder_r: Save all encoder representations (for cross-attention of
              EDCT)

        Returns:
            autoregressive test data
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            logger.info(
                "%s",
                f"Processing {self.subset_name} dataset before testing" " (autoregressive)",
            )

            current_treatments = self.data_original["current_treatments"]
            prev_treatments = self.data_original["prev_treatments"][
                :, 1:, :
            ]  # Without zero_init_treatment

            sequence_lengths = self.data_original["sequence_lengths"]
            num_patient_points, max_seq_length = current_treatments.shape[:2]

            current_dataset = dict()  # Same as original, but only with last n-steps
            current_dataset["vitals"] = np.zeros(
                (
                    num_patient_points,
                    projection_horizon,
                    self.data_original["vitals"].shape[-1],
                )
            )
            current_dataset["prev_treatments"] = np.zeros(
                (
                    num_patient_points,
                    projection_horizon,
                    self.data_original["prev_treatments"].shape[-1],
                )
            )
            current_dataset["current_treatments"] = np.zeros(
                (
                    num_patient_points,
                    projection_horizon,
                    self.data_original["current_treatments"].shape[-1],
                )
            )
            current_dataset["init_state"] = np.zeros((num_patient_points, encoder_r.shape[-1]))
            current_dataset["active_encoder_r"] = np.zeros(
                (num_patient_points, max_seq_length - projection_horizon)
            )
            current_dataset["active_entries"] = np.ones(
                (num_patient_points, projection_horizon, 1)
            )

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                current_dataset["init_state"][i] = encoder_r[i, fact_length - 1]
                if encoder_outputs is not None:
                    current_dataset["vitals"][i, 0, 0] = encoder_outputs[i, fact_length - 1]
                current_dataset["active_encoder_r"][i, :fact_length] = 1.0
                current_dataset["prev_treatments"][i] = prev_treatments[
                    i, fact_length - 1 : fact_length + projection_horizon - 1, :
                ]
                current_dataset["current_treatments"][i] = current_treatments[
                    i, fact_length : fact_length + projection_horizon, :
                ]

            current_dataset["prev_outputs"] = current_dataset["vitals"][:, :, :1]

            if self.type_static_feat in ["rep", "true"]:
                current_dataset["static_features"] = self.data_original["static_features"]

            self.data_processed_seq = deepcopy(self.data)
            self.data = current_dataset
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info("%s", f"Shape of processed {self.subset_name} data: {data_shapes}")

            if save_encoder_r:
                self.encoder_r = encoder_r[:, : max_seq_length - projection_horizon, :]

            self.processed_autoregressive = True

        else:
            logger.info("%s", f"{self.subset_name} Dataset already processed (autoregressive)")

        return self.data

    def process_sequential_multi(self, projection_horizon):
        """Pre-process test dataset for multiple-step-ahead prediction.

        for multi-input model: marking rolling origin with
            'future_past_split'

        Args:
            projection_horizon: Projection horizon

        Returns:
            processed sequential multi data
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data["future_past_split"] = self.data["sequence_lengths"] - projection_horizon
            self.processed_autoregressive = True

        else:
            logger.info("%s", f"{self.subset_name} Dataset already processed (autoregressive)")

        return self.data

    def process_sequential_rep_est(self, projection_horizon):
        assert self.processed

        if not self.processed_autoregressive:
            self.data["future_past_split"] = self.data["sequence_lengths"] - projection_horizon
            self.process_autoregressive = True
        else:
            logger.info("%s", f"{self.subset_name} Dataset already processed (autoregressive)")

        return self.data

    def process_sequential_split(self, rep_static=None):
        assert self.processed
        logger.info("%s", f"Augmenting {self.subset_name} dataset before training")

        self.data_before_aug = deepcopy(self.data)
        data_length = self.data_before_aug["prev_treatments"].shape[0]
        valid_keys = [
            k
            for k, v in self.data_before_aug.items()
            if hasattr(v, "__len__") and len(v) == data_length
        ]
        self.data = DefaultDict(list)
        for patient_id in range(data_length):
            sample_seq_len = int(self.data_before_aug["sequence_lengths"][patient_id])
            sample_future_past_split = np.arange(1, sample_seq_len + 1, dtype=np.int64)
            self.data["future_past_split"].append(sample_future_past_split)
            for vk in valid_keys:
                self.data[vk].append(
                    np.repeat(
                        self.data_before_aug[vk][patient_id][None, Ellipsis],
                        sample_seq_len,
                        axis=0,
                    )
                )
        for k in self.data:
            self.data[k] = np.concatenate(self.data[k], axis=0)
        return self.data


class ARSimulationDatasetCollection(SyntheticDatasetCollection):
    """ARSimulationDataset collection.

    (train_f, val_f, test_cf_one_step, test_cf_treatment_seq)
    """

    def __init__(
        self,
        d_vitals: int,
        d_random_effects: int,
        treat_imb: float,
        p: int,
        cov_rho: float,
        cov_var: float,
        re_centers: int,
        re_cluster_std,
        num_patients: int,
        window_size: int,
        seq_length: int,
        subset_name: str,
        mode: str = "factual",
        projection_horizon: int = None,
        seed: int = None,
        lag: int = 0,
        num_treatments: int = 1,
        coeff: float = 1.2,
        cf_seq_mode: str = "sliding_treatment",
        treatment_mode: str = "multiclass",
        type_static_feat: str = None,
        **kwargs,
    ):
        """Initialization.

        Args:
          num_patients: Number of patients in dataset
          seed: random seed
          window_size: Used for biased treatment assignment
          max_seq_length: Max length of time series
          projection_horizon: Range of tau-step-ahead prediction (tau =
            projection_horizon + 1)
          lag: Lag for treatment assignment window
          cf_seq_mode: sliding_treatment / random_trajectories
          treatment_mode: multiclass / multilabel
          few_shot_sample_num: if > 0, resampling training samples to make the
            dataset few-shot learning
          same_subjs_train_test: use same subjects in train and test
          **kwargs: other args

        Returns:
          dataset collection
        """
        super().__init__()
        self.seed = seed
        np.random.seed(self.seed)
        self.type_static_feat = type_static_feat

        self.train_f = ARSimulationDataset(
            d_vitals=d_vitals,
            d_random_effects=d_random_effects,
            treat_imb=treat_imb,
            p=p,
            cov_rho=cov_rho,
            cov_var=cov_var,
            re_centers=re_centers,
            re_cluster_std=re_cluster_std,
            num_patients=num_patients["train"],
            window_size=window_size,
            seq_length=seq_length,
            subset_name="train",
            seed=seed,
            lag=lag,
            treatment_mode=treatment_mode,
            coeff=coeff,
            type_static_feat=self.type_static_feat,
        )

        self.val_f = ARSimulationDataset(
            d_vitals=d_vitals,
            d_random_effects=d_random_effects,
            treat_imb=treat_imb,
            p=p,
            cov_rho=cov_rho,
            cov_var=cov_var,
            re_centers=re_centers,
            re_cluster_std=re_cluster_std,
            num_patients=num_patients["val"],
            window_size=window_size,
            seq_length=seq_length,
            subset_name="val",
            seed=seed,
            lag=lag,
            treatment_mode=treatment_mode,
            coeff=coeff,
            type_static_feat=self.type_static_feat,
        )

        self.test_cf_one_step = ARSimulationDataset(
            d_vitals=d_vitals,
            d_random_effects=d_random_effects,
            treat_imb=treat_imb,
            p=p,
            cov_rho=cov_rho,
            cov_var=cov_var,
            re_centers=re_centers,
            re_cluster_std=re_cluster_std,
            num_patients=num_patients["test"],
            window_size=window_size,
            seq_length=seq_length,
            subset_name="test",
            mode="counterfactual_one_step",
            seed=seed,
            lag=lag,
            treatment_mode=treatment_mode,
            num_treatments=num_treatments,
            coeff=coeff,
            type_static_feat=self.type_static_feat,
        )

        self.test_cf_treatment_seq = ARSimulationDataset(
            d_vitals=d_vitals,
            d_random_effects=d_random_effects,
            treat_imb=treat_imb,
            p=p,
            cov_rho=cov_rho,
            cov_var=cov_var,
            re_centers=re_centers,
            re_cluster_std=re_cluster_std,
            num_patients=num_patients["test"],
            window_size=window_size,
            seq_length=seq_length,
            subset_name="test",
            mode="counterfactual_treatment_seq",
            projection_horizon=projection_horizon,
            seed=seed,
            lag=lag,
            cf_seq_mode=cf_seq_mode,
            treatment_mode=treatment_mode,
            coeff=coeff,
            type_static_feat=self.type_static_feat,
        )

        self.projection_horizon = projection_horizon
        self.autoregressive = True
        self.has_vitals = True
        self.train_scaling_params = self.train_f.get_scaling_params()

        self.max_seq_length = seq_length

    def save_data_in_crn_format(self, savepath):
        pickle_map = {
            "num_time_steps": self.max_seq_length,
            "training_data": self.train_f.data,
            "validation_data": self.val_f.data,
            "test_data": self.test_cf_one_step.data,
            "scaling_data": self.train_scaling_params,
        }
        logger.info("%s", f"Saving pickle map to {savepath}")

    def save_to_pkl(self, savepath):
        logger.info("%s", f"Saving dataset collection to {savepath}")
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)
        filepath = os.path.join(savepath, "dataset_collection.pt")
        with gzip.open(filepath, "wb") as f:
            pkl.dump(self, f)
        finish_flag = os.path.join(savepath, "finished.txt")
        with open(finish_flag, "w") as f:
            f.write("finished")
