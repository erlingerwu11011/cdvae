import logging

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import omegaconf

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel

# Aliases
DictConfig = omegaconf.DictConfig
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
tqdm = tqdm.tqdm

logger = logging.getLogger(__name__)


class RepEncoder(TimeVaryingCausalModel):
    """Representation encoder."""

    model_type = "rep_encoder"
    possible_model_types = {"rep_encoder"}

    def __init__(
        self,
        args,
        dataset_collection,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        **kwargs,
    ):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        # Compute input size based on features present
        self.input_size = self.dim_treatments + self.dim_static_features
        if self.has_vitals:
            self.input_size += self.dim_vitals
        if self.autoregressive:
            self.input_size += self.dim_outcome

        self.alpha = None
        self.update_alpha = None

        self._init_specific(args.model.rep_encoder)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args):
        """Initialize model-specific parameters. Must be implemented by subclasses."""
        raise NotImplementedError()

    def prepare_data(self):
        """Process data if it hasn't been processed yet."""
        if self.dataset_collection and not self.dataset_collection.process_data_rep_est:
            self.dataset_collection.process_data_rep_est()

    def configure_optimizers(self):
        """Configure optimizers. Must be implemented by subclasses."""
        raise NotImplementedError()

    def training_step(self, batch, batch_ind):
        """Perform a single training step."""
        loss = self(batch)
        return loss

    def _eval_step(self, batch, batch_ind, subset_name):
        """Generic evaluation step."""
        loss = self.training_step(batch, batch_ind)
        self.log(
            f"{self.model_type}_{subset_name}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        # Log additional validation metric if evaluating validation subset
        if subset_name == self.val_dataloader().dataset.subset_name:
            self.log(f"{self.model_type}-val_metric", loss)

    def validation_step(self, batch, batch_ind, **kwargs):
        """Validation step wrapper."""
        subset_name = self.val_dataloader().dataset.subset_name
        self._eval_step(batch, batch_ind, subset_name)

    def test_step(self, batch, batch_ind, **kwargs):
        """Test step wrapper."""
        subset_name = self.test_dataloader().dataset.subset_name
        self._eval_step(batch, batch_ind, subset_name)

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        """Predict step: return encoded representation."""
        return self.encode(batch)

    def get_representations(self, dataset):
        """Compute and return representations for the given dataset."""
        # If dataset is not already a DataLoader, create one
        if not isinstance(dataset, DataLoader):
            logger.info("Collecting representations for %s.", dataset.subset_name)
            data_loader = DataLoader(
                dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
            )
        else:
            data_loader = dataset

        self.eval()
        representations = []
        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader), desc="rep"):
                batch_rep = self.encode(batch)
                representations.append(batch_rep.detach().cpu())

        # Concatenate and convert to numpy array
        reps = torch.cat(representations, dim=0).numpy()
        return reps

class EstHeadAutoreg(TimeVaryingCausalModel):
    """Estimation head for autoregressive modeling."""

    model_type = "est_head"
    possible_model_types = {"est_head"}

    def __init__(
        self,
        args,
        rep_encoder,
        dataset_collection,
        autoregressive=None,
        has_vitals=None,
        bce_weights=None,
        prefix="",
        init_spec=True,
        **kwargs,
    ):
        """
        Initialize the Estimation Head.

        Args:
            args: Experiment/configuration arguments.
            rep_encoder: A representation encoder instance.
            dataset_collection: Collection of datasets.
            autoregressive: Flag for autoregressive modeling.
            has_vitals: Flag indicating whether vital features are present.
            bce_weights: Optional weights for BCE loss.
            prefix: Prefix for logging.
            init_spec: Whether to initialize specific parameters.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.rep_encoder = rep_encoder
        self.projection_horizon = args.dataset.projection_horizon
        self.output_horizon = self.projection_horizon + 1
        self.prefix = prefix
        self.alpha = None
        self.update_alpha = None

        if init_spec:
            self._init_specific(args.model.est_head)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args):
        """
        Initialize model-specific parameters.

        To be implemented by subclasses if needed.
        """
        pass

    def prepare_data(self):
        """
        Prepare data for the representation estimator.

        Processes the data if it has not already been processed.
        """
        if self.dataset_collection and not self.dataset_collection.process_data_rep_est:
            self.dataset_collection.process_data_rep_est()

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate schedulers.

        Returns:
            Optimizer or optimizer with LR schedulers.
        """
        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]["optimizer"]["lr_scheduler"]:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    def _unroll_horizon(self, x, horizon):
        """
        Unroll a time series tensor over a given horizon.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].
            horizon (int): Number of steps to unroll.
        Returns:
            torch.Tensor: Tensor of shape [B, T, horizon, D].
        """
        unrolled = []
        total_len = x.shape[1]
        for h in range(horizon):
            unrolled_h = x[:, h:, :]
            # Pad to maintain original time dimension if necessary
            if unrolled_h.shape[1] < total_len:
                pad_num = total_len - unrolled_h.shape[1]
                pad_tensor = torch.zeros(unrolled_h.shape[0], pad_num, unrolled_h.shape[2],
                                         dtype=unrolled_h.dtype, device=unrolled_h.device)
                unrolled_h = torch.cat([unrolled_h, pad_tensor], dim=1)
            unrolled.append(unrolled_h)
        return torch.stack(unrolled, dim=2)

    def _calc_mse_loss(self, outcome_pred, outputs, active_entries):
        """
        Calculate the mean squared error (MSE) loss.

        Args:
            outcome_pred (torch.Tensor): Predicted outcomes.
            outputs (torch.Tensor): Ground truth outputs.
            active_entries (torch.Tensor): Mask of active entries.
        Returns:
            torch.Tensor: The computed MSE loss.
        """
        unrolled_outputs = outputs.unsqueeze(2)
        unrolled_active_entries = active_entries.unsqueeze(2)
        mse_loss = F.mse_loss(outcome_pred, unrolled_outputs, reduce=False)
        mse_loss = (unrolled_active_entries * mse_loss).sum() / unrolled_active_entries.sum()
        return mse_loss

    def training_step(self, batch, batch_ind):
        """
        Perform a single training step.

        Args:
            batch: A batch of training data.
            batch_ind: Index of the current batch.
        Returns:
            torch.Tensor: Loss value.
        """
        outcome_pred = self(batch, one_step=True)
        mse_loss = self._calc_mse_loss(outcome_pred, batch["outputs"], batch["active_entries"])
        loss = mse_loss

        self.log(
            f"{self.model_type}_train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            f"{self.model_type}_train_mse_loss",
            mse_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def _eval_step(self, batch, batch_ind, subset_name):
        """
        Generic evaluation step.

        Args:
            batch: A batch of evaluation data.
            batch_ind: Index of the current batch.
            subset_name (str): Name of the current subset (e.g., validation or test).
        """
        outcome_pred = self(batch, one_step=True)
        mse_loss = self._calc_mse_loss(outcome_pred, batch["outputs"], batch["active_entries"])
        loss = mse_loss

        self.log(
            f"{self.model_type}_{subset_name}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            f"{self.model_type}_{subset_name}_mse_loss",
            mse_loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        if subset_name == self.val_dataloader().dataset.subset_name:
            self.log(
                f"{self.prefix}-{self.model_type}-val_metric",
                mse_loss,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_ind, **kwargs):
        """
        Perform a validation step.
        """
        subset_name = self.val_dataloader().dataset.subset_name
        self._eval_step(batch, batch_ind, subset_name)

    def test_step(self, batch, batch_ind, **kwargs):
        """
        Perform a test step.
        """
        subset_name = self.test_dataloader().dataset.subset_name
        self._eval_step(batch, batch_ind, subset_name)

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        """
        Perform a prediction step.

        Returns:
            torch.Tensor: Predicted outcomes on CPU.
        """
        outcome_pred = self(batch)
        return outcome_pred.cpu()

    def get_predictions(self, dataset, one_step=False):
        """
        Compute and return model predictions for a given dataset.

        Args:
            dataset: A torch Dataset or DataLoader.
            one_step (bool): If True, use one-step prediction.
        Returns:
            numpy.ndarray: Predictions as a NumPy array.
        """
        logger.info("Predictions for %s.", dataset.subset_name)
        loader = DataLoader(
            dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
        )

        if one_step:
            self.to(self.device)
            self.eval()
            predictions = []
            with torch.no_grad():
                for batch in tqdm.tqdm(loader, total=len(loader), desc="1-step pred"):
                    # Move all batch elements to the device
                    for k in batch:
                        batch[k] = batch[k].to(self.device)
                    pred = self(batch, one_step=True).detach().cpu()
                    predictions.append(pred)
            outcome_pred = torch.cat(predictions)
        else:
            outcome_pred = torch.cat(self.trainer.predict(self, loader))

        return outcome_pred.cpu().numpy()

    def get_representations(self, dataset):
        """
        Collect and return latent representations from a given dataset.

        If the dataset is not a DataLoader, it will be wrapped into one.

        Args:
            dataset (torch.utils.data.Dataset or DataLoader): The dataset from which to extract representations.
        Returns:
            numpy.ndarray: Concatenated latent representations.
        """
        if not isinstance(dataset, DataLoader):
            logger.info("Collecting representations for %s.", dataset.subset_name)
            loader = DataLoader(
                dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False
            )
        else:
            loader = dataset

        self.to(self.device)
        self.eval()
        representations = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, total=len(loader), desc="rep"):
                # Expecting the model to return (outcome_pred, rep)
                _, rep = self(batch, return_rep=True)
                representations.append(rep.detach().cpu())

        reps = torch.cat(representations, dim=0).numpy()
        return reps

    def get_autoregressive_predictions(self, dataset, one_step=False):
        """
        Get autoregressive predictions for the dataset.

        Args:
            dataset: Input dataset.
            one_step (bool): If True, use one-step prediction.
        Returns:
            numpy.ndarray: Autoregressive predictions.
        """
        logger.info("Autoregressive Prediction for %s.", dataset.subset_name)
        return self.get_predictions(dataset, one_step=one_step)

    def get_normalised_1_step_rmse_syn(self, dataset, datasets_mc=None, prefix=None):
        """
        Compute normalized 1-step RMSE for synthetic counterfactual predictions.

        Args:
            dataset: The primary dataset.
            datasets_mc: Optional counterfactual dataset.
            prefix: Optional prefix for logging.
        Returns:
            float: Normalized RMSE value.
        """
        logger.info("RMSE calculation for %s, 1-step counterfactual.", dataset.subset_name)

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        outputs_scaled = self.get_autoregressive_predictions(
            dataset if datasets_mc is None else datasets_mc, one_step=True
        )
        outputs_scaled = outputs_scaled[:, :, 0, :]

        if unscale:
            output_stds, output_means = (
                dataset.scaling_params["output_stds"],
                dataset.scaling_params["output_means"],
            )
            outputs_unscaled = outputs_scaled * output_stds + output_means
            real_unscaled_outputs = dataset.data["outputs"] * output_stds + output_means
            outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
        else:
            outputs_calc, real_outputs_calc = outputs_scaled, dataset.data["outputs"]

        data_to_save = {}
        if dataset.subset_name == "test":
            data_to_save.update(
                {
                    "means": outputs_calc,
                    "output": real_outputs_calc,
                    "active_entries": dataset.data["active_entries"],
                }
            )

        # Compute last active entry differences
        num_samples, _, output_dim = dataset.data["active_entries"].shape
        last_entries = dataset.data["active_entries"] - np.concatenate(
            [
                dataset.data["active_entries"][:, 1:, :],
                np.zeros((num_samples, 1, output_dim)),
            ],
            axis=1,
        )
        mse_last = (((outputs_calc - real_outputs_calc) ** 2) * last_entries).sum() / last_entries.sum()
        rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

        if percentage:
            rmse_normalised_last *= 100.0

        return rmse_normalised_last

    def get_normalised_n_step_rmses_syn(self, dataset, datasets_mc=None, prefix=None):
        """
        Compute normalized n-step RMSE for synthetic counterfactual predictions.

        Args:
            dataset: The primary dataset.
            datasets_mc: Optional counterfactual dataset.
            prefix: Optional prefix for logging.
        Returns:
            numpy.ndarray: Normalized RMSE values for each time step.
        """
        logger.info("RMSE calculation for %s, n-step counterfactual.", dataset.subset_name)

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        outputs_scaled = self.get_autoregressive_predictions(
            dataset if datasets_mc is None else datasets_mc
        )

        if unscale:
            output_stds, output_means = (
                dataset.scaling_params["output_stds"],
                dataset.scaling_params["output_means"],
            )
            outputs_unscaled = outputs_scaled * output_stds + output_means
            real_unscaled_outputs = dataset.data["outputs"] * output_stds + output_means
            outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
        else:
            outputs_calc, real_outputs_calc = outputs_scaled, dataset.data["outputs"]

        data_to_save = {}
        if dataset.subset_name == "test":
            data_to_save.update(
                {
                    "means": outputs_calc,
                    "output": real_outputs_calc,
                    "active_entries": dataset.data["active_entries"],
                }
            )

        # Select the relevant sequence positions based on factual sequence lengths
        factual_seq_lengths = dataset.data["sequence_lengths"].astype(int) - self.projection_horizon - 1
        outputs_calc = outputs_calc[np.arange(len(outputs_calc)), factual_seq_lengths]
        outputs_calc = outputs_calc[:, 1:]

        real_outputs_calc_ms = []
        active_entries_ms = []
        for i in range(len(real_outputs_calc)):
            start = factual_seq_lengths[i] + 1
            end = start + self.projection_horizon
            real_outputs_calc_ms.append(real_outputs_calc[i, start:end])
            active_entries_ms.append(dataset.data["active_entries"][i, start:end])
        real_outputs_calc_ms = np.stack(real_outputs_calc_ms, axis=0)  # [B, H, D]
        active_entries_ms = np.stack(active_entries_ms, axis=0)

        mse_last = (((outputs_calc - real_outputs_calc_ms) ** 2) * active_entries_ms).sum(axis=-1).sum(axis=0) / active_entries_ms.sum(axis=-1).sum(axis=0)
        rmse_normalised_last = np.sqrt(mse_last) / dataset.norm_const

        if percentage:
            rmse_normalised_last *= 100.0

        return rmse_normalised_last

    def get_normalised_n_step_rmses_real(self, dataset, datasets_mc=None, prefix=None):
        """
        Compute normalized n-step RMSE for real counterfactual predictions.

        Args:
            dataset: The primary dataset.
            datasets_mc: Optional counterfactual dataset.
            prefix: Optional prefix for logging.
        Returns:
            numpy.ndarray: Normalized RMSE values for each time step.
        """
        logger.info("RMSE calculation for %s, n-step counterfactual.", dataset.subset_name)

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        outputs_scaled = self.get_autoregressive_predictions(
            dataset if datasets_mc is None else datasets_mc
        )

        if unscale:
            output_stds, output_means = (
                dataset.scaling_params["output_stds"],
                dataset.scaling_params["output_means"],
            )
            outputs_unscaled = outputs_scaled * output_stds + output_means
            real_unscaled_outputs = dataset.data["outputs"] * output_stds + output_means
            outputs_calc, real_outputs_calc = outputs_unscaled, real_unscaled_outputs
        else:
            outputs_calc, real_outputs_calc = outputs_scaled, dataset.data["outputs"]

        data_to_save = {}
        if dataset.subset_name == "test":
            data_to_save.update(
                {
                    "means": outputs_calc,
                    "output": real_outputs_calc,
                    "active_entries": dataset.data["active_entries"],
                }
            )

        horizon_rmses = []
        # Compute RMSE for each time step (horizon)
        for horizon in range(self.output_horizon):
            outputs_calc_h = outputs_calc[:, :, horizon]
            real_outputs_calc_h = real_outputs_calc[:, horizon:, :]
            active_entries_h = dataset.data["active_entries"][:, horizon:, :]
            # Pad if necessary to match the dimensions
            if real_outputs_calc_h.shape[1] < outputs_calc_h.shape[1]:
                pad_num = outputs_calc_h.shape[1] - real_outputs_calc_h.shape[1]
                real_outputs_calc_h = np.concatenate(
                    [
                        real_outputs_calc_h,
                        np.zeros(
                            (real_outputs_calc_h.shape[0], pad_num, real_outputs_calc_h.shape[2]),
                            dtype=real_outputs_calc_h.dtype,
                        ),
                    ],
                    axis=1,
                )
                active_entries_h = np.concatenate(
                    [
                        active_entries_h,
                        np.zeros(
                            (active_entries_h.shape[0], pad_num, active_entries_h.shape[2]),
                            dtype=active_entries_h.dtype,
                        ),
                    ],
                    axis=1,
                )
            mse = (((outputs_calc_h - real_outputs_calc_h) ** 2) * active_entries_h).sum() / active_entries_h.sum()
            rmse_normalised = np.sqrt(mse) / dataset.norm_const
            horizon_rmses.append(rmse_normalised)

        horizon_rmses = np.array(horizon_rmses)
        if percentage:
            horizon_rmses *= 100.0

        return horizon_rmses
