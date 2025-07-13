import logging
import random
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from src.utils import instantiate_callbacks, instantiate_loggers

log = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ComponentInstantiator:
    def __init__(self, cfg: DictConfig, process_type: str = "multi"):
        """
        Class to instantiate all components based on the configuration.

        :param cfg: Configuration object.
        :param process_type: Type of processing ("multi", "encoder", "decoder").
        """
        self.cfg = cfg
        self._process_type = process_type
        self._dataset_collection = None
        self.callbacks = None
        self.logger = None
        self._trainer = None
        self._val_dataloader = None
        self._encoder = None
        self._cdvae = None
        self._batch_size = self.cfg.dataset.val_batch_size
        self.dataset_collection = hydra.utils.instantiate(self.cfg.dataset)

    @property
    def process_type(self) -> str:
        """
        Get the process type.

        :return: The current process type.
        """
        return self._process_type

    @property
    def cdvae(self) -> str:
        """
        Get the cdvae.

        :return: The current cdvae.
        """
        return self._cdvae

    @cdvae.setter
    def cdvae(self, value):
        """
        Set the cdvae.

        :param value: The cdvae object.
        """
        self._cdvae = value

    @property
    def val_dataloader(self) -> str:
        """
        Get the val_dataloader.

        :return: The current val_dataloader.
        """
        return self._val_dataloader

    @process_type.setter
    def process_type(self, process_type: str):
        """
        Set the process type and re-instantiate the dataset collection if needed.

        :param process_type: Type of processing ("multi", "encoder", "decoder").
        """
        if process_type not in ["multi", "encoder", "decoder", "rep_est"]:
            raise ValueError(
                "Invalid process type. Must be 'multi', 'encoder', or 'decoder', or 'rep_est'."
            )
        self._process_type = process_type

        self.instantiate_dataset_collection()
        # self.update_model_dimensions()
        self.instantiate_callbacks()
        self.instantiate_loggers()
        self.instantiate_trainer()
        self.instantiate_val_dataloader()

    @property
    def dataset_collection(self):
        """
        Get the current dataset collection.

        :return: The current dataset collection.
        """
        return self._dataset_collection

    @dataset_collection.setter
    def dataset_collection(self, value):
        """
        Set the dataset collection.

        :param value: The dataset collection object.
        """
        self._dataset_collection = value

    @property
    def batch_size(self):
        """
        Get the current batch_size.

        :return: The current batch_size.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """
        Set the dataset collection.

        :param value: The dataset collection object.
        """
        self._batch_size = value

    @property
    def trainer(self):
        return self._trainer

    @property
    def encoder(self) -> str:
        """
        Get the current encoder.

        :return: The current encoder.
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: str):
        """
        Set the encoder to be used when process type is 'decoder'.

        :param encoder: The encoder to be used.
        """
        self._encoder = encoder

    def instantiate_dataset_collection(self):

        log.info(f"Processing dataset_collection")
        if self.process_type == "multi":
            self.dataset_collection.process_data_multi(cdvae=self._cdvae)
        elif self.process_type == "encoder":
            self.dataset_collection.process_data_encoder(cdvae=self._cdvae)
        elif self.process_type == "rep_est":
            self.dataset_collection.process_data_rep_est(cdvae=self._cdvae)
        elif self.process_type == "decoder":
            if not self._encoder:
                raise ValueError("Encoder must be provided when process type is 'decoder'.")
            self.dataset_collection.process_data_decoder(encoder=self._encoder, cdvae=self._cdvae)
            self._batch_size = 10 * self.cfg.dataset.val_batch_size
        else:
            raise ValueError("Unsupported processing type")

    def update_model_dimensions(self):
        log.info(f"Update dim outcomes, treatments, vitals for model")
        self.cfg.model.dim_outcomes = self.dataset_collection.train_f.data["outputs"].shape[-1]
        self.cfg.model.dim_treatments = self.dataset_collection.train_f.data[
            "current_treatments"
        ].shape[-1]
        self.cfg.model.dim_vitals = (
            self.dataset_collection.train_f.data["vitals"].shape[-1]
            if self.dataset_collection.has_vitals
            else 0
        )
        self.cfg.model.dim_vitals = (
            self.cfg.model.dim_vitals - 2
            if self.cfg.exp.test_robustness
            else self.cfg.model.dim_vitals
        )
        print("self.dataset_collection.type_static_feat", self.dataset_collection.type_static_feat)
        self.cfg.model.dim_static_features = (
            self.dataset_collection.train_f.data["static_features"].shape[-1]
            if self.dataset_collection.type_static_feat
            in ["true", "rep"]  # "static_features" in self.dataset_collection.train_f.data.keys()
            else 0
        )

    def instantiate_callbacks(self):
        log.info("Instantiating callbacks...")
        self.callbacks: List[Callback] = instantiate_callbacks(self.cfg.get("callbacks"))

    def instantiate_loggers(self):
        log.info("Instantiating loggers...")
        self.logger: List[Any] = instantiate_loggers(self.cfg.get("logger"))

    def instantiate_trainer(self):
        log.info(f"Instantiating trainer <{self.cfg.trainer._target_}>")
        self._trainer: Trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=self.callbacks, logger=self.logger
        )

    def instantiate_val_dataloader(self):
        self._val_dataloader = DataLoader(
            self.dataset_collection.val_f,
            batch_size=self._batch_size,
            shuffle=False,
        )

    def instantiate_all(self, update_model_dim: bool = True) -> Dict[str, object]:
        """
        Instantiate all components and return them in a dictionary.

        :return: Dictionary containing instantiated components.
        """
        self.instantiate_dataset_collection()
        if update_model_dim:
            self.update_model_dimensions()
        self.instantiate_callbacks()
        self.instantiate_loggers()
        self.instantiate_trainer()
        self.instantiate_val_dataloader()


def get_substitutes_static_feat(cfg: DictConfig, cfg_cdvae: DictConfig, process_type_model: str):

    # Default seed is that of the model backbone in cfg
    cfg_cdvae.exp.seed = cfg.exp.seed
    cfg_cdvae.dataset = cfg.dataset
    cfg_cdvae.trainer = cfg.trainer

    log.info("Running first CDVAE to get a substitute for unobserved static features")
    cfg_cdvae.dataset.type_static_feat = None

    assert (
        cfg_cdvae.dataset.treatment_mode == "multilabel"
    )  # Only binary multilabel is regime possible for CDVAE
    instantiate_components = ComponentInstantiator(cfg_cdvae, process_type="multi")
    instantiate_components.instantiate_all()
    dataset_collection_cdvae = instantiate_components.dataset_collection

    # ============================== Initialisation & Training of WRep_encoder ==============================
    wrep_encoder = instantiate(
        cfg_cdvae.model.wrep_encoder, cfg_cdvae, dataset_collection_cdvae, _recursive_=False
    )

    cdvae = instantiate(
        cfg_cdvae.model.cdvae, cfg_cdvae, wrep_encoder, dataset_collection_cdvae, _recursive_=False
    )
    if cfg_cdvae.model.cdvae.tune_hparams:
        cdvae.finetune(resources_per_trial=cfg_cdvae.model.multi.resources_per_trial)

    trainer_cdvae = instantiate_components.trainer
    trainer_cdvae.fit(cdvae)

    instantiate_components.cdvae = cdvae
    instantiate_components.dataset_collection.type_static_feat = cfg.dataset.type_static_feat
    instantiate_components.dataset_collection.train_f.type_static_feat = (
        cfg.dataset.type_static_feat
    )
    instantiate_components.dataset_collection.val_f.type_static_feat = cfg.dataset.type_static_feat
    instantiate_components.dataset_collection.test_cf_one_step.type_static_feat = (
        cfg.dataset.type_static_feat
    )

    instantiate_components.dataset_collection.train_f.reinitialize_processing()
    instantiate_components.dataset_collection.val_f.reinitialize_processing()
    instantiate_components.dataset_collection.test_cf_one_step.reinitialize_processing()

    instantiate_components.process_type = process_type_model

    dataset_collection = instantiate_components.dataset_collection
    cfg.model.dim_outcomes = dataset_collection.train_f.data["outputs"].shape[-1]
    cfg.model.dim_treatments = dataset_collection.train_f.data["current_treatments"].shape[-1]
    cfg.model.dim_vitals = (
        dataset_collection.train_f.data["vitals"].shape[-1] if dataset_collection.has_vitals else 0
    )
    cfg.model.dim_vitals = (
        cfg.model.dim_vitals - 2 if cfg.exp.test_robustness else cfg.model.dim_vitals
    )
    cfg.model.dim_static_features = dataset_collection.train_f.data["static_features"].shape[-1]

    return instantiate_components, cfg, cfg_cdvae


def get_results(model, dataset_collection):
    results = {}

    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f)
    log.info(
        f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}"
    )
    if hasattr(dataset_collection, "test_cf_one_step"):  
        test_pehe = model.get_pehe_one_step(dataset_collection.test_cf_one_step)
        log.info(f"Test pehe_last: {test_pehe}")
        results["test_pehe"] = test_pehe

    elif hasattr(dataset_collection, "test_f"):  
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
        log.info(
            f"Test normalised RMSE (all): {test_rmse_all}; "
            f"Test normalised RMSE (orig): {test_rmse_orig}."
        )
        results["test_rmse_all"] = test_rmse_all
        results["test_rmse_orig"] = test_rmse_orig

    return results
