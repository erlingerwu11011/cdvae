import logging
from typing import Optional

import hydra
import numpy as np
import rootutils
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from runnables.utils_train import (
    ComponentInstantiator,
    seed_everything,
)
from src.utils import extras, task_wrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)
torch.autograd.set_detect_anomaly(True)

OmegaConf.register_new_resolver("divide", lambda a, b: a / b)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)


@task_wrapper
def run_pipeline(cfg: DictConfig, cfg_cdvae: DictConfig = None) -> Optional[float]:
    """Runs the entire pipeline including training and testing.

    :param cfg: Configuration object.
    :return: Optional[float] with optimized metric value.
    """

    assert cfg.dataset.treatment_mode == "multilabel"  # Only binary multilabel regime possible
    metrics_seeds = {}

    seeds = cfg.exp.seed
    for seed in seeds:
        results = {}
        seed_everything(seed)
        cfg.exp.seed = seed

        instantiate_components = ComponentInstantiator(cfg, process_type="multi")
        instantiate_components.instantiate_all()
        dataset_collection = instantiate_components.dataset_collection

        # ============================== Initialisation & Training of WRep_encoder ==============================
        wrep_encoder = instantiate(
            cfg.model.wrep_encoder, cfg, dataset_collection, _recursive_=False
        )

        if cfg.model.train_wrep_encoder:
            if cfg.model.wrep_encoder.tune_hparams:
                wrep_encoder.finetune(resources_per_trial=cfg.model.multi.resources_per_trial)

            trainer = instantiate_components.trainer
            trainer.fit(wrep_encoder)

            val_dataloader = instantiate_components.val_dataloader
            trainer.validate(wrep_encoder, dataloaders=val_dataloader)

            # ============================== Initialisation & Training of cdvae ==============================
            instantiate_components.instantiate_callbacks()
            instantiate_components.instantiate_loggers()
            instantiate_components.instantiate_trainer()

        cdvae = instantiate(
            cfg.model.cdvae, cfg, wrep_encoder, dataset_collection, _recursive_=False
        )
        if cfg.model.cdvae.tune_hparams:
            cdvae.finetune(resources_per_trial=cfg.model.multi.resources_per_trial)

        trainer = instantiate_components.trainer
        trainer.fit(cdvae)

        val_dataloader = instantiate_components.val_dataloader
        trainer.validate(cdvae, dataloaders=val_dataloader)

        val_rmse_orig, val_rmse_all = cdvae.get_normalised_masked_rmse(dataset_collection.val_f)
        log.info(
            f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}"
        )

        p_values = cdvae.get_predictive_check_p_values(val_dataloader)

        log.info(f" p_values : {p_values}")

        if hasattr(dataset_collection, "test_cf_one_step"):
            test_pehe = cdvae.get_pehe_one_step(dataset_collection.test_cf_one_step)
            log.info(f"Test pehe_last: {test_pehe}")
            results["test_pehe"] = test_pehe
            _, _, test_rmse_cf_one_step = cdvae.get_normalised_masked_rmse(
                dataset_collection.test_cf_one_step, one_step_counterfactual=True
            )
            results["test_rmse_cf_one_step"] = test_rmse_cf_one_step
            log.info(f"Test rmse_cf_one_step: {test_rmse_cf_one_step}")

        elif hasattr(dataset_collection, "test_f"):  
            test_rmse_orig, test_rmse_all = cdvae.get_normalised_masked_rmse(
                dataset_collection.test_f
            )
            log.info(
                f"Test normalised RMSE (all): {test_rmse_all}; "
                f"Test normalised RMSE (orig): {test_rmse_orig}."
            )
            results["test_rmse_all"] = test_rmse_all
            results["test_rmse_orig"] = test_rmse_orig

        with torch.no_grad():
            torch.cuda.empty_cache()

        for key, value in results.items():
            if key not in metrics_seeds:
                metrics_seeds[key] = []
            metrics_seeds[key].append(float(value))

        with open(
            cfg.paths.output_dir + f"/metrics_seeds_{cfg.model.name}.yaml",
            "w",
        ) as f:
            yaml.dump(metrics_seeds, f)

    avg_std_metrics = {}
    for key, value in metrics_seeds.items():
        if key not in avg_std_metrics:
            avg_std_metrics[key] = []
        avg_std_metrics[key] = [float(np.mean(value)), float(np.std(value))]

    with open(
        cfg.paths.output_dir + f"/avg_std_metrics_{cfg.model.name}.yaml",
        "w",
    ) as f:
        yaml.dump(avg_std_metrics, f)

    return metrics_seeds, {"cdvae": cdvae}


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_cdvae.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_yaml(cfg, resolve=True)

    extras(cfg)

    return run_pipeline(cfg, None)


if __name__ == "__main__":
    main()
