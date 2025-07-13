import logging
from typing import Optional

import hydra
import numpy as np
import rootutils
import torch
import yaml
from hydra import compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from runnables.utils_train import (
    ComponentInstantiator,
    get_substitutes_static_feat,
    inspect_rep_quality,
    seed_everything,
)
from src.utils import extras, task_wrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

OmegaConf.register_new_resolver("divide", lambda a, b: a / b)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)


@task_wrapper
def run_pipeline(cfg: DictConfig, cfg_cdvae: DictConfig) -> Optional[float]:
    """Runs the entire pipeline including training and testing.

    :param cfg: Configuration object.
    :return: Optional[float] with optimized metric value.
    """

    seeds = cfg.exp.seed
    metrics_seeds = {}
    for seed in seeds:
        results = {}
        seed_everything(seed)
        cfg.exp.seed = seed

        if cfg.dataset.type_static_feat == "rep":
            instantiate_components, cfg, cfg_cdvae = get_substitutes_static_feat(
                cfg, cfg_cdvae, process_type_model="encoder"
            )
            dataset_collection = instantiate_components.dataset_collection

        else:
            log.info("No Running of CDVAE")
            instantiate_components = ComponentInstantiator(cfg, process_type="encoder")
            instantiate_components.instantiate_all()
            dataset_collection = instantiate_components.dataset_collection

        encoder = instantiate(cfg.model.encoder, cfg, dataset_collection, _recursive_=False)
        if cfg.model.encoder.tune_hparams:
            encoder.finetune(resources_per_trial=cfg.model.multi.resources_per_trial)

        trainer_encoder = instantiate_components.trainer
        trainer_encoder.fit(encoder)

        trainer_encoder.validate(encoder, dataloaders=instantiate_components.val_dataloader)

        val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.val_f)
        log.info(
            f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}"
        )

        if hasattr(dataset_collection, "test_cf_one_step"):
            test_pehe = encoder.get_pehe_one_step(
                dataset_collection.test_cf_one_step,
            )
            results["test_pehe"] = test_pehe
            log.info(f"Test pehe_last: {test_pehe}")
            _, _, test_rmse_cf_one_step = encoder.get_normalised_masked_rmse(
                dataset_collection.test_cf_one_step, one_step_counterfactual=True
            )
            results["test_rmse_cf_one_step"] = test_rmse_cf_one_step
            log.info(f"Test rmse_cf_one_step: {test_rmse_cf_one_step}")

        elif hasattr(dataset_collection, "test_f"):  # Test factual rmse
            test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(
                dataset_collection.test_f
            )
            log.info(
                f"Test normalised RMSE (all): {test_rmse_all}; "
                f"Test normalised RMSE (orig): {test_rmse_orig}."
            )

            results["test_rmse_all"] = test_rmse_all
            results["test_rmse_orig"] = test_rmse_orig

        if cfg.dataset.name == "ar_sim_generator":
            val_dataloader = instantiate_components.val_dataloader
            quality_rep = inspect_rep_quality(val_dataloader, encoder, cfg)
            results["Mean MI"] = quality_rep["Mean MI"]

        for key, value in results.items():
            if key not in metrics_seeds:
                metrics_seeds[key] = []
            metrics_seeds[key].append(float(value))

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

    return metrics_seeds, {"encoder": encoder}


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_enc_dec")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    cfg_cdvae = compose(config_name="train_cdvae")

    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_yaml(cfg, resolve=True)

    OmegaConf.set_struct(cfg_cdvae, False)
    OmegaConf.to_yaml(cfg_cdvae, resolve=True)

    extras(cfg)

    return run_pipeline(cfg, cfg_cdvae)


if __name__ == "__main__":
    main()
