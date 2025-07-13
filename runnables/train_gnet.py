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
    """
    Training / evaluation script for G-Net
    Args:
        cfg: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    metrics_seeds = {}

    seeds = cfg.exp.seed
    for seed in seeds:
        results = {}
        seed_everything(seed)
        cfg.exp.seed = seed

        if cfg.dataset.type_static_feat == "rep":
            instantiate_components, cfg, cfg_cdvae = get_substitutes_static_feat(
                cfg, cfg_cdvae, process_type_model="multi"
            )
            dataset_collection = instantiate_components.dataset_collection
        else:
            log.info("No Running of CDVAE")
            instantiate_components = ComponentInstantiator(cfg, process_type="multi")
            instantiate_components.instantiate_all()
            dataset_collection = instantiate_components.dataset_collection

        cfg.model.g_net.comp_sizes = [
            (cfg.model.dim_outcomes + cfg.model.dim_vitals) // cfg.model.g_net.num_comp
        ] * cfg.model.g_net.num_comp

        # ============================== Initialisation & Training of G-Net ==============================
        model = instantiate(cfg.model.g_net, cfg, dataset_collection, _recursive_=False)
        if cfg.model.g_net.tune_hparams:
            model.finetune(resources_per_trial=cfg.model.g_net.resources_per_trial)

        trainer = instantiate_components.trainer
        trainer.fit(model)

        val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(
            instantiate_components.dataset_collection.val_f
        )
        log.info(
            f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}"
        )

        if cfg.dataset.name == "ar_sim_generator":
            val_dataloader = instantiate_components.val_dataloader
            quality_rep = inspect_rep_quality(val_dataloader, model, cfg)
            results["Mean MI"] = quality_rep["Mean MI"]

        if hasattr(dataset_collection, "test_cf_one_step"):  # Test one_step_counterfactual rmse
            test_pehe = model.get_pehe_one_step(dataset_collection.test_cf_one_step)
            log.info(f"Test pehe_last: {test_pehe}")
            results["test_pehe"] = test_pehe

        elif hasattr(dataset_collection, "test_f"):  # Test factual rmse
            test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(
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

    return metrics_seeds, {"gnet": model}


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_gnet")
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

    cfg_cdvae.exp.seed = cfg.exp.seed

    extras(cfg)

    return run_pipeline(cfg, cfg_cdvae)


if __name__ == "__main__":
    main()
