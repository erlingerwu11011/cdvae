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
from pytorch_lightning.utilities.seed import seed_everything

from runnables.utils_train import (
    ComponentInstantiator,
    get_results,
    get_substitutes_static_feat,
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

        # ============================== Initialisation & Training of multimodel ==============================
        dataset_collection = instantiate_components.dataset_collection
        multi = instantiate(cfg.model.multi, cfg, dataset_collection, _recursive_=False)
        if cfg.model.multi.tune_hparams:
            multi.finetune(resources_per_trial=cfg.model.multi.resources_per_trial)

        trainer = instantiate_components.trainer
        trainer.fit(multi)

        val_dataloader = instantiate_components.val_dataloader
        trainer.validate(multi, dataloaders=val_dataloader)

        results = get_results(multi, dataset_collection)

        if cfg.dataset.name == "ar_sim_generator":
            val_dataloader = instantiate_components.val_dataloader
            quality_rep = inspect_rep_quality(val_dataloader, multi, cfg)
            results["Mean MI"] = quality_rep["Mean MI"]

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

    return metrics_seeds, {"multi": multi}


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_multi")
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
