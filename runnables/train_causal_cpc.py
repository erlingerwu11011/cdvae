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
    """Training / evaluation script for models with representation-head.

    cfg:
        cfg: arguments of run as DictConfig

    Returns:
        dict with results (one and nultiple-step-ahead RMSEs)
    """

    seeds = cfg.exp.seed
    metrics_seeds = {}
    for seed in seeds:
        results = {}
        seed_everything(seed)
        cfg.exp.seed = seed

        if cfg.dataset.type_static_feat == "rep":
            instantiate_components, cfg, cfg_cdvae = get_substitutes_static_feat(
                cfg, cfg_cdvae, process_type_model="rep_est"
            )
            dataset_collection = instantiate_components.dataset_collection
        else:
            log.info("No Running of CDVAE")
            instantiate_components = ComponentInstantiator(cfg, process_type="rep_est")
            instantiate_components.instantiate_all()
            dataset_collection = instantiate_components.dataset_collection

        rep = instantiate(cfg.model.rep_encoder, cfg, dataset_collection, _recursive_=False)

        if cfg.model.rep_encoder.tune_hparams:
            rep.finetune(resources_per_trial=cfg.model.rep_encoder.resources_per_trial)

        rep_trainer = instantiate_components.trainer
        rep_trainer.fit(rep)

        logging.info("Instantiate est_head ")
        head = instantiate(cfg.model.est_head, cfg, rep, dataset_collection, _recursive_=False)
        instantiate_components.instantiate_callbacks()
        instantiate_components.instantiate_loggers()
        instantiate_components.instantiate_trainer()

        head_trainer = instantiate_components.trainer
        head_trainer.fit(head)

        # test_rmses = {}
        if hasattr(dataset_collection, "test_cf_one_step"):
            test_pehe = head.get_pehe_one_step(
                dataset_collection.test_cf_one_step,
            )
            log.info(f"Test pehe_last: {test_pehe}")
            results["test_pehe"] = test_pehe

        if hasattr(dataset_collection, "test_f"):
            rmses = head.get_normalised_n_step_rmses_real(
                dataset_collection.test_f, prefix=f"test_f"
            )
            for k, v in enumerate(rmses):
                results[f"decoder_test_rmse_{k + 1}-step"] = v

        log.info("%s", f"Metrics : {results}")

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

    return metrics_seeds, {"rep_encoder": rep, "decoder": head}


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_causal_cpc")
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
