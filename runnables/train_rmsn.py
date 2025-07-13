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

    assert cfg.dataset.treatment_mode == "multilabel"  # Only binary multilabel regime possible

    metrics_seeds = {}

    seeds = cfg.exp.seed
    for seed in seeds:
        results = {}
        seed_everything(seed)
        cfg.exp.seed = seed

        if cfg.dataset.type_static_feat == "rep":
            instantiate_components, cfg, cfg_cdvae = get_substitutes_static_feat(
                cfg, cfg_cdvae, process_type_model="encoder"
            )
            rmsn_dataset_collection = instantiate_components.dataset_collection
        else:
            log.info("No Running of CDVAE")

            instantiate_components = ComponentInstantiator(cfg, process_type="encoder")
            instantiate_components.instantiate_all()
            rmsn_dataset_collection = instantiate_components.dataset_collection

        propensity_treatment = instantiate(
            cfg.model.propensity_treatment, cfg, rmsn_dataset_collection, _recursive_=False
        )
        if cfg.model.propensity_treatment.tune_hparams:
            propensity_treatment.finetune(
                resources_per_trial=cfg.model.propensity_treatment.resources_per_trial
            )

        propensity_treatment_trainer = instantiate_components.trainer
        propensity_treatment_trainer.fit(propensity_treatment)

        val_bce_orig, val_bce_all = propensity_treatment.get_masked_bce(
            rmsn_dataset_collection.val_f
        )
        log.info(
            f"Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig): {val_bce_orig}"
        )
        if hasattr(rmsn_dataset_collection, "test_cf_one_step"): 
            test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(
                rmsn_dataset_collection.test_cf_one_step
            )
        elif hasattr(rmsn_dataset_collection, "test_f"):
            test_bce_orig, test_bce_all = propensity_treatment.get_masked_bce(
                rmsn_dataset_collection.test_f
            )

        log.info(
            f"Test normalised RMSE (all): {test_bce_orig}; Test normalised RMSE (orig): {test_bce_all}."
        )

        propensity_history = instantiate(
            cfg.model.propensity_history, cfg, rmsn_dataset_collection, _recursive_=False
        )
        if cfg.model.propensity_history.tune_hparams:
            propensity_history.finetune(
                resources_per_trial=cfg.model.propensity_history.resources_per_trial
            )
        instantiate_components.instantiate_callbacks()
        instantiate_components.instantiate_loggers()
        instantiate_components.instantiate_trainer()

        propensity_history_trainer = instantiate_components.trainer
        propensity_history_trainer.fit(propensity_history)

        # Validation BCE
        val_bce_orig, val_bce_all = propensity_history.get_masked_bce(
            rmsn_dataset_collection.val_f
        )
        log.info(
            f"Val normalised BCE (all): {val_bce_all}; Val normalised RMSE (orig): {val_bce_orig}"
        )

        # Test BCE
        if hasattr(rmsn_dataset_collection, "test_cf_one_step"):  # Test one_step_counterfactual
            test_bce_orig, test_bce_all = propensity_history.get_masked_bce(
                rmsn_dataset_collection.test_cf_one_step
            )
        elif hasattr(rmsn_dataset_collection, "test_f"):  # Test factual
            test_bce_orig, test_bce_all = propensity_history.get_masked_bce(
                rmsn_dataset_collection.test_f
            )

        log.info(
            f"Test normalised BCE (all): {test_bce_orig}; Test normalised BCE (orig): {test_bce_all}."
        )

        # ============================== Initialisation & Training of Encoder ==============================
        encoder = instantiate(
            cfg.model.encoder,
            cfg,
            propensity_treatment,
            propensity_history,
            rmsn_dataset_collection,
            _recursive_=False,
        )
        if cfg.model.encoder.tune_hparams:
            encoder.finetune(resources_per_trial=cfg.model.encoder.resources_per_trial)

        instantiate_components.instantiate_callbacks()
        instantiate_components.instantiate_loggers()
        instantiate_components.instantiate_trainer()
        instantiate_components.instantiate_val_dataloader()

        encoder_trainer = instantiate_components.trainer
        encoder_trainer.fit(encoder)

        encoder_trainer.test(encoder, test_dataloaders=instantiate_components.val_dataloader)
        val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(
            rmsn_dataset_collection.val_f
        )
        log.info(
            f"Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}"
        )

        if hasattr(
            rmsn_dataset_collection, "test_cf_one_step"
        ):  
            test_pehe = encoder.get_pehe_one_step(rmsn_dataset_collection.test_cf_one_step)
            log.info(f"Test pehe_last: {test_pehe}")
            results["test_pehe"] = test_pehe

        elif hasattr(rmsn_dataset_collection, "test_f"):  
            test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(
                rmsn_dataset_collection.test_f
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

    return results, {
        "propensity_treatment": propensity_treatment,
        "propensity_history": propensity_history,
        "encoder": encoder,
    }


@hydra.main(version_base="1.3", config_path="../settings", config_name="train_rmsn")
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
