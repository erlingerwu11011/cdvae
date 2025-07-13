import logging
from typing import Optional

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

from src.runnables.utils_train import (
    analyze_model_output,
    instantiate_components,
    run_preprocessing_pipeline,
    run_queries_pipeline,
    test_model,
    train_model,
    update_model_cfg,
)
from src.utils import extras, get_metric_value, task_wrapper

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)


@task_wrapper
def run_pipeline(cfg: DictConfig) -> Optional[float]:
    """Runs the entire pipeline including training and testing.

    :param cfg: Configuration object.
    :return: Optional[float] with optimized metric value.
    """
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

    components = instantiate_components(cfg)

    run_queries_pipeline(components["queries_pipeline"], components["preprocess_pipeline"])
    run_preprocessing_pipeline(components["preprocess_pipeline"], components["datamodule"])

    update_model_cfg(cfg, components["preprocess_pipeline"])

    log.info(f"Instantiating model <{cfg.model.leap._target_}>")
    components["model"] = hydra.utils.instantiate(cfg.model.leap)  # type: ignore

    train_model(cfg, components["trainer"], components["model"], components["datamodule"])

    analyze_model_output(
        cfg, components["model"], components["preprocess_pipeline"], components["datamodule"]
    )

    test_model(cfg, components["trainer"], components["model"], components["datamodule"])

    metric_value = get_metric_value(
        metric_dict=components["trainer"].callback_metrics, metric_name=cfg.get("optimized_metric")
    )

    return metric_value, components


@hydra.main(version_base="1.3", config_path="../settings", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    OmegaConf.set_struct(cfg, False)
    OmegaConf.register_new_resolver("divide", lambda a, b: a / b)
    OmegaConf.to_yaml(cfg, resolve=True)

    extras(cfg)

    return run_pipeline(cfg)


if __name__ == "__main__":
    main()
