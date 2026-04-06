import torch as th
from lightning import LightningModule
from torch.distributions import Distribution
from typing import Dict, Any
from zuko.distributions import NormalizingFlow
from zuko.lazy import LazyDistribution, LazyTransform

from common import Iota
from dataset import metric
from model.exogenous_distribution import exogenous_distribution
from model.solution_mapping import solution_mapping
from model.optimizer import optimizer


class ProxySCM(LazyDistribution):

    def __init__(
        self,
        exogenous_distribution: LazyDistribution,
        solution_mapping: LazyTransform,
    ):
        """Initialize proxy SCM.

        Args:
            `exogenous_distribution` (LazyDistribution): Exogenous distribution of the proxy SCM.
            `solution_mapping` (LazyTransform): Solution mapping of the proxy SCM.
        """
        super().__init__()
        self.exogenous_distribution = exogenous_distribution
        self.solution_mapping = solution_mapping
        self.causal_order = solution_mapping.causal_order

    def forward(self) -> Distribution:
        base = self.exogenous_distribution()
        transform = self.solution_mapping()
        return NormalizingFlow(transform, base)

    def counterfactual_outcome(
        self,
        observation: th.Tensor,
        intervention: th.Tensor,
        intervention_mask: th.BoolTensor,
    ):
        """Infer counterfactual outcomes using `Pseudo Potential Respons` algorithm.

        Args:
            `observation` (Tensor): Observational endogenous values.
            `intervention` (Tensor): Intervened endogenous values.
            `intervention_mask` (BoolTensor):  Intervention mask, indicating whether a endogenous variable is intervened.

        Returns:
            Tensor: The counterfactual outcome.
        """
        # Get solution mapping :math:`\gamma` and its inverse
        pv = self()
        gamma_inv = pv.transform
        gamma = pv.transform.inv

        # Sample initial exogenous noise
        exogenous = gamma_inv(observation).detach()
        # Set initial couterfactual outcome
        counterfactual = observation.clone()

        # Iterate over each dimension and update
        for i in range(self.causal_order.size(0)):
            # Target index
            j = self.causal_order[i]
            # Target prefix and prefix
            pr, su = self.causal_order[:i + 1], self.causal_order[i:]
            # Set invervened value
            counterfactual_ = counterfactual.clone()
            mask_i = intervention_mask[:, j]
            counterfactual_[:, j][mask_i] = intervention[:, j][mask_i]
            # Find inverse exogenous noise to fully explain first i dimensions
            exogenous_ = gamma_inv(counterfactual_).detach()
            # Keep other dimensions the same as previous
            exogenous[:, pr] = exogenous_[:, pr]
            # Update counterfactual outcome
            counterfactual[:, su] = gamma(exogenous)[:, su].detach()

        return counterfactual


class LightningProxySCM(LightningModule):

    def __init__(
        self,
        iota: Iota,
        exogenous_distribution_kwargs: Dict[str, Any],
        solution_mapping_kwargs: Dict[str, Any],
        optimizer_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()

        # Build proxy SCM
        self.proxy_scm = ProxySCM(
            exogenous_distribution=exogenous_distribution(
                iota=iota,
                exogenous_distribution_kwargs=exogenous_distribution_kwargs,
            ),
            solution_mapping=solution_mapping(
                iota=iota,
                solution_mapping_kwargs=solution_mapping_kwargs,
            ),
        )

        # Configure optimizer
        self.optimization = optimizer(
            model=self,
            optimizer_kwargs=optimizer_kwargs,
        )

    def forward(self) -> Distribution:
        return self.proxy_scm()

    def configure_optimizers(self):
        return self.optimization

    def training_step(self, batch, batch_idx):
        # Train with NLL loss
        v = batch['observation']
        pv = self.proxy_scm()
        nll = metric.safe_mean(-pv.log_prob(v))
        self.log('nll', nll, on_step=True, prog_bar=True)
        return nll

    def validation_step(self, batch, batch_idx):
        # Validate observational distribution
        v = batch['observation']
        pv = self.proxy_scm()
        v_hat = pv.sample((v.size(0), )).detach()

        # Validate counterfactual outcomes
        i, im = batch['intervention'], batch['intervention_mask']
        o = batch['counterfactual_outcome']
        o_hat = self.proxy_scm.counterfactual_outcome(v, i, im).detach()

        # Calculate and record metrics
        log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=True)

        obs_mmd = metric.safe_metric(metric.mmd, v, v_hat, dim=-1)
        self.log('obs_mmd', obs_mmd, **log_kwargs)

        obs_wd = metric.safe_metric(metric.wd, v, v_hat, dim=-1)
        self.log('obs_wd', obs_wd, **log_kwargs)

        rmse = metric.safe_metric(metric.rmse, o, o_hat, dim=-1)
        self.log('ctf_rmse', rmse, **log_kwargs)

        ctf_wd = metric.safe_metric(metric.wd, o, o_hat, dim=-1)
        self.log('ctf_wd', ctf_wd, **log_kwargs)

    def test_step(self, batch, batch_idx):
        # Test is the same as validation
        self.validation_step(batch, batch_idx)
