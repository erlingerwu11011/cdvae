import torch as th
from torch.distributions import Distribution
from typing import List, Dict, Tuple, Callable

from common import topological_sort, invert_graph, Iota


class TMSCM():

    def __init__(
        self,
        dependencies: Dict[str, List[str]],
        mechanisms: Dict[str, Callable],
        exogenous_distributions: Dict[str, Distribution],
        endo_exo_index_pairs: List[Tuple[str, str]],
    ):
        """A (Markovian) TM-SCM to generate ground-truth data.

        Args:
            `dependencies` (Dict[str, List[str]]): Dependency graph of endogenous variables.
            `mechanisms` (Dict[str, Callable]): Causal mechanisms of endogenous variables.
            `exogenous_distributions` (Dict[str, Distribution]): Distributions of exogenous variables.
            `endo_exo_index_pairs` (List[Tuple[str, str]]): Pairs of endogenous and exogenous variables.
        """
        self.dependencies = dependencies
        self.graph = invert_graph(self.dependencies)
        self.mechanisms = mechanisms
        self.exogenous_distributions = exogenous_distributions
        self.exogenous_indexes = {
            endo_index: exo_index
            for endo_index, exo_index in endo_exo_index_pairs
        }
        self.endogenous_indexes = {
            exo_index: endo_index
            for endo_index, exo_index in endo_exo_index_pairs
        }

        self.topological_order = topological_sort(graph=self.graph)

    def pushforward(
        self,
        exogenous_values: Dict[str, th.Tensor],
        intervention_values: Dict[str, th.Tensor] = None,
        intervention_masks: Dict[str, th.Tensor] = None,
    ) -> Dict[str, th.Tensor]:
        """Solving endogenous variables by exogenous values and interventions

        Args:
            `exogenous_values` (Dict[str, th.Tensor]): Exogenous values.
            `intervention_values` (Dict[str, th.Tensor]): Intervened values.
            `intervention_masks` (Dict[str, th.Tensor]): Intervention mask, indicating whether a endogenous variable is intervened.

        Returns:
            Dict (Dict[str, th.Tensor]): Endogenous values.
        """
        endogenous_values = {
            self.endogenous_indexes[exo_index]: th.zeros_like(exo_value)
            for exo_index, exo_value in exogenous_values.items()
        }

        for endo_index in self.topological_order:
            exo_index = self.exogenous_indexes[endo_index]
            kwargs = {
                pa_index: endogenous_values[pa_index].clone()
                for pa_index in self.dependencies[endo_index]
            } | {
                exo_index: exogenous_values[exo_index].clone()
            }

            f = self.mechanisms[endo_index]
            endogenous_values[endo_index] = f(**kwargs)

            if intervention_masks is None or intervention_values is None or endo_index not in intervention_masks:
                continue
            intervened = intervention_masks[endo_index]
            intervention = intervention_values[endo_index]
            endogenous_values[endo_index][intervened] = intervention[
                intervened].clone().detach()

        return endogenous_values

    def sample_exogenous(
        self,
        n_samples: int,
    ) -> Dict[str, th.Tensor]:
        """Sampling exogenous values from exogenous distribution

        Args:
            `n_samples` (int): Number of samples.

        Returns:
            Dict (Dict[str, th.Tensor]): Exogenous values.
        """
        return {
            exo_index: p.sample((n_samples, ))
            for exo_index, p in self.exogenous_distributions.items()
        }

    def sample_endogenous(
        self,
        n_samples: int,
        intervention_values: Dict[str, th.Tensor] = None,
        intervention_masks: Dict[str, th.Tensor] = None,
    ) -> Dict[str, th.Tensor]:
        """Sampling endogenous values from endogenous distribution

        Args:
            `n_samples` (int): Number of samples.
            `intervention_values` (Dict[str, th.Tensor]): Intervened values.
            `intervention_masks` (Dict[str, th.Tensor]): Intervention mask, indicating whether a endogenous variable is intervened.

        Returns:
            Dict (Dict[str, th.Tensor]): Endogenous values.
        """
        exogenous_values = self.sample_exogenous(n_samples=n_samples)
        endogenous_values = self.pushforward(
            exogenous_values=exogenous_values,
            intervention_values=intervention_values,
            intervention_masks=intervention_masks,
        )
        return endogenous_values

    @property
    def iota(self) -> Iota:
        """Iota inferred from TM-SCM"""
        exogenous_values = self.sample_exogenous(1)
        return Iota(
            dimensions={
                self.endogenous_indexes[exo_index]: exogenous_value.size(-1)
                for exo_index, exogenous_value in exogenous_values.items()
            },
            order=self.topological_order,
        )

    @staticmethod
    def check_tmscm(scm: "TMSCM", n_samples: int = 64):
        """Check if an instantiated TM-SCM object is indeed a TM-SCM.

        Args:
            `scm`: An instantiated TM-SCM object.
            `n_samples`: Number of samples used to check parametric assumptions.

        Returns:
            bool: Yes or Asserted
        """
        ex = scm.sample_exogenous(n_samples=n_samples)
        ex = th.concat(
            [ex[scm.exogenous_indexes[i]] for i in scm.iota.order],
            dim=-1,
        )

        def f(x):
            u = {}
            for i in scm.iota.order:
                u[scm.exogenous_indexes[i]] = x[scm.iota.i(i=i)]
            v = scm.pushforward(u)
            return th.concat(
                [v[i] for i in scm.iota.order],
                dim=-1,
            )

        flags = None
        for i in range(ex.size(0)):
            x = ex[i].requires_grad_(True)
            j = th.autograd.functional.jacobian(f, x)
            if not th.all(j.triu(1) == 0):
                assert False, "Assert: SCM is not TM-SCM, since it is not triagular"
            if i == 0:
                flags = th.diag(j) > 0
            elif not th.all((th.diag(j) > 0) == flags):
                assert False, "Assert: SCM is not TM-SCM, since it is not consistently monotonic"

        return True
