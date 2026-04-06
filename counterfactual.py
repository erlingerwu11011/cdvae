import torch as th
from torch.utils.data import StackDataset

from common import interventionally_meaningful_subsets, Iota
from dataset.column_dataset import ColumnDataset
from dataset.tmscm import TMSCM
from dataset.wrapper import *


# Counterfactual Dataset, with all ground truth stacked
class CounterfactualDictDatasetTMSCM(StackDataset):

    def __init__(
        self,
        tmscm: TMSCM,
        n_samples: int = 16384,
    ):
        self._iota = tmscm.iota
        TMSCM.check_tmscm(tmscm)

        observation, intervention, intervention_mask, counterfactual_outcome = \
            CounterfactualDictDatasetTMSCM.setup_datasets(
                tmscm=tmscm,
                n_samples=n_samples,
            )
        super().__init__(
            **{
                'observation': observation,
                'intervention': intervention,
                'intervention_mask': intervention_mask,
                'counterfactual_outcome': counterfactual_outcome,
            })

    @staticmethod
    def setup_datasets(
        tmscm: TMSCM,
        n_samples: int,
    ):
        exogenous_values = tmscm.sample_exogenous(n_samples=n_samples)

        observation_values = tmscm.pushforward(
            exogenous_values=exogenous_values,
            intervention_values=None,
            intervention_masks=None,
        )

        intervention_values = tmscm.sample_endogenous(n_samples=n_samples, )

        intervention_masks = CounterfactualDictDatasetTMSCM.random_intervention_masks(
            tmscm=tmscm,
            n_samples=n_samples,
        )

        for endo_index in intervention_values:
            intervened = intervention_masks[endo_index]
            intervention_values[endo_index][~intervened] = th.nan

        counterfactual_outcome_values = tmscm.pushforward(
            exogenous_values=exogenous_values,
            intervention_values=intervention_values,
            intervention_masks=intervention_masks,
        )

        observation_datasets = StackDataset(
            **{
                endo_index: ColumnDataset(observation)
                for endo_index, observation in observation_values.items()
            })
        intervention_datasets = StackDataset(
            **{
                endo_index: ColumnDataset(intervention)
                for endo_index, intervention in intervention_values.items()
            })
        intervention_mask_datasets = StackDataset(
            **{
                endo_index: ColumnDataset(mask)
                for endo_index, mask in intervention_masks.items()
            })
        counterfactual_outcome_datasets = StackDataset(
            **{
                endo_index: ColumnDataset(counterfactual)
                for endo_index, counterfactual in
                counterfactual_outcome_values.items()
            })
        return observation_datasets, intervention_datasets, intervention_mask_datasets, counterfactual_outcome_datasets

    @staticmethod
    def random_intervention_masks(
        tmscm: TMSCM,
        n_samples: int = 16384,
    ):
        graph = tmscm.graph
        subsets = interventionally_meaningful_subsets(graph=graph)

        mask_candidates = th.stack(
            tensors=[
                th.tensor([
                    1 if endo_index in s else 0
                    for endo_index in tmscm.iota.order
                ]) for s in subsets
            ],
            dim=0,
        ).bool()
        mask_samples = mask_candidates[th.randint(
            high=mask_candidates.size(0),
            size=(n_samples, ),
        )]

        masks = {
            endo_index:
            th.zeros(
                n_samples,
                tmscm.iota.dimensions[endo_index],
            ).bool()
            for endo_index in tmscm.iota.order
        }
        for i, endo_index in enumerate(tmscm.iota.order):
            masks[endo_index][mask_samples[:, i]] = True

        return masks

    @property
    def iota(self) -> Iota:
        return self._iota


# Counterfactual Dataset, with 3 additional preprocessing wrappers
class CounterfactualDatasetTMSCM(
        saved(standardized(vectorized(cls=CounterfactualDictDatasetTMSCM)))):

    def __init__(
        self,
        tmscm: TMSCM,
        filepath: str,
        n_samples: int = 16384,
    ):
        self._iota = tmscm.iota

        super().__init__(
            tmscm=tmscm,
            n_samples=n_samples,
            vectorize_rule={
                'observation': tmscm.iota.order,
                'intervention': tmscm.iota.order,
                'intervention_mask': tmscm.iota.order,
                'counterfactual_outcome': tmscm.iota.order,
            },
            standardize_rule={
                'observation': {
                    'mean': 'observation',
                    'std': 'observation'
                },
                'intervention': {
                    'mean': 'observation',
                    'std': 'observation'
                },
                'intervention_mask': None,
                'counterfactual_outcome': {
                    'mean': 'observation',
                    'std': 'observation'
                },
            },
            filepath=filepath,
        )

    @property
    def iota(self) -> Iota:
        return self._iota
