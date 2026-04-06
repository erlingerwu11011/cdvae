from copy import deepcopy
from typing import Dict, Any
from zuko.lazy import LazyDistribution

from common import *
from model.exogenous_distributions import *

exogenous_distributions = {
    'n': ExogenousStandardNormal,
    'gmm': ExogenousGaussianMixture,
    'nf': ExogenousNormalizingFlow,
}


def exogenous_distribution(
    iota: Iota,
    exogenous_distribution_kwargs: Dict[str, Any],
) -> LazyDistribution:
    """Instantiate exogenous distribution from `iota` and `exogenous_distribution_kwargs`.

    Args:
        `iota` (Iota): Rules for vectorization by causal order.
        `exogenous_distribution_kwargs` (Dict[str, Any]): Key-value pairs used to instantiate exogenous distribution.

    Returns:
        LazyDistribution: Exogenous distribution.
    """
    kwargs = deepcopy(exogenous_distribution_kwargs)
    type = kwargs['type']
    del kwargs['type']
    return exogenous_distributions[type](
        iota=iota,
        **kwargs,
    )
