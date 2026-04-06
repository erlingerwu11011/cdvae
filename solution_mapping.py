from copy import deepcopy
from typing import Dict, Any
from zuko.lazy import LazyTransform

from common import *
from model.solution_mappings import *

solution_mappings = {
    'dnme': DiagonalNoiseSolutionMapping,
    'tnme': TriangularNoiseSolutionMapping,
    'cmsm': ComposedMappedSolutionMapping,
    'tvsm': TriangularVelocityFieldSolutionMapping,
}


def solution_mapping(
    iota: Iota,
    solution_mapping_kwargs: Dict[str, Any],
) -> LazyTransform:
    """Instantiate solution mapping from `iota` and `solution_mapping_kwargs`.

    Args:
        `iota` (Iota): Rules for vectorization by causal order.
        `solution_mapping_kwargs` (Dict[str, Any]): Key-value pairs used to instantiate solution mapping.

    Returns:
        LazyTransform: Solution mapping.
    """
    kwargs = deepcopy(solution_mapping_kwargs)
    type = kwargs['type']
    del kwargs['type']
    return solution_mappings[type](
        iota=iota,
        **kwargs,
    )
