"""
Backward-compatibility shim. Use cogformer.simulator.design_utils directly.
"""
from .design_utils import (
    split_design_space,
    build_design_config,
    build_random_design_config,
    build_parameter_mask,
    build_random_parameter_mask,
    build_intrinsic_priors,
    build_discrete_mask,
    build_random_discrete_mask,
    build_regressor_mask,
    build_design_matrix,
    build_link_functions,
    mask_to_design_config,
    sample_parameter_matrix,
    build_column_labels,
)


class ContextManager:
    """Stateless wrapper around design_utils free functions. Kept for backward compatibility."""

    def build_design_config(self, *args, **kwargs):
        return build_design_config(*args, **kwargs)

    def build_random_design_config(self, *args, **kwargs):
        return build_random_design_config(*args, **kwargs)

    def build_parameter_mask(self, *args, **kwargs):
        return build_parameter_mask(*args, **kwargs)

    def build_random_parameter_mask(self, *args, **kwargs):
        return build_random_parameter_mask(*args, **kwargs)

    def build_intrinsic_priors(self, *args, **kwargs):
        return build_intrinsic_priors(*args, **kwargs)

    def build_discrete_mask(self, *args, **kwargs):
        return build_discrete_mask(*args, **kwargs)

    def build_random_discrete_mask(self, *args, **kwargs):
        return build_random_discrete_mask(*args, **kwargs)

    def build_regressor_mask(self, *args, **kwargs):
        return build_regressor_mask(*args, **kwargs)

    def build_design_matrix(self, *args, **kwargs):
        return build_design_matrix(*args, **kwargs)

    def build_link_functions(self, *args, **kwargs):
        return build_link_functions(*args, **kwargs)

    def mask_to_design_config(self, *args, **kwargs):
        return mask_to_design_config(*args, **kwargs)

    def sample_parameter_matrix(self, *args, **kwargs):
        return sample_parameter_matrix(*args, **kwargs)

    def build_column_labels(self, *args, **kwargs):
        return build_column_labels(*args, **kwargs)
