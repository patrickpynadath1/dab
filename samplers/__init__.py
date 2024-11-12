from .bolt_sampler import BoltSampler
from .dlp_embed import LangevinSampler
from .dlp_variants import SelectiveMaskingDLP


def get_sampler(sampler_name):
    if sampler_name == "bolt":
        return BoltSampler
    elif sampler_name == "dlp":
        return LangevinSampler
    elif sampler_name == "deterministic_masking_langevin" or "stochastic_langevin":
        return SelectiveMaskingDLP
    else:
        raise ValueError(f"Sampler {sampler_name} not recognized")
