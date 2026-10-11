"""Raw CMR preparation and pretraining, with no downstream or protocol-audit entry."""
from .config import default_settings
from .data.prepare import prepare_data

__all__ = ["default_settings", "prepare_data", "PretrainingExperiment"]


def __getattr__(name):
    if name == "PretrainingExperiment":
        from .pretrain import PretrainingExperiment
        return PretrainingExperiment
    raise AttributeError(name)
