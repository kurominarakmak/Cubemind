from .mlp import MLPPolicy
from .transformer import TransformerPolicy


def build_model(config):
    model_type = config["type"]
    if model_type == "mlp":
        return MLPPolicy(config)
    if model_type == "transformer":
        return TransformerPolicy(config)
    raise ValueError("Unsupported model type: {}".format(model_type))


__all__ = ["MLPPolicy", "TransformerPolicy", "build_model"]
