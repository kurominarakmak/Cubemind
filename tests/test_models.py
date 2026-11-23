import torch

from cubemind.models import build_model
from cubemind.models.common import encode_history


def test_mlp_and_transformer_produce_move_logits():
    states = torch.zeros((2, 54), dtype=torch.long)
    histories = encode_history([[1, 5], []], 8, torch.device("cpu"))
    configs = [
        {"type": "mlp", "use_history": True, "max_history": 8, "hidden_dims": [32]},
        {"type": "transformer", "use_history": True, "max_history": 8,
         "embedding_dim": 32, "attention_heads": 4, "layers": 1, "feedforward_dim": 64},
    ]
    for config in configs:
        model = build_model(config)
        assert model(states, histories).shape == (2, 18)
