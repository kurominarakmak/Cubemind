import torch
import torch.nn as nn

from .common import PAD_MOVE_ID


class TransformerPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_history = bool(config.get("use_history", True))
        self.max_history = int(config.get("max_history", 8)) if self.use_history else 0
        dimension = int(config.get("embedding_dim", 64))
        heads = int(config.get("attention_heads", 4))
        layers = int(config.get("layers", 2))
        if dimension % heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        self.state_embedding = nn.Embedding(6, dimension)
        self.move_embedding = nn.Embedding(PAD_MOVE_ID + 1, dimension, padding_idx=PAD_MOVE_ID)
        self.cls = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.position = nn.Parameter(torch.empty(1, 1 + 54 + self.max_history, dimension))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=int(config.get("feedforward_dim", dimension * 4)),
            dropout=float(config.get("dropout", 0.1)),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, 18)

    def forward(self, states, histories):
        batch_size = states.shape[0]
        tokens = [self.cls.expand(batch_size, -1, -1), self.state_embedding(states.long())]
        padding = [torch.zeros((batch_size, 1 + 54), dtype=torch.bool, device=states.device)]
        if self.use_history:
            tokens.append(self.move_embedding(histories.long()))
            padding.append(histories.eq(PAD_MOVE_ID))
        x = torch.cat(tokens, dim=1) + self.position[:, :sum(part.shape[1] for part in tokens)]
        mask = torch.cat(padding, dim=1)
        return self.head(self.norm(self.encoder(x, src_key_padding_mask=mask)[:, 0]))
