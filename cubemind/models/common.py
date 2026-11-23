import torch
import torch.nn.functional as F

from cubemind.environment.cube import MOVE_TOKENS

PAD_MOVE_ID = len(MOVE_TOKENS)


def encode_history(histories, max_history, device):
    if max_history == 0:
        return torch.empty((len(histories), 0), dtype=torch.long, device=device)
    batch = torch.full((len(histories), max_history), PAD_MOVE_ID, dtype=torch.long, device=device)
    for row, history in enumerate(histories):
        values = history[-max_history:]
        if values:
            batch[row, max_history - len(values):] = torch.as_tensor(values, dtype=torch.long, device=device)
    return batch


def state_one_hot(states):
    return F.one_hot(states.long(), num_classes=6).float().flatten(start_dim=1)
