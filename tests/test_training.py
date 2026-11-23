import torch

from cubemind.data.prepare import make_synthetic_records, prepare
from cubemind.environment.cube import SOLVED_STATE, apply_move, cross_complete
from cubemind.evaluate import evaluate
from cubemind.training.engine import beam_search_cross, train_model


def test_policy_beam_search_returns_a_verified_cross_path():
    class UniformPolicy(torch.nn.Module):
        def forward(self, states, histories):
            return torch.zeros((len(states), 18), device=states.device)

    start = apply_move(SOLVED_STATE, "R")
    moves, final_state, nodes_scored, solved = beam_search_cross(
        UniformPolicy(), start, max_history=0, device=torch.device("cpu"), beam_width=18, max_moves=1
    )
    assert solved
    assert len(moves) == 1
    assert cross_complete(final_state)
    assert nodes_scored == 1


def test_tiny_training_and_evaluation_round_trip(tmp_path):
    data_dir = tmp_path / "data"
    prepare(data_dir, make_synthetic_records(24, seed=5), seed=5)
    checkpoint = tmp_path / "tiny.pt"
    config = {
        "name": "tiny-mlp",
        "data": {"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "validation.jsonl")},
        "model": {"type": "mlp", "use_history": False, "hidden_dims": [32]},
        "train": {"seed": 5, "device": "cpu", "cpu_threads": 1, "epochs": 2, "batch_size": 32,
                  "learning_rate": 0.001, "early_stopping_patience": 2,
                  "checkpoint": str(checkpoint), "metrics": str(tmp_path / "metrics.jsonl")},
    }
    train_model(config, "tiny-config.yaml")
    result = evaluate(checkpoint, data_dir / "test.jsonl", data_dir / "train.jsonl", beam_width=4, beam_max_moves=5)
    assert checkpoint.exists()
    assert result["teacher_forced"]["samples"] > 0
    assert result["autoregressive_cross_rollout"]["solves"] > 0
    assert result["policy_guided_beam_search"]["solves"] > 0
