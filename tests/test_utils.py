import numpy as np
import torch

from competitive_rl.utils.utils import FrameStackTensor, flatten_dict, summary


def test_flatten_dict_nested_values():
    payload = {
        "episode": 3,
        "stats": {
            "reward": 10,
            "length": 42,
        },
        "meta": {"seed": 7},
    }

    flattened = flatten_dict(payload)

    assert flattened == {
        "episode": 3,
        "stats/reward": 10,
        "stats/length": 42,
        "meta/seed": 7,
    }


def test_summary_handles_empty_and_populated_arrays():
    populated = summary([1, 5, 3], "score")
    empty = summary([], "score")

    assert populated == {"score_mean": 3.0, "score_min": 1.0, "score_max": 5.0}
    assert np.isnan(empty["score_mean"])
    assert np.isnan(empty["score_min"])
    assert np.isnan(empty["score_max"])


def test_frame_stack_tensor_rolls_and_masks():
    frame_stack = FrameStackTensor(num_envs=2, obs_shape=(1, 2, 2), frame_stack=3, device="cpu")

    first_obs = np.ones((2, 1, 2, 2), dtype=np.float32)
    frame_stack.update(first_obs)
    first_state = frame_stack.get()
    assert torch.allclose(first_state[:, -1:], torch.ones_like(first_state[:, -1:]))

    second_obs = np.full((2, 1, 2, 2), 2.0, dtype=np.float32)
    mask = torch.tensor([[0.0], [1.0]])
    frame_stack.update(second_obs, mask)
    second_state = frame_stack.get()

    assert torch.all(second_state[0, :-1] == 0)
    assert torch.all(second_state[0, -1] == 2)
    assert torch.all(second_state[1, -2] == 1)
    assert torch.all(second_state[1, -1] == 2)
