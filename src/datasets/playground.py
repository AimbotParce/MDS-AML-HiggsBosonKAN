from typing import Callable, List, Optional, Tuple

import torch


def create_dataset(
    func: Callable[[torch.Tensor], torch.Tensor],
    n_var: int,
    train_num: int,
    test_num: int = 0,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    noise: float = 0.1,
    ranges: List[Tuple[float, float]] = [(-1, 1)],
):
    assert n_var >= 1, "n_var must be at least 1"
    assert len(ranges) == n_var, "Length of ranges must be equal to n_var"
    assert all(len(r) == 2 for r in ranges), "Each range must be a tuple of (low, high)"
    assert all(high > low for (low, high) in ranges), "Each range must have high > low"

    if seed is not None:
        torch.manual_seed(seed)

    x_train = torch.zeros(train_num, n_var).to(device)
    for i in range(n_var):
        low, high = ranges[i]
        x_train[:, i] = (high - low) * torch.rand(train_num).to(device) + low
    y_train = func(x_train) + noise * torch.randn(train_num, 1).to(device)
    dataset = {
        "train_input": x_train,
        "train_label": y_train,
    }
    if test_num > 0:
        x_test = torch.zeros(test_num, n_var).to(device)
        for i in range(n_var):
            low, high = ranges[i]
            x_test[:, i] = (high - low) * torch.rand(test_num).to(device) + low
        y_test = func(x_test) + noise * torch.randn(test_num, 1).to(device)
        dataset["test_input"] = x_test
        dataset["test_label"] = y_test
    return dataset
