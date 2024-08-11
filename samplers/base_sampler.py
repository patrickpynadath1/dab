import torch.nn as nn


class BaseSampler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize_batch(self, model, seq_length, sentiment):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError
