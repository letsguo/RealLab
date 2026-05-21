from typing import List
import torch
from torch.utils.data import DataLoader

class GradAccumulator:
    def __init__(self, params: List[torch.nn.Parameter]):
        self.params = params
        self.grads = {p: torch.zeros_like(p) for p in self.params}
        self.iters = 0

    def add_grads(self):

        for p in self.params:
            if p.grad is None:
                raise ValueError("Parameter has no gradient yet")
            self.grads[p] += p.grad.detach().clone()

        self.iters += 1

    def get_grads(self):
        return {p: self.grads[p] / self.iters for p in self.params}

    def reset(self):
        for p in self.params:
            self.grads[p].zero_()

def endless_dataloader(dl: DataLoader):
    while True:
        for batch in dl:
            yield batch
