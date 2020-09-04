import json

import torch
from typing import Type


def compute_accuracy(ans: Type[torch.Tensor], gt_ans: Type[torch.Tensor]):
    _, ans_idx = ans.max(dim=1, keepdim=True)
    match = gt_ans.gather(dim=1, index=ans_idx)
    return (match * 0.3).clamp(max=1)


class M3:
    def __init__(self):
        self.meters = {}

    def add_meter(self, key, meter):
        self.meters[key] = meter

    def get_meter(self, key):
        return self.meters[key]

    def to_dict(self):
        return {k: json.dumps(v) for k, v in self.meters.items()}

    def encode(self):
        return json.dumps(self.meters)

    def decode(self, encoded_obj):
        self.meters = json.loads(encoded_obj)


class MeanMeter:
    def __init__(self):
        self.n = 0
        self.value = 0

    def update(self, curr_value):
        self.value += curr_value
        self.n += 1

    def update_all(self, curr_value, curr_n):
        self.value += curr_value
        self.n += curr_n

    @property
    def metric(self):
        return self.value / self.n


class MovingMeanMeter:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.value = 0
        self.first = True

    def update(self, curr_value):
        if self.first:
            self.first = False
            self.value = curr_value
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * curr_value

    @property
    def metric(self):
        return self.value
