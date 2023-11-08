from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class AccHolder:
    def __init__(self):
        self.__acc = []

        self.acc = []

    def __iadd__(self, packed_acc):
        self.__acc.append(packed_acc.item())

        return self

    def calc(self):
        self.acc.append(np.mean(self.__acc))

        self.__acc.clear()



@dataclass
class AccLogger:
    model_name: str
    train: AccHolder = AccHolder()
    val: AccHolder = AccHolder()
