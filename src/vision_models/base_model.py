import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import abc
import torch

class BaseModel(abc.ABC):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def load_model(self):
        pass
