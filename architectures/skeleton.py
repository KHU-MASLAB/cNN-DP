import torch.nn as nn
import numpy as np
from copy import deepcopy


class Skeleton(nn.Module):
    def count_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += np.prod(param.data.shape)
        print(f"Number of trainable parameters: {num_params:,}")
        return num_params

    def print_model_info(self):
        assert "model_info" in self.__dict__.keys(), "self.model_info not declared."
        print(f"{'=' * 100}")
        for k, v in self.model_info.items():
            print(f"{k.upper()}: {v}")
        self.count_params()
        print(f"{'=' * 100}")
        print()

    def write_model_info(self, locals):  # put locals() to record configurations
        try:
            del locals["self"]  # Cannot be pickled
        except KeyError:
            pass
        self.__dict__.update(locals)
        self.model_info.update(locals)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}

    def write_init_args(self, locals):
        assert (
            "model_init_args" not in self.__dict__.keys()
        ), "model_init_args already exists in self.__dict__"
        locals = deepcopy(locals)
        try:
            del locals["self"], locals["__class__"]  # Cannot be pickled
        except KeyError:
            pass
        self.model_init_args = {}
        self.model_init_args.update(locals)
