from .n_c import Net_C
import torch
from .skeleton import Skeleton
from utils.initializer import Initializer


class Net_DP(Skeleton):
    def __init__(self, input_dim, width, depth, output_dim):
        super(Net_DP, self).__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.write_model_info(locals())
        self.write_init_args(locals())
        self.net_type = "n_dp"

        self.dp_0 = Net_C(self.input_dim, self.width, self.depth, self.output_dim)
        self.dp_1 = Net_C(
            self.input_dim + self.output_dim, self.width, self.depth, self.output_dim
        )
        self.dp_2 = Net_C(
            self.input_dim + self.output_dim * 2,
            self.width,
            self.depth,
            self.output_dim,
        )

        self.apply(Initializer)

    def forward(self, x):
        assert len(x.shape) == 2
        # cNN-DP forward
        y = self.dp_0(x)  # Zeroth order time derivative
        yDot = self.dp_1(torch.cat([x, y], dim=1))  # First order time derivative
        yDDot = self.dp_2(
            torch.cat([x, y, yDot], dim=1)
        )  # Second order time derivative

        return y, yDot, yDDot
