import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from .n_c import Net_C
from .n_ag import Net_AG
from .n_dp import Net_DP
import platform
import time


class NetInterface(nn.Module):
    def __init__(self, path_model, use_gpu: bool = True):
        super().__init__()
        # Load model and set device
        self.model = torch.load(path_model)
        self.__dict__.update(
            self.model
        )  # net_type, state_dict, loss_history, model_info, model_init_args, scale_params
        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale_params = {k: v.to(self.device) for k, v in self.scale_params.items()}

        # Create net instance
        if self.net_type == "n_c":
            self.net = Net_C(**self.model["model_init_args"]).to(self.device)
        elif self.net_type == "n_ag":
            self.net = Net_AG(**self.model["model_init_args"]).to(self.device)
        elif self.net_type == "n_dp":
            self.net = Net_DP(**self.model["model_init_args"]).to(self.device)

        # Load parameter and eval()
        self.net.load_state_dict(self.model["state_dict"])
        self.net.eval()

    def forward(
        self,
        batch_x,
        return_time=False,
    ):
        # scale
        batch_x = (batch_x - self.scale_params["mean_x"]) / self.scale_params["std_x"]
        # compute, unscale and copy to cpu
        if self.net_type == "n_c":
            with torch.no_grad():
                lap1 = time.process_time() if return_time else None
                yDDot = self.net(batch_x)
                lap2 = time.process_time() if return_time else None
                yDDot = (
                    yDDot * self.scale_params["std_yDDot"]
                    + self.scale_params["mean_yDDot"]
                )
                if not return_time:
                    return yDDot.cpu()
                elif return_time:
                    return yDDot.cpu(), lap2 - lap1
        elif self.net_type == "n_ag":
            lap1 = time.process_time() if return_time else None
            y, yDot, yDDot = self.net(batch_x)
            lap2 = time.process_time() if return_time else None
            with torch.no_grad():
                y = y * self.scale_params["std_y"] + self.scale_params["mean_y"]
                yDot = yDot / (
                    self.scale_params["std_x"][0] / self.scale_params["std_y"]
                )
                yDDot = yDDot / (
                    self.scale_params["std_x"][0] ** 2 / self.scale_params["std_y"]
                )
                if not return_time:
                    return y.cpu(), yDot.cpu(), yDDot.cpu()
                elif return_time:
                    return y.cpu(), yDot.cpu(), yDDot.cpu(), lap2 - lap1
        elif self.net_type == "n_dp":
            with torch.no_grad():
                lap1 = time.process_time() if return_time else None
                y, yDot, yDDot = self.net(batch_x)
                lap2 = time.process_time() if return_time else None
                y = y * self.scale_params["std_y"] + self.scale_params["mean_y"]
                yDot = (
                    yDot * self.scale_params["std_yDot"]
                    + self.scale_params["mean_yDot"]
                )
                yDDot = (
                    yDDot * self.scale_params["std_yDDot"]
                    + self.scale_params["mean_yDDot"]
                )
                if not return_time:
                    return y.cpu(), yDot.cpu(), yDDot.cpu()
                elif return_time:
                    return y.cpu(), yDot.cpu(), yDDot.cpu(), lap2 - lap1

    def predict(self, x: torch.Tensor, return_time=False, batch_size=2048):
        """
        Includes GPU-CPU transfer, scaling/unscaling for each type of nets
        :param x: Unscaled input
        :return: Unscaled output
        """
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float()
        dataloader = DataLoader(
            TensorDataset(x),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        times = [] if return_time else None

        preds = [[], [], []]
        for idx, batch in enumerate(dataloader):
            (batch_x,) = batch
            #  to gpu
            batch_x = batch_x.to(self.device)
            out = self.forward(batch_x, return_time=return_time)
            batch_pred = out[:-1] if return_time else out
            t = out[-1] if return_time else None
            if return_time:
                times.append(t)
            if type(batch_pred) is tuple and len(batch_pred) > 1:
                for i in range(3):
                    preds[i].append(batch_pred[i])
            else:
                preds[-1].append(batch_pred)
            if (idx + 1) % 1 == 0:
                print(f"{self.net.net_type}: Batch {idx+1}/{len(dataloader)}")
        del batch, batch_x, batch_pred
        torch.cuda.empty_cache()

        for i in range(3):
            try:
                preds[i] = torch.cat(preds[i], dim=0).numpy()
            except:
                continue
        if not return_time:
            return preds
        elif return_time:
            return preds, np.mean(times[5:])
