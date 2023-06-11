import torch
import torch.nn as nn
import architectures
import platform


class MSELoss(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def forward(self, *components):
        assert (
            len(components) == 2 * self.n_components
        ), "Number of loss components should be even."
        assert (
            components[0].device == components[-1].device
        ), f"Device mismatch. Prior components located in {components[0].device}, posterior components located in {components[-1].device}."
        labels = components[: self.n_components]
        preds = components[self.n_components :]
        losses = torch.zeros(self.n_components, device=components[0].device)
        for i in range(self.n_components):
            losses[i] = torch.mean(torch.square(labels[i] - preds[i]))
        return losses  # return loss vector


class WeightedMSELoss(nn.Module):
    def __init__(self, n_components: int = 3):
        super().__init__()
        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_components = n_components
        self.alpha = nn.Parameter(
            torch.full([self.n_components], 0, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )

        # Parameter mapping function
        # self.mapper=lambda x:torch.sigmoid(x)
        self.mapper = lambda x: torch.softmax(x, dim=0) * self.n_components

    def forward(self, *components):
        assert (
            len(components) == 2 * self.n_components
        ), "Number of loss components should be even."
        labels = components[: self.n_components]
        preds = components[self.n_components :]
        weights = self.mapper(self.alpha)
        losses = torch.zeros(self.n_components, device=self.device)
        for i in range(self.n_components):
            losses[i] = weights[i] * torch.mean(torch.square(labels[i] - preds[i]))
        return losses
