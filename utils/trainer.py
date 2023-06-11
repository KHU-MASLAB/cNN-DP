import time
from datetime import datetime
import pandas as pd
from architectures.n_c import Net_C
from .loss import *
from .snippets import *
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import RAdam
from sklearn.metrics import r2_score
import os
import platform


class Trainer:
    """
    Trainer class
    :param net: model instances
    :type net: architectures.n_c.Net_C, architectures.n_ag.Net_AG, architectures.n_dp.Net_DP
    """

    def __init__(self, net: Net_C):
        if platform.system() == "Darwin":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net.to(self.device)
        self.net_type = net.net_type
        self.datetime = datetime.now().strftime("%y%m%d_%H%M%S")

    def fit(
        self,
        epochs=5000,
        initial_lr=1e-3,
        lr_halflife=500,
        loss_fn="mse",
        multioptim=True,
        save_name=None,
        save_grad=False,
        decouple_ord=None,
    ):
        self.setup_loss_fn(loss_fn=loss_fn)
        self.setup_optimizer(initial_lr=initial_lr, multioptim=multioptim)
        self.net.write_model_info(locals())
        self.net.print_model_info()
        if decouple_ord is not None:
            assert (
                self.net_type == "n_ag"
            ), "Derivative decoupling is supported for AG networks only"
            self.loss_fn_d = MSELoss(n_components=1)

        # Define scalers/unscalers
        def mse(x, y):
            return torch.mean(torch.square(x - y))

        def scale_x(x):
            return (x - self.scale_params["mean_x"]) / self.scale_params["std_x"]

        if (
            self.net_type == "n_ag"
        ):  # Apply unit Gaussian normalization on y and time-diff chain rule on derivatives

            def scale_0(y):
                return (y - self.scale_params["mean_y"]) / self.scale_params["std_y"]

            def scale_1(yDot):
                return yDot * (
                    self.scale_params["std_x"][0] / self.scale_params["std_y"]
                )

            def scale_2(yDDot):
                return yDDot * (
                    self.scale_params["std_x"][0] ** 2 / self.scale_params["std_y"]
                )

            def unscale_0(y):
                return y * self.scale_params["std_y"] + self.scale_params["mean_y"]

            def unscale_1(yDot):
                return yDot / (
                    self.scale_params["std_x"][0] / self.scale_params["std_y"]
                )

            def unscale_2(yDDot):
                return yDDot / (
                    self.scale_params["std_x"][0] ** 2 / self.scale_params["std_y"]
                )

        else:  # Apply unit Gaussian normalization

            def scale_0(y):
                return (y - self.scale_params["mean_y"]) / self.scale_params["std_y"]

            def scale_1(yDot):
                return (yDot - self.scale_params["mean_yDot"]) / self.scale_params[
                    "std_yDot"
                ]

            def scale_2(yDDot):
                return (yDDot - self.scale_params["mean_yDDot"]) / self.scale_params[
                    "std_yDDot"
                ]

            def unscale_0(y):
                return y * self.scale_params["std_y"] + self.scale_params["mean_y"]

            def unscale_1(yDot):
                return (
                    yDot * self.scale_params["std_yDot"]
                    + self.scale_params["mean_yDot"]
                )

            def unscale_2(yDDot):
                return (
                    yDDot * self.scale_params["std_yDDot"]
                    + self.scale_params["mean_yDDot"]
                )

        ########################################################################################################################
        ########################################################################################################################
        time_start = time.perf_counter()
        # Training epoch loop
        self.best_model_param = None
        err_min = torch.inf
        loss_argmin = 0
        self.loss_history = {
            "train": torch.zeros(epochs, 3),
            "valid": torch.zeros(epochs, 3),
        }
        self.error_history = {
            "train": torch.zeros(epochs, 3),
            "valid": torch.zeros(epochs, 3),
        }
        for epoch in range(epochs):
            ########################################################################################################################
            # Batch training loop
            self.net.train()
            self.scale_params = {
                k: v.to(self.device) for k, v in self.scale_params.items()
            }
            for idx_b, batch in enumerate(self.dataloader_train):
                train_x, train_y, train_yDot, train_yDDot = batch

                # GPU transfer
                train_x = train_x.to(self.device)
                train_yDDot = train_yDDot.to(self.device)
                if not self.net_type == "n_c":
                    train_y = train_y.to(self.device)
                    train_yDot = train_yDot.to(self.device)

                # Data scaling
                with torch.no_grad():
                    train_x = scale_x(train_x)
                    train_yDDot = scale_2(train_yDDot)
                    if not self.net_type == "n_c":
                        train_y = scale_0(train_y)
                        train_yDot = scale_1(train_yDot)

                # Forward pass
                if self.net_type == "n_c":
                    pred_yDDot = self.net(train_x)
                elif self.net_type == "n_ag":
                    pred_y, pred_yDot, pred_yDDot = self.net(train_x)
                elif self.net_type == "n_dp":
                    pred_y, pred_yDot, pred_yDDot = self.net(train_x)

                # Compute loss
                if self.net_type == "n_c":
                    loss = self.loss_fn(train_yDDot, pred_yDDot)
                else:
                    if loss_fn == "wmse_errorbased":
                        # Compute loss weights
                        with torch.no_grad():
                            loss_weights = torch.empty(
                                self.loss_fn.n_components, device=self.device
                            )
                            # / torch.abs(train_y - pred_y).max()
                            loss_weights[0] = 1
                            loss_weights[1] = (
                                1 / torch.abs(train_yDot - pred_yDot).max()
                            )
                            loss_weights[2] = (
                                1 / torch.abs(train_yDDot - pred_yDDot).max()
                            )
                            # loss_weights = flip(loss_weights)
                        loss = (
                            self.loss_fn(
                                train_y,
                                train_yDot,
                                train_yDDot,
                                pred_y,
                                pred_yDot,
                                pred_yDDot,
                            )
                            * loss_weights
                        )
                    elif loss_fn == "mse" and decouple_ord is None:
                        loss = self.loss_fn(
                            train_y,
                            train_yDot,
                            train_yDDot,
                            pred_y,
                            pred_yDot,
                            pred_yDDot,
                        )
                    elif loss_fn == "mse" and decouple_ord is not None:
                        if decouple_ord == 0:
                            loss = self.loss_fn_d(train_y, pred_y)
                        elif decouple_ord == 1:
                            loss = self.loss_fn_d(train_yDot, pred_yDot)
                        elif decouple_ord == 2:
                            loss = self.loss_fn_d(train_yDDot, pred_yDDot)
                        else:
                            raise UserWarning("Decoupling order must be 0, 1 or 2")
                    loss = torch.sum(loss)

                # Backward
                for param in self.net.parameters():  # Initialize parameter gradients
                    param.grad = None
                loss.backward()
                # for ord, l in enumerate(loss):
                #     l.backward(retain_graph=True)
                #     # Save parameter gradients every 100 epochs
                #     if save_grad:
                #         if (epoch + 1) % 100 == 0:
                #             if not self.net_type == 'n_c':
                #                 self.save_grad(epoch, ord)
                #             else:
                #                 self.save_grad(epoch, 2)
                for (
                    optim
                ) in (
                    self.optimizers
                ):  # n_dp uses three optimizer instances for a single backward
                    optim.step()

                if (idx_b + 1) % int(len(self.dataloader_train) / 3) == 0:
                    print(
                        f"Batch progress {idx_b}/{len(self.dataloader_train)}: {loss}"
                    )
            # Free memory
            del train_x, train_y, train_yDot, train_yDDot, pred_yDDot
            if "pred_y" in locals().keys():
                del pred_y, pred_yDot
            torch.cuda.empty_cache()
            ########################################################################################################################

            ########################################################################################################################
            # Training data evaluation loop
            self.net.eval()
            label = []
            prediction = []
            for batch in self.dataloader_train_eval:
                train_x, train_y, train_yDot, train_yDDot = batch

                # GPU transfer
                train_x = train_x.to(self.device)
                train_yDDot = train_yDDot.to(self.device)
                if not self.net_type == "n_c":
                    train_y = train_y.to(self.device)
                    train_yDot = train_yDot.to(self.device)

                # Data scaling
                with torch.no_grad():
                    train_x = scale_x(train_x)
                    train_yDDot = scale_2(train_yDDot)
                    if not self.net_type == "n_c":
                        train_y = scale_0(train_y)
                        train_yDot = scale_1(train_yDot)

                # Forward pass
                if self.net_type == "n_c":
                    with torch.no_grad():
                        pred_yDDot = self.net(train_x)
                elif self.net_type == "n_ag":
                    pred_y, pred_yDot, pred_yDDot = self.net(train_x)
                elif self.net_type == "n_dp":
                    with torch.no_grad():
                        pred_y, pred_yDot, pred_yDDot = self.net(train_x)

                # Compute loss
                with torch.no_grad():
                    # if self.net_type == 'n_c':
                    #     loss = self.loss_fn(train_yDDot, pred_yDDot)
                    # else:
                    #     loss = self.loss_fn(train_y, train_yDot, train_yDDot, pred_y, pred_yDot, pred_yDDot)
                    # loss = torch.sum(loss)

                    # Save training batches
                    if self.net_type == "n_c":
                        label.append(train_yDDot.cpu())
                        prediction.append(pred_yDDot.cpu())
                    else:
                        label.append(
                            torch.cat([train_y, train_yDot, train_yDDot], dim=1).cpu()
                        )
                        prediction.append(
                            torch.cat([pred_y, pred_yDot, pred_yDDot], dim=1).cpu()
                        )

            # Free memory
            del train_x, train_y, train_yDot, train_yDDot, pred_yDDot
            if "pred_y" in locals().keys():
                del pred_y, pred_yDot
            torch.cuda.empty_cache()

            # Concatenate and evaluate training data performance
            self.scale_params = {k: v.cpu() for k, v in self.scale_params.items()}
            with torch.no_grad():
                label = torch.cat(label, dim=0)
                prediction = torch.cat(prediction, dim=0)
                if self.net_type == "n_c":
                    self.loss_history["train"][epoch][2] = self.loss_fn(
                        label, prediction
                    )
                    self.error_history["train"][epoch][2] = torch.mean(
                        torch.square(unscale_2(label) - unscale_2(prediction))
                    )
                else:
                    self.loss_history["train"][epoch] = self.loss_fn(
                        label[:, : self.net.output_dim],
                        label[:, self.net.output_dim : 2 * self.net.output_dim],
                        label[:, 2 * self.net.output_dim :],
                        prediction[:, : self.net.output_dim],
                        prediction[:, self.net.output_dim : 2 * self.net.output_dim],
                        prediction[:, 2 * self.net.output_dim :],
                    )
                    if loss_fn == "wmse_errorbased":
                        loss_weights = torch.empty(self.loss_fn.n_components)
                        # / torch.abs(label[:, :self.net.output_dim] - prediction[:,
                        loss_weights[0] = 1
                        #:self.net.output_dim]).max()
                        loss_weights[1] = (
                            1
                            / torch.abs(
                                label[:, self.net.output_dim : 2 * self.net.output_dim]
                                - prediction[
                                    :, self.net.output_dim : 2 * self.net.output_dim
                                ]
                            ).max()
                        )
                        loss_weights[2] = (
                            1
                            / torch.abs(
                                label[:, 2 * self.net.output_dim :]
                                - prediction[:, 2 * self.net.output_dim :]
                            ).max()
                        )
                        # loss_weights = flip(loss_weights)
                        self.loss_history["train"][epoch] *= loss_weights

                    self.error_history["train"][epoch][0] = mse(
                        unscale_0(label[:, : self.net.output_dim]),
                        unscale_0(prediction[:, : self.net.output_dim]),
                    )
                    self.error_history["train"][epoch][1] = mse(
                        unscale_1(
                            label[:, self.net.output_dim : 2 * self.net.output_dim]
                        ),
                        unscale_1(
                            prediction[:, self.net.output_dim : 2 * self.net.output_dim]
                        ),
                    )
                    self.error_history["train"][epoch][2] = mse(
                        unscale_2(label[:, 2 * self.net.output_dim :]),
                        unscale_2(prediction[:, 2 * self.net.output_dim :]),
                    )

            ########################################################################################################################

            ########################################################################################################################
            # Validation data evaluation loop
            self.scale_params = {
                k: v.to(self.device) for k, v in self.scale_params.items()
            }
            label = []
            prediction = []
            for batch in self.dataloader_valid:
                valid_x, valid_y, valid_yDot, valid_yDDot = batch

                # GPU transfer
                valid_x = valid_x.to(self.device)
                valid_yDDot = valid_yDDot.to(self.device)
                if not self.net_type == "n_c":
                    valid_y = valid_y.to(self.device)
                    valid_yDot = valid_yDot.to(self.device)

                # Data scaling
                with torch.no_grad():
                    valid_x = scale_x(valid_x)
                    valid_yDDot = scale_2(valid_yDDot)
                    if not self.net_type == "n_c":
                        valid_y = scale_0(valid_y)
                        valid_yDot = scale_1(valid_yDot)

                # Forward pass
                if self.net_type == "n_c":
                    with torch.no_grad():
                        pred_yDDot = self.net(valid_x)
                elif self.net_type == "n_ag":
                    pred_y, pred_yDot, pred_yDDot = self.net(valid_x)
                elif self.net_type == "n_dp":
                    with torch.no_grad():
                        pred_y, pred_yDot, pred_yDDot = self.net(valid_x)

                # Compute loss
                with torch.no_grad():
                    # Save validation batches
                    if self.net_type == "n_c":
                        label.append(valid_yDDot.cpu())
                        prediction.append(pred_yDDot.cpu())
                    else:
                        label.append(
                            torch.cat([valid_y, valid_yDot, valid_yDDot], dim=1).cpu()
                        )
                        prediction.append(
                            torch.cat([pred_y, pred_yDot, pred_yDDot], dim=1).cpu()
                        )

            # Free memory
            del valid_x, valid_y, valid_yDot, valid_yDDot, pred_yDDot
            if "pred_y" in locals().keys():
                del pred_y, pred_yDot
            torch.cuda.empty_cache()

            # Concatenate and evaluate validation data performance
            self.scale_params = {k: v.cpu() for k, v in self.scale_params.items()}
            with torch.no_grad():
                label = torch.cat(label, dim=0)
                prediction = torch.cat(prediction, dim=0)
                if self.net_type == "n_c":
                    self.loss_history["valid"][epoch][2] = self.loss_fn(
                        label, prediction
                    )
                    self.error_history["valid"][epoch][2] = torch.mean(
                        torch.square(unscale_2(label) - unscale_2(prediction))
                    )
                else:
                    self.loss_history["valid"][epoch] = self.loss_fn(
                        label[:, : self.net.output_dim],
                        label[:, self.net.output_dim : 2 * self.net.output_dim],
                        label[:, 2 * self.net.output_dim :],
                        prediction[:, : self.net.output_dim],
                        prediction[:, self.net.output_dim : 2 * self.net.output_dim],
                        prediction[:, 2 * self.net.output_dim :],
                    )
                    if loss_fn == "wmse_errorbased":
                        loss_weights = torch.empty(self.loss_fn.n_components)
                        # / torch.abs(label[:, :self.net.output_dim] - prediction[:,
                        loss_weights[0] = 1
                        #:self.net.output_dim]).max()
                        loss_weights[1] = (
                            1
                            / torch.abs(
                                label[:, self.net.output_dim : 2 * self.net.output_dim]
                                - prediction[
                                    :, self.net.output_dim : 2 * self.net.output_dim
                                ]
                            ).max()
                        )
                        loss_weights[2] = (
                            1
                            / torch.abs(
                                label[:, 2 * self.net.output_dim :]
                                - prediction[:, 2 * self.net.output_dim :]
                            ).max()
                        )
                        # loss_weights = flip(loss_weights)
                        self.loss_history["valid"][epoch] *= loss_weights

                    self.error_history["valid"][epoch][0] = mse(
                        unscale_0(label[:, : self.net.output_dim]),
                        unscale_0(prediction[:, : self.net.output_dim]),
                    )
                    self.error_history["valid"][epoch][1] = mse(
                        unscale_1(
                            label[:, self.net.output_dim : 2 * self.net.output_dim]
                        ),
                        unscale_1(
                            prediction[:, self.net.output_dim : 2 * self.net.output_dim]
                        ),
                    )
                    self.error_history["valid"][epoch][2] = mse(
                        unscale_2(label[:, 2 * self.net.output_dim :]),
                        unscale_2(prediction[:, 2 * self.net.output_dim :]),
                    )
            # val_loss = torch.sum(self.loss_history['valid'][epoch])
            val_err = self.error_history["valid"][epoch][2]
            ########################################################################################################################

            # Save the best model
            if bool(val_err < err_min):
                err_min = val_err
                loss_argmin = epoch
                self.best_model_param = deepcopy(self.net.state_dict())

            # Decay lr by half
            if lr_halflife:
                if (epoch + 1) % lr_halflife == 0:
                    self.decay_lr()

            # Print
            if (epoch + 1) % 1 == 0:
                print(
                    f"{self.net_type.upper()}, Epoch: {epoch + 1}, Minimum validation loss: {err_min:.5f} at epoch {loss_argmin + 1}"
                )
                try:
                    print(f"Loss weights: {self.loss_fn.mapper(self.loss_fn.alpha)}")
                except:
                    pass
                try:
                    print(f"Loss weights: {loss_weights}")
                except:
                    pass
                print(f"Loss values: {loss}")

                if not self.net_type == "n_c":
                    print(
                        f"R2(y): {r2_score(label[:, :self.net.output_dim], prediction[:, :self.net.output_dim],multioutput='raw_values')}"
                    )
                    print(
                        f"R2(yDot): {r2_score(label[:, self.net.output_dim:2 * self.net.output_dim], prediction[:, self.net.output_dim:2 * self.net.output_dim],multioutput='raw_values')}"
                    )
                    print(
                        f"R2(yDDot): {r2_score(label[:, 2 * self.net.output_dim:], prediction[:, 2 * self.net.output_dim:],multioutput='raw_values')}\n"
                    )
                else:
                    print(
                        f'R2(yDDot): {r2_score(label, prediction,multioutput="raw_values")}\n'
                    )
        ########################################################################################################################
        ########################################################################################################################

        time_end = time.perf_counter()
        h, m, s = sec2hms(time_end - time_start)
        print(f"Training time: {h}hrs {m}min {s:.4f}sec")

        # End of training epochs
        self.save_model(save_name=save_name)

    def to_tensor(self, data: pd.DataFrame):
        return torch.FloatTensor(data.to_numpy())

    def setup_dataloader(
        self,
        batch_size: int,
        data_train: pd.DataFrame,
        data_valid: pd.DataFrame,
        input_cols: list,
        y_cols: list = None,
        yDot_cols: list = None,
        yDDot_cols: list = None,
    ):
        self.net.write_model_info(locals())
        del self.net.model_info["data_train"], self.net.model_info["data_valid"]

        assert len(input_cols) == self.net.input_dim
        assert yDDot_cols is not None

        # Data dividing
        cols = [input_cols, y_cols, yDot_cols, yDDot_cols]
        tensors_train = []
        tensors_valid = []
        for c in cols:
            tensors_train.append(self.to_tensor(data_train[c]))
            tensors_valid.append(self.to_tensor(data_valid[c]))

        # Compute Mean and Std
        names = ["x", "y", "yDot", "yDDot"]
        self.scale_params = {}
        for idx, n in enumerate(names):
            self.scale_params[f"mean_{n}"] = tensors_train[idx].mean(dim=0)
            self.scale_params[f"std_{n}"] = tensors_train[idx].std(dim=0)
            self.scale_params[f"range_{n}"] = (
                tensors_train[idx].max(dim=0).values
                - tensors_train[idx].min(dim=0).values
            )
            self.scale_params[f"min_{n}"] = tensors_train[idx].min(dim=0).values
        self.scale_params = {k: v.to(self.device) for k, v in self.scale_params.items()}

        # Dataloaders
        self.dataloader_train = DataLoader(
            TensorDataset(
                tensors_train[0], tensors_train[1], tensors_train[2], tensors_train[3]
            ),
            batch_size=self.net.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.dataloader_train_eval = DataLoader(
            TensorDataset(
                tensors_train[0], tensors_train[1], tensors_train[2], tensors_train[3]
            ),
            batch_size=2048,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.dataloader_valid = DataLoader(
            TensorDataset(
                tensors_valid[0], tensors_valid[1], tensors_valid[2], tensors_valid[3]
            ),
            batch_size=2048,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def setup_optimizer(self, initial_lr, multioptim=True):
        self.net.write_model_info(locals())
        if self.net_type == "n_dp" and multioptim:
            self.optimizers = [
                RAdam(
                    [
                        {"params": self.net.dp_0.parameters(), "lr": initial_lr},
                        {"params": self.loss_fn.parameters(), "lr": initial_lr},
                    ]
                ),
                RAdam(self.net.dp_1.parameters(), lr=initial_lr),
                RAdam(self.net.dp_2.parameters(), lr=initial_lr),
            ]
        else:
            self.optimizers = [
                RAdam(
                    [
                        {"params": self.net.parameters(), "lr": initial_lr},
                        {"params": self.loss_fn.parameters(), "lr": initial_lr},
                    ]
                )
            ]

    def setup_loss_fn(self, loss_fn="mse"):
        if self.net_type == "n_c":
            n_components = 1
        else:
            n_components = 3
        loss_fns = {
            "mse": MSELoss(n_components=n_components),
            "wmse": WeightedMSELoss(n_components=n_components),
            "wmse_errorbased": MSELoss(n_components=n_components),
        }
        self.loss_fn = loss_fns[loss_fn]

    def decay_lr(self):
        assert "optimizers" in self.__dict__.keys()
        for optim in self.optimizers:
            optim.param_groups[0]["lr"] /= 2

    def save_model(self, save_name=None):
        result = {
            "net_type": self.net_type,
            "state_dict": self.best_model_param,
            "loss_history": self.loss_history,
            "error_history": self.error_history,
            "model_info": self.net.model_info,
            "model_init_args": self.net.model_init_args,
            "scale_params": self.scale_params,
        }
        if save_name:
            path = f"models/{save_name}.pt"
        else:
            path = f"models/{self.net_type}_{self.datetime}.pt"
        torch.save(result, path)
        print(f"Model saved at: {path}")

    def save_grad(self, epoch, ord):
        try:
            path = f"models/{self.net_type}_{self.datetime}_grad/epoch_{epoch + 1:05d}"
            os.makedirs(path)
        except:
            pass
        grad = {n: p.grad.detach().cpu() for n, p in self.net.named_parameters()}
        torch.save(grad, f"{path}/grad_{ord}.pt")
