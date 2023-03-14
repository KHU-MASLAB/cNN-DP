import torch
import pandas as pd
from modules.n_c import Net_C
from modules.utils import mse
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import RAdam
from sklearn.metrics import r2_score

class Trainer:
    def __init__(self, net: Net_C):
        """
        Trainer class
        :param net: model instances
        :type net: modules.n_c.Net_C, modules.n_ag.Net_AG, modules.n_dp.Net_DP
        """
        self.net = net
        self.net_type = net.net_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_tensor(self, data: pd.DataFrame):
        return torch.FloatTensor(data.to_numpy())
    
    def to_device(self, data: torch.Tensor):
        return data.to(self.device, non_blocking=True)
    
    def setup_dataloader(self,
            batch_size: int,
            data_train: pd.DataFrame,
            data_valid: pd.DataFrame,
            input_cols: list,
            y_cols: list = None,
            yDot_cols: list = None,
            yDDot_cols: list = None):
        self.batch_size = batch_size
        self.input_cols = input_cols
        self.y_cols = y_cols
        self.yDot_cols = yDot_cols
        self.yDDot_cols = yDDot_cols
        assert len(input_cols) == self.net.input_dim
        assert yDDot_cols is not None
        
        # Data dividing
        train_x = self.to_tensor(data_train[input_cols])
        train_y = self.to_tensor(data_train[y_cols])
        train_yDot = self.to_tensor(data_train[yDot_cols])
        train_yDDot = self.to_tensor(data_train[yDDot_cols])
        valid_x = self.to_tensor(data_valid[input_cols])
        valid_y = self.to_tensor(data_valid[y_cols])
        valid_yDot = self.to_tensor(data_valid[yDot_cols])
        valid_yDDot = self.to_tensor(data_valid[yDDot_cols])
        
        # Compute Mean and Std
        self.mean_x = self.to_device(train_x.mean(dim=0))
        self.mean_y = self.to_device(train_y.mean(dim=0))
        self.mean_yDot = self.to_device(train_yDot.mean(dim=0))
        self.mean_yDDot = self.to_device(train_yDDot.mean(dim=0))
        self.std_x = self.to_device(train_x.std(dim=0))
        self.std_y = self.to_device(train_y.std(dim=0))
        self.std_yDot = self.to_device(train_yDot.std(dim=0))
        self.std_yDDot = self.to_device(train_yDDot.std(dim=0))
        
        # Dataloaders
        self.dataloader_train = DataLoader(TensorDataset(train_x, train_y, train_yDot, train_yDDot),
                                           batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        self.dataloader_valid = DataLoader(TensorDataset(valid_x, valid_y, valid_yDot, valid_yDDot),
                                           batch_size=10000, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def setup_optimizer(self, initial_lr):
        if self.net_type == 'net_dp':
            self.optimizers = [RAdam(self.net.dp0.parameters(), lr=initial_lr),
                               RAdam(self.net.dp1.parameters(), lr=initial_lr),
                               RAdam(self.net.dp2.parameters(), lr=initial_lr)]
        else:
            self.optimizers = [RAdam(self.net.parameters(), lr=initial_lr)]
    
    def fit(self, epochs=5000, initial_lr=1e-3, lr_halflife=500):
        self.setup_optimizer(initial_lr=initial_lr)
        # Training epoch loop
        best_model_param = None
        best_mse_value = torch.inf
        for epoch in range(epochs):
            # Training batch loop
            self.net.train()
            for batch in self.dataloader_train:
                train_x, train_y, train_yDot, train_yDDot = batch
                
                # Optimize GPU transfer
                if self.net_type == 'n_c':
                    train_x = self.to_device(train_x)
                    train_yDDot = self.to_device(train_yDDot)
                else:
                    train_x = self.to_device(train_x)
                    train_y = self.to_device(train_y)
                    train_yDot = self.to_device(train_yDot)
                    train_yDDot = self.to_device(train_yDDot)
                
                # Data scaling
                with torch.no_grad():
                    train_x = (train_x - self.mean_x) / self.std_x  # Apply unit Gaussian normalization
                    if self.net_type == 'n_c':  # Apply unit Gaussian normalization on yDDot
                        train_yDDot = (train_yDDot - self.mean_yDDot) / self.std_yDDot
                    elif self.net_type == 'n_ag':  # Apply time differentiation chain rule on yDot, yDDot
                        train_y = (train_y - self.mean_y) / self.std_y
                        train_yDot = train_yDot * (self.std_x[0] / self.std_y)
                        train_yDDot = train_yDDot * (self.std_x[0] ** 2 / self.std_y)
                    elif self.net_type == 'n_dp':  # Apply unit Gaussian normalization on three derivatives
                        train_y = (train_y - self.mean_y) / self.std_y
                        train_yDot = (train_yDot - self.mean_yDot) / self.std_yDot
                        train_yDDot = (train_yDDot - self.mean_yDDot) / self.std_yDDot
                
                # Forward pass
                if self.net_type == 'n_c':
                    pred_yDDot = self.net(train_x)
                else:
                    pred_y, pred_yDot, pred_yDDot = self.net(train_x)
                
                # Compute loss
                if self.net_type == 'n_c':  # yDDot loss only
                    loss = mse(train_yDDot, pred_yDDot)
                else:  # Add losses of lower order derivatives
                    loss = mse(train_y, pred_y)
                    loss += mse(train_yDot, pred_yDot)
                    loss += mse(train_yDDot, pred_yDDot)
                
                # Backward
                for param in self.net.parameters():  # Initialize parameter gradients
                    param.grad = None
                loss.backward()
                for optim in self.optimizers:  # n_dp uses three optimizer instances for a single backward
                    optim.step()
            
            # Validation batch loop
            self.net.eval()
            label = []
            prediction = []
            for batch in self.dataloader_valid:
                valid_x, valid_y, valid_yDot, valid_yDDot = batch
                
                # Optimize GPU transfer
                if self.net_type == 'n_c':
                    valid_x = self.to_device(valid_x)
                    valid_yDDot = self.to_device(valid_yDDot)
                else:
                    valid_x = self.to_device(valid_x)
                    valid_y = self.to_device(valid_y)
                    valid_yDot = self.to_device(valid_yDot)
                    valid_yDDot = self.to_device(valid_yDDot)
                
                # Data scaling
                with torch.no_grad():  # Apply unit Gaussian normalization
                    valid_x = (valid_x - self.mean_x) / self.std_x
                    if self.net_type == 'n_c':
                        valid_yDDot = (valid_yDDot - self.mean_yDDot) / self.std_yDDot
                    elif self.net_type == 'n_ag':  # Apply chain rule wrt time
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = valid_yDot * (self.std_x[0] / self.std_y)
                        valid_yDDot = valid_yDDot * (self.std_x[0] ** 2 / self.std_y)
                    elif self.net_type == 'n_dp':  # Apply unit Gaussian normalization
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot - self.mean_yDot) / self.std_yDot
                        valid_yDDot = (valid_yDDot - self.mean_yDDot) / self.std_yDDot
                
                # Forward pass
                if self.net_type == 'n_c':
                    with torch.no_grad():
                        pred_yDDot = self.net(valid_x)
                elif self.net_type == 'n_ag':
                    pred_y, pred_yDot, pred_yDDot = self.net(valid_x)
                    for p in self.net.parameters():  # Remove accumulated gradients by differential operators in params
                        p.grad = None
                elif self.net_type == 'n_dp':
                    with torch.no_grad():
                        pred_y, pred_yDot, pred_yDDot = self.net(valid_x)
                
                # Save validation batches
                with torch.no_grad():
                    if self.net_type == 'n_c':
                        label.append(valid_yDDot.cpu())
                        prediction.append(pred_yDDot.cpu())
                    else:
                        label.append(torch.cat([valid_y, valid_yDot, valid_yDDot], dim=1))
                        prediction.append(torch.cat([pred_y, pred_yDot, pred_yDDot], dim=1))
            
            # Compute total loss and save the best net parameter
            label = torch.cat(label, dim=0)
            prediction = torch.cat(prediction, dim=0)
            loss_total = torch.mean(torch.square(label - prediction))
            if loss_total < best_mse_value:
                best_mse_value = loss_total
                best_model_param = deepcopy(self.net.state_dict())
            
            # Decay lr by half
            if (epoch + 1) % lr_halflife == 0:
                for optim in self.optimizers:
                    optim.param_groups[0]['lr'] /= 2
            
            # Print
            if (epoch + 1) % 100 == 0:
                print(f'{self.net_type.upper()}, Epoch: {epoch + 1}, Best MSE value: {best_mse_value.item():.5f}')
                if not self.net_type == 'n_c':
                    print(f'R2(y): {r2_score(valid_y.detach().cpu().numpy(), pred_y.detach().cpu().numpy()):.5f}', )
                    print(f'R2(yDot): {r2_score(valid_yDot.detach().cpu().numpy(), pred_yDot.detach().cpu().numpy()):.5f}')
                print(f'R2(yDDot): {r2_score(valid_yDDot.detach().cpu().numpy(), pred_yDDot.detach().cpu().numpy()):.5f}\n')
        
        # End of training epochs
        result = {'net_type': self.net_type,
                  'state_dict': best_model_param,
                  'input_dim': self.net.input_dim,
                  'width': self.net.width,
                  'depth': self.net.depth,
                  'output_dim': self.net.output_dim,
                  'mean_x': self.mean_x,
                  'mean_y': self.mean_y,
                  'mean_yDot': self.mean_yDot,
                  'mean_yDDot': self.mean_yDDot,
                  'std_x': self.std_x,
                  'std_y': self.std_y,
                  'std_yDot': self.std_yDot,
                  'std_yDDot': self.std_yDDot, }
        torch.save(result, f"models/{self.net_type}.pt")
