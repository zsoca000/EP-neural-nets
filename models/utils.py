import torch
import numpy as np


def hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def data_to_tensor(x:np.array):
    return torch.tensor(
        x.astype(np.float32),
        dtype=torch.float32
    ).unsqueeze(-1)


class MinMaxScaler:
    def __init__(self, x=None):
        if x is not None:
            self.x_max, self.x_min = x.max(), x.min()
        else:
            self.x_max, self.x_min = None, None

    def transform(self, x):
        if isinstance(x, np.ndarray):
            x_max = self.x_max.cpu().numpy() if torch.is_tensor(self.x_max) else self.x_max
            x_min = self.x_min.cpu().numpy() if torch.is_tensor(self.x_min) else self.x_min
        else:
            x_max, x_min = self.x_max, self.x_min
        return 2 * (x - x_min) / (x_max - x_min) - 1

    def inverse_transform(self, x):
        if isinstance(x, np.ndarray):
            x_max = self.x_max.cpu().numpy() if torch.is_tensor(self.x_max) else self.x_max
            x_min = self.x_min.cpu().numpy() if torch.is_tensor(self.x_min) else self.x_min
        else:
            x_max, x_min = self.x_max, self.x_min
        return (x + 1) * (x_max - x_min) / 2 + x_min
    
    def state_dict(self):
        return {
            'x_max': self.x_max,
            'x_min': self.x_min
        }

    def load_state_dict(self, state_dict):
        self.x_max = state_dict['x_max']
        self.x_min = state_dict['x_min']


class ErrorMetrics:
    def __init__(self, X_t, X_p, scaler: MinMaxScaler):

        if isinstance(X_t, torch.Tensor):
            X_t = X_t.detach().cpu().numpy()
        if isinstance(X_p, torch.Tensor):
            X_p = X_p.detach().cpu().numpy()

        if X_t.shape != X_p.shape:
            raise ValueError(
                f"X_t and X_p must have the same shape, "
                f"got {X_t.shape} and {X_p.shape}"
            )

        self.X_t = X_t
        self.X_p = X_p
        self.scaler = scaler

    @property
    def MSE_rel(self):
        ret = (self.X_t - self.X_p) ** 2
        ret = ret.mean()
        return float(ret)
    
    @property
    def MAE_rel(self):
        ret = np.abs(self.X_t - self.X_p)
        ret = ret.mean()
        return float(ret)

    @property
    def MAE(self):
        X_t_inv = self.scaler.inverse_transform(self.X_t)
        X_p_inv = self.scaler.inverse_transform(self.X_p)
        ret = np.abs(X_t_inv - X_p_inv)
        ret = ret.mean()
        return float(ret)

