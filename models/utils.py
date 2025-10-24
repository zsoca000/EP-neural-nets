import torch
import numpy as np

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


def hhmmss(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"