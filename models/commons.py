import torch.nn as nn
import torch
import numpy as np

def data_to_tensor(x:np.array):
    return torch.tensor(
        x.astype(np.float32),
        dtype=torch.float32
    ).unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self,input_size,output_size,p,q,seed=42):
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, p)])
        for _ in range(q - 2):
            self.hidden_layers.append(nn.Linear(p, p))
        self.hidden_layers.append(nn.Linear(p, output_size))
        self.ReLU = nn.ReLU()
        
        # He initialization (for ReLU activations)
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.ReLU(layer(x))
        return self.hidden_layers[-1](x)


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, p, q, seed=42):
        super(LSTM, self).__init__()
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(input_size, p, num_layers=q, batch_first=True)
        self.fc = nn.Linear(p, output_size)

        # Xavier/He initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, nonlinearity='sigmoid')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


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
