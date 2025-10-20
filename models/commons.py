import torch.nn as nn
import torch

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


class MinMaxScaler:
    def __init__(self,x):
        self.x_max, self.x_min = x.max(), x.min()

    def transform(self,x):
        return 2*(x - self.x_min)/(self.x_max - self.x_min) - 1
    
    def inverse_transform(self,x):
        return (x+1) * (self.x_max - self.x_min)/2 + self.x_min
