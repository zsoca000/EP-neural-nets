import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,input_size,output_size,p,q):
        super(MLP, self).__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, p)])
        for _ in range(q - 2):
            self.hidden_layers.append(nn.Linear(p, p))
        self.hidden_layers.append(nn.Linear(p, output_size))
        self.ReLU = nn.ReLU()

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
