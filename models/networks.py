"""
EP-Neural-Nets: Core PyTorch Neural Network Architectures

This module defines the raw PyTorch neural network structures (MLP and LSTM) 
that serve as the mathematical engines for the surrogate models. It includes 
parameterized layer construction and custom weight initialization techniques 
(Kaiming Normal, Orthogonal, Xavier) to ensure stable and reproducible training.

Key Components:
---------------
* MLP  - Multi-Layer Perceptron (feedforward network) with customizable depth (q) 
         and width (p), ReLU hidden activations, and Kaiming/Xavier weight initialization.
* LSTM - Recurrent network wrapping PyTorch's nn.LSTM layer with customizable hidden 
         states, num layers, orthogonal recurrent weight initialization, and a fully 
         connected output layer.
"""

import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
            self, input_size, output_size, 
            p, q, final_activation=None, seed=42, **kwargs
        ):
        super().__init__(**kwargs)
        self.p, self.q = p,q
        self.seed = seed
        torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, p)])
        for _ in range(q - 2):
            self.hidden_layers.append(nn.Linear(p, p))
        self.hidden_layers.append(nn.Linear(p, output_size))
        self.ReLU = nn.ReLU()
        
        self.final_activation = final_activation
        
        # He initialization (for ReLU activations)
        self._init_weights()

    def _init_weights(self):
        for layer in self.hidden_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        # A kimeneti réteget érdemesebb Xavier-rel indítani, ha nem ReLU követi
        nn.init.xavier_normal_(self.hidden_layers[-1].weight)
        nn.init.zeros_(self.hidden_layers[-1].bias)

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.ReLU(layer(x))
        
        x = self.hidden_layers[-1](x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, p, q, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.p, self.q = p,q
        self.seed = seed
        torch.manual_seed(seed)
        self.lstm = nn.LSTM(input_size, hidden_size=p, num_layers=q, batch_first=True)
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
    


if __name__ == "__main__":
    
    k,p,q = 5,8,1

    model = LSTM(input_size=k+1, output_size=1, p=p, q=q)
    # print(model)
    x = torch.randn(200, 150, k+1)

    out, (h_n, c_n) = model.lstm(x)

    print(out.shape)   
    print(h_n.shape)   
    print(c_n.shape)

    total_params = sum(p.numel() for p in model.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    fc_params = sum(p.numel() for p in model.fc.parameters())
    print("Total parameters:", total_params)
    print("LSTM params:", lstm_params)
    print("FC params:", fc_params)
    print("Total parameters:", 4 * (p * (k + 1) + p**2 + 2*p) + (q - 1) * 4 * (2 * p**2 + 2*p))
    
