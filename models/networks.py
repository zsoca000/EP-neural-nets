import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_size, output_size, p, q, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.p, self.q = p,q
        self.seed = seed
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

    
