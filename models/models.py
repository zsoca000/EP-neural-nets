import torch

from models.networks import LSTM, MLP
from models.utils import MinMaxScaler, ErrorMetrics


class Preprocessor:
    def __init__(self,k,autoreg,incr):
        self.autoreg = autoreg
        self.incr = incr
        self.k = k
        self.u_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.dy_scaler = MinMaxScaler()

    def fit(self,u:torch.tensor,y:torch.tensor)->None:
        self.u_scaler.fit(u)
        self.y_scaler.fit(y)
        if self.incr:
            self.dy_scaler.fit(torch.diff(y, dim=1))

    def transform(self,u,y) -> tuple[torch.tensor, torch.tensor]: 
        return self.calc_input(u,y), self.calc_output(y)
    
    def calc_output(self,y):
        
        if self.incr:
            dy = torch.diff(y, dim=1) # (S, T-1, Fy)
            dy_hat = self.dy_scaler.transform(dy) # (S, T-1, Fy)
            output = dy_hat[:,self.k-1:,:] # (S, T-k, Fy)
        else:
            y_hat = self.y_scaler.transform(y) # (S, T, Fy)
            output = y_hat[:, self.k:, :] # (S, T-k, Fy)
        
        return output # (S, T-k, Fy)

    def calc_input(self,u,y=None):
        
        u_hat = self.u_scaler.transform(u) # (S, T, Fu)
        u_next = u_hat[:, self.k:, :].unsqueeze(2) # (S, T-k, 1, Fu)
        u_prev = u_hat.unfold(1, self.k, 1)[:, :-1, :, :] # (S, T-k, Fu, k)
        u_prev = u_prev.transpose(-1,-2) # (S, T-k, k, Fu)

        input = torch.cat([u_prev,u_next],dim=-2) # (S, T-k, k+1, Fu)
        input = input.flatten(start_dim=2) # (S, T-k, (k+1)*Fu)

        if y is not None and self.autoreg:
            y_hat = self.y_scaler.transform(y) # (S, T, Fy)
            y_prev = y_hat.unfold(1, self.k, 1)[:, :-1, :, :] # (S, T-k, Fy, k)
            y_prev = y_prev.transpose(-1,-2) # (S, T-k, k, Fy)
            y_prev = y_prev.flatten(start_dim=2) # (S, T-k, k*Fy)
            input = torch.cat([y_prev,input],dim=-1) # (S, T-k, k*Fy+(k+1)*Fu)

        return input # (S, T-k, k*Fy+(k+1)*Fu)


class SeqModelBase(torch.nn.Module):
    def __init__(self, k, incr, autoreg, **kwargs):
        super().__init__(**kwargs)
        self.k = k

        self.incr = incr
        self.autoreg = autoreg
        self.preprocessor = Preprocessor(k,autoreg=autoreg,incr=incr)


        if incr and not k:
            raise ValueError(f'Incremental model with k=0 cannot be created')
    
    def rollout(self, u, y_init):
        raise NotImplementedError("Child class must implement rollout")

    @torch.inference_mode()
    def glob_err(self, y_test: torch.Tensor, u_test: torch.Tensor):
        y_init = y_test[:, :self.k, :]
        y_pred = self.rollout(u_test, y_init)
        
        y_test_scaled = self.preprocessor.y_scaler.transform(y_test)
        y_pred_scaled = self.preprocessor.y_scaler.transform(y_pred)

        return ErrorMetrics(y_test_scaled, y_pred_scaled, self.preprocessor.y_scaler)

    @torch.inference_mode()
    def loc_err(self, y_test: torch.Tensor, u_test: torch.Tensor):
        input, output_t = self.preprocessor.transform(u_test, y_test)

        output_p = self(input)

        output_scaler = self.preprocessor.dy_scaler if self.incr else self.preprocessor.y_scaler
        return ErrorMetrics(output_t, output_p, output_scaler)
    
    @property
    def num_params(self):
        return sum(
            p.numel() 
            for p in self.parameters() 
            if p.requires_grad
        )

    @property
    def mode(self):
        return 'incr' if self.incr else 'dir'

    @property
    def name(self):
        net_name = getattr(self, 'network_name', 'UnknownNet')
        p = getattr(self, 'p', 'p?')
        q = getattr(self, 'q', 'q?')
        seed = getattr(self, 'seed', 'seed?')
        return f'{net_name}-{self.mode}-{self.k}-{p}-{q}-{seed}'


class SeqMLP(MLP, SeqModelBase):
    def __init__(self,k,p,q,incr,seed=42):
        super().__init__(
            input_size=2*k+1, output_size=1, p=p, q=q, seed=seed, # LSTM 
            k=k, incr=incr, autoreg=True # SeqModelBase
        )
        self.network_name = 'MLP'

    def rollout(self,u,y_init):

        y_init = self.preprocessor.y_scaler.transform(y_init) # (S,k,Fy)
        u = self.preprocessor.u_scaler.transform(u) # (S,T,Fu)
        
        y_pred = y_init #  (S,k,Fy)
        for i in range(u.shape[1]-self.k):
            u_prev = u[:,i:i+self.k,:] # (S,k,Fu)
            u_next = u[:,i+self.k:i+self.k+1,:] # (S,1,Fu)

            input = torch.cat([u_prev,u_next], dim=1) # (S,k+1,Fu)
            input = input.flatten(start_dim=1) # (S,(k+1)*Fu)

            y_prev = y_pred[:,i:i+self.k,:] # (S,k,Fy)
            y_prev = y_prev.flatten(start_dim=1) # (S,k*Fy)

            input  = torch.cat([y_prev,input], dim=1) # (S,k*Fy+(k+1)*Fu)
            
            if self.incr:
                # prediction
                dy = self(input) # (S,Fy)
                # scale to MPa
                dy = self.preprocessor.dy_scaler.inverse_transform(dy) # (S,Fy)
                # Last pred value
                y_last = y_pred[:,-1,:] # (S,Fy)
                # scale to MPa
                y_last = self.preprocessor.y_scaler.inverse_transform(y_last) # (S,Fy)
                # step
                y_next = y_last + dy # (S,Fy)
                # rescale to rel y scale
                y_next = self.preprocessor.y_scaler.transform(y_next)  # (S,Fy)
            else:
                # prediction
                y_next = self(input) # (S,Fy)
            
            y_next = y_next.unsqueeze(dim=1) # (S,1,Fy)
            y_pred = torch.cat([y_pred,y_next],dim=1)  # (S,k+i,Fy)
        
        # rescale to MPa 
        return self.preprocessor.y_scaler.inverse_transform(y_pred) # (S,T,Fy)


class SeqLSTM(LSTM, SeqModelBase):
    def __init__(self,k,p,q,incr,seed=42):
        super().__init__(
            input_size=k+1, output_size=1, p=p, q=q, seed=seed, # LSTM 
            k=k, incr=incr, autoreg=False # SeqModelBase
        )
        self.network_name = 'LSTM'
   
    
    def rollout(self,u,y_init):
        input = self.preprocessor.calc_input(u) # (S, T-k, (k+1)*Fu)
        output = self(input) # (S, T-k, Fy)

        if not self.incr:
            y = self.preprocessor.y_scaler.inverse_transform(output)
            y_pred = torch.cat([y_init, y], dim=1) # (S, T, Fy)
        else:
            y_pred = y_init
            for i in range(output.shape[1]): # T-k
                dy = output[:,i:i+1,:] # (S, 1, Fy)
                # Rescale to MPa
                dy = self.preprocessor.dy_scaler.inverse_transform(dy) # (S, 1, Fy)
                y_next = y_pred[:,-1:,:] + dy # (S, 1, Fy)
                y_pred = torch.cat([y_pred,y_next],dim=1) # (S, k+i, Fy)
        
        return y_pred
    


def get_model(k,p,q,mode,network_name,seed=42) -> SeqMLP | SeqMLP:
    
    incr = mode == 'incr'
    
    if network_name == 'MLP':
        return SeqMLP(k,p,q,incr,seed) 
    elif network_name == 'LSTM':
        return SeqLSTM(k,p,q,incr,seed)







