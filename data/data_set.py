import torch
from pathlib import Path
import numpy as np

class DataSet:
    
    def __init__(self,mat_name,inp_type,inp_name,data_path='data'):
        
        self.mat_name = mat_name
        self.inp_type = inp_type
        self.inp_name = inp_name

        eps_list = self.load_eps(data_path)
        sig_list = self.load_sig(data_path)

        self.u_list = self.data_to_tensor(eps_list)
        self.y_list = self.data_to_tensor(sig_list)


    def load_eps(self,data_path:str)->np.array:
        return np.load(
            Path(data_path,'input',self.inp_type,f'{self.inp_name}.npy'),
            allow_pickle=True
        )
    
    def load_sig(self,data_path:str)->np.array:
        return np.load(
            Path(data_path,'output',self.mat_name,self.inp_type,f'{self.inp_name}.npy'),
            allow_pickle=True
        )

    @staticmethod
    def data_to_tensor(x:np.array):
        return torch.tensor(
            x.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1)
    
    

