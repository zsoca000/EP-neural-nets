import torch
from pathlib import Path
import numpy as np

class DataSet:
    
    def __init__(self,mat_name,inp_type,inp_name,data_path='data'):
        
        # Save props
        self.mat_name = mat_name
        self.inp_type = inp_type
        self.inp_name = inp_name
        
        # Find the data files
        data_path = Path(data_path)
        self.input_path = data_path / 'input' / inp_type / f'{self.inp_name}.npy'
        self.output_path = data_path / 'output'  / mat_name / inp_type / f'{self.inp_name}.npy'

    @property
    def u_list(self):
        ret = np.load(self.input_path, allow_pickle=True)
        ret = self.data_to_tensor(ret)
        return ret
    
    @property
    def y_list(self):
        ret = np.load(self.output_path, allow_pickle=True)
        ret = self.data_to_tensor(ret)
        return ret

    @staticmethod
    def data_to_tensor(x:np.array):
        return torch.tensor(
            x.astype(np.float32),
            dtype=torch.float32
        ).unsqueeze(-1)
    
    

