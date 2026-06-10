"""
EP-Neural-Nets: Dataset Handler and Loader

This module defines the DataSet class, which serves as a unified interface 
for loading and managing elastoplastic time-series data. It maps specific 
combinations of loading histories and material models to their corresponding 
input (strain), output (stress), and state files, providing lazy-loading 
access directly as PyTorch float32 tensors.

Key Components:
---------------
* DataSet (class) - Represents a target behavior dataset, encapsulating path resolutions 
                    and properties for inputs (u), outputs (y), and state vectors (x).

Key Methods/Properties:
-----------------------
* u_list (property) - Loads and returns the input strain histories as a PyTorch tensor (N, T, 1).
* y_list (property) - Loads and returns the output stress responses as a PyTorch tensor (N, T, 1).
* x_list (property) - Loads and returns the internal material state variables (gamma, eps_p, alpha) 
                      as a PyTorch tensor (N, T, 3).
* data_to_tensor()  - Utility method to cast NumPy arrays into PyTorch float32 tensors.
"""

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
        self.states_path = data_path / 'states'  / mat_name / inp_type / f'{self.inp_name}.npy'

    @property
    def u_list(self):
        ret = np.load(self.input_path, allow_pickle=True)
        ret = self.data_to_tensor(ret).unsqueeze(-1)
        return ret # (N,T,1)

    @property
    def x_list(self):
        ret = np.load(self.states_path, allow_pickle=True)
        ret = self.data_to_tensor(ret)
        return ret # (N,T,3)


    @property
    def y_list(self):
        ret = np.load(self.output_path, allow_pickle=True)
        ret = self.data_to_tensor(ret).unsqueeze(-1)
        return ret # (N,T,1)



    @staticmethod
    def data_to_tensor(x:np.array):
        return torch.tensor(
            x.astype(np.float32),
            dtype=torch.float32
        )
    
    

