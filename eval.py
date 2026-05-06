import json

from pathlib import Path
from train import Trainer
from models.models import SeqMLP, SeqLSTM
from models.utils import ErrorMetrics
from data.materials import load_responses
from data.data_set import DataSet


class Evaluator:
    
    def __init__(self, datasets:list[DataSet]):
        self.datasets = datasets
        self.eval_metrics = {}

    
    def evaluate(self, load_dir:str, verbose:bool=False, overwrite:bool=False):

        model_path = Path(load_dir,'model.pth')
        eval_metrics_path = Path(load_dir,'eval_metrics.json')
        
        if eval_metrics_path.exists() and not overwrite:
            print(f"{load_dir} already evaluated — skipping ⚠️")
        else:  
            
            # Load the model from path
            state_dict = Trainer.load_state_dict_from_path(model_path)
            model = Trainer.load_model(state_dict)

            for dataset in self.datasets:
                
                if verbose: print(f'Evaluate {load_dir} on {dataset.inp_name}')

                self.eval_metrics[dataset.inp_name] = self.eval_model(
                    model, dataset,
                )
                    
        
            with open(eval_metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f)

    
    @staticmethod
    def eval_model(
        model:SeqMLP|SeqLSTM, 
        dataset:DataSet
    ) -> tuple[ErrorMetrics,ErrorMetrics]:

        u_list = dataset.u_list
        y_list = dataset.y_list
        
        glob_err = model.glob_err(y_list,u_list)
        loc_err = model.loc_err(y_list,u_list)

        return {
            'global' : glob_err.dictionary,
            'local' : loc_err.dictionary
        }
    

    
    