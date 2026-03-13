import json

from pathlib import Path
from train import Trainer
from models.models import SeqMLP, SeqLSTM
from models.utils import ErrorMetrics
from data.materials import load_responses


class Evaluator:
    
    def __init__(self, mat_name:str, inp_names:list[str]):
        self.mat_name = mat_name
        self.inp_names = inp_names
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

            for inp_name in self.inp_names:
                
                if verbose: 
                    print(f'Evaluate {load_dir} on {inp_name}')

                self.eval_metrics[inp_name] = self.eval_model(
                    model, self.mat_name, inp_name,
                )
                    
        
            with open(eval_metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f)

    
    @staticmethod
    def eval_model(
        model:SeqMLP|SeqLSTM, 
        mat_name:str, inp_name:str, 
    ) -> tuple[ErrorMetrics,ErrorMetrics]:


        u_list, y_list = load_responses(
            mat_name, 'eval' ,inp_name,
            data_dir='data'
        )
        
        glob_err = model.glob_err(y_list,u_list)
        loc_err = model.loc_err(y_list,u_list)

        return {
            'global' : glob_err.dictionary,
            'local' : loc_err.dictionary
        }
    


if __name__ == '__main__':
    
    random_evals = [
        'bl_ms_56_20',
        'combined_56_20',
        'gp_56_20',
        'pd_ms_56_20',
        'rw_56_20',
    ]

    critical_evals = [
        'amplitude',
        'cyclic',
        'impulse',
        'piecewise',
        'resolution',
    ]

    evaluator = Evaluator(
        mat_name='isotropic-swift',
        inp_names= random_evals + critical_evals,
    )

    evaluator.evaluate(
        load_dir='/mnt/c/users/rdsup/desktop/EP-neural-nets/metrics/isotropic-swift/bl_ms_42_200/MLP-incr-3-5-3-90',
        verbose=True,
        overwrite=True,
    )

    

    
    