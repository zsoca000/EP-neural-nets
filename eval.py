import json

from pathlib import Path
from train import Trainer
from models.models import SeqMLP, SeqLSTM
from models.utils import ErrorMetrics
from data.materials import load_responses


class Evaluator:
    
    def __init__(self, mat_name:str, eval_sets:dict):
        self.mat_name = mat_name
        self.eval_sets = eval_sets
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

            for inp_type, eval_inp_names in self.eval_sets.items():
                for eval_inp_name in eval_inp_names:
                    
                    if verbose: 
                        print(f'Evaluate {load_dir} on {inp_type}/{eval_inp_name}')

                    self.eval_metrics[eval_inp_name] = self.eval_model(
                        model, self.mat_name, 
                        inp_type, eval_inp_name,
                    )
                    
        
            with open(eval_metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f)

    
    @staticmethod
    def eval_model(
        model:SeqMLP|SeqLSTM, 
        mat_name:str, inp_type:str, inp_name:str, 
    ) -> tuple[ErrorMetrics,ErrorMetrics]:


        u_list, y_list = load_responses(
            mat_name,inp_type,inp_name,
            data_dir='data'
        )
        
        glob_err = model.glob_err(y_list,u_list)
        loc_err = model.loc_err(y_list,u_list)

        return {
            'global' : glob_err.dictionary,
            'local' : loc_err.dictionary
        }
    


if __name__ == '__main__':
    
    evaluator = Evaluator(
        mat_name='isotropic-swift',
        eval_sets = {
            'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
            'random': ['bl_ms_42_200','combined_42_200','gp_42_200','pd_ms_42_200','rw_42_200']
        }
    )


    evaluator.evaluate(
        load_dir='tmp2/isotropic-swift/pd_ms_42_200/MLP-dir-5-2-2-29',
        verbose=True,
    )

    

    
    