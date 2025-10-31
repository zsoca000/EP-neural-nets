import os
import time
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split

from models.ep_nn import EP_NN, load_model
from models.networks import LSTM, MLP
from data.materials import load_responses
from models.utils import ErrorMetrics, data_to_tensor, hhmmss


def train_model(
    model:EP_NN, 
    mat_name:str, inp_name:str, 
    config_path:str, verbose:bool=False,
    epoch:int=500,
) -> np.float32: 
    
    eps_list, sig_list = load_responses(
        mat_name,'random',inp_name,
        data_path='data'
    )

    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)
    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)

    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

    if verbose: print(f'Train {model.name} with {model.num_params} number of params...')

    model.fit(
        epochs=epoch,
        y_train=y_train, u_train=u_train,
        y_val=y_val,u_val=u_val,
        config_path=config_path,
        verbose=verbose,
    )

    return model.glob_err(y_test,u_test).MSE_rel


def eval_model(
    model:EP_NN, 
    mat_name:str, inp_type:str, inp_name:str, 
    save_plot:bool=False,plot_path:str=None
) -> tuple[ErrorMetrics,ErrorMetrics]:
 
    eps_list, sig_list = load_responses(
        mat_name,inp_type,inp_name,
        data_path='data'
    )
    
    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)

    glob_err = model.glob_err(
        y_list,u_list,
        save_plot=save_plot,
        path=plot_path
    )

    loc_err = model.loc_err(
        y_list,u_list,
    )

    return glob_err, loc_err


def task1():

    # Trainer config
    trainer = Trainer(
        mat_name='isotropic-swift', 
        inp_name='pd_ms_42_200', 
        config_path='models/train_config.json',
        seeds=[42, 56, 17, 83, 29, 64, 90, 11, 75, 38],
        epochs=500,
    )

    # Evaluator config
    evaluator = Evaluator(
        mat_name='isotropic-swift',
        eval_sets = {
            'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
            'random': ['bl_ms_42_200','gp_42_200','rw_42_200']
        }
    )
    
    # Model space
    model_space = [
        (MLP , False, list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        (MLP , True , list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        (LSTM, False, list(product([0,2,3,5],[2,3,5,8],[1,2]))),
        (LSTM, True, list(product([2,3,5],[2,3,5,8],[1,2]))),
    ]
    
    num_runs = len(model_space) * sum([len(model_space[-1]) for i in range(len(model_space))])

    sum_time = 0.0
    count = 0

    for network, incr, search_space in model_space:
        for k,p,q in search_space:
            
            tic = time.time()

            # Seek and train model
            model = trainer.get_best_model(k,p,q, incr, network)
            model_dir = trainer.save_model(model)

            # Eval model
            evaluator.evaluate(model_dir)

            toc = time.time()
            sum_time += toc - tic
            count += 1
            avg_time = sum_time / count

            print(
                f"Estimated remaining time ({count} / {num_runs}):", 
                hhmmss(avg_time * (num_runs - count))
            )


def task2():
    """Check large parameter models"""

    trainer = Trainer(
        mat_name='isotropic-swift', 
        inp_name='pd_ms_42_200', 
        config_path='models/train_config.json',
        seeds=[56],
        epochs=500,
    )

    evaluator = Evaluator(
        mat_name='isotropic-swift',
        eval_sets = {
            'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
            'random': ['bl_ms_42_200','gp_42_200','rw_42_200']
        }
    )

    # Train model
    model = trainer.get_best_model(5,16,3, incr=False, network=MLP,verbose=True)
    model_dir = trainer.save_model(model)
    
    # Eavluation
    evaluator.evaluate(model_dir)


class Trainer:
    
    def __init__(self,mat_name:str, inp_name:str, config_path:str, seeds:list[int],epochs:int):
        self.mat_name = mat_name
        self.inp_name = inp_name
        self.config_path = config_path
        self.seeds = seeds
        self.epochs = epochs

        parent_folder = osp.join('metrics',mat_name,inp_name)
        if not osp.exists(parent_folder): os.makedirs(parent_folder)
        

    def load_data(self,data_path):
        eps_list, sig_list = load_responses(
            self.mat_name,'random',self.inp_name,
            data_path=data_path
        )

        y_list = data_to_tensor(sig_list)
        u_list = data_to_tensor(eps_list)

        self.y_train, y_tmp, self.u_train, u_tmp = train_test_split(
            y_list, u_list, test_size=0.3, random_state=42
        )

        self.y_val, self.y_test, self.u_val, self.u_test = train_test_split(
            y_tmp, u_tmp, test_size=0.5, random_state=42
        )


    def get_best_model(
        self,
        k:int, p:int, q:int, incr:bool, network:MLP|LSTM,
        verbose=False,
    ) -> tuple[EP_NN,np.float32]:

        exists, folder_path = self.model_exists(k,p,q,incr,network)

        if exists:
            name = f'{k}-{p}-{q}-{'incr' if incr else 'dir'}-{network.__name__}'
            print(f'{name} already exists - skipping ⚠️')
            return load_model(osp.join(folder_path,'model.pth'))
        else:
            
            best_test_err = float('inf')
            best_model = None
        
            for i,seed in enumerate(self.seeds):
                model = EP_NN(k,p,q,incr=incr,network=network,seed=seed)

                log = (
                    f'Train {model.name} model, '
                    f'for {self.mat_name} on {self.inp_name} '
                    f'{i+1}/{len(self.seeds)} 🔄'
                )
                print(log, end='\r')

                test_err = self._train_single(model,verbose=verbose)
                
                log = log.replace(
                    '🔄', f'✅ - Test Error: {test_err:.8f}'
                )
                print(log, end='\n')

                if best_test_err > test_err:
                    best_test_err = test_err
                    best_model = model
            
            return best_model


    def model_exists(self,k,p,q,incr,network):
        
        parent_folder = osp.join(
            'metrics',self.mat_name, self.inp_name
        )

        prefix = (
            f"{network.__name__}-"
            f"{'incr' if incr else 'dir'}-"
            f"{k}-{p}-{q}"
        )

        folder_path = None
        for f in os.listdir(parent_folder):
            full = os.path.join(parent_folder, f)
            if os.path.isdir(full) and f.startswith(prefix):
                folder_path = full
                break

        return folder_path is not None, folder_path


    def _train_single(self,model:EP_NN,verbose:bool=False):
        
        self.load_data(data_path='data')

        if verbose: print(f'Train {model.name} with {model.num_params} number of params...')

        model.fit(
            epochs=self.epochs,
            y_train=self.y_train, u_train=self.u_train,
            y_val=self.y_val,u_val=self.u_val,
            config_path=self.config_path,
            verbose=verbose,
        )

        return model.glob_err(self.y_test,self.u_test).MSE_rel
    

    def save_model(self,model:EP_NN):
        
        model_dir = osp.join('metrics',self.mat_name,self.inp_name,model.name)
        model_path = osp.join(model_dir,'model.pth')

        if not osp.exists(model_dir): 
            os.makedirs(model_dir)
        
        if not osp.exists(model_path):
            model.save(model_path)
        
        return model_dir


class Evaluator:
    
    def __init__(self, mat_name:str, eval_sets:dict):
        self.mat_name = mat_name
        self.eval_sets = eval_sets
        self.eval_metrics = {}

    
    def evaluate(self, model_dir:str, verbose=False):
        
        model_path = osp.join(model_dir,'model.pth')
        eval_metrics_path = osp.join(model_dir,'eval_metrics.json')
        
        if osp.exists(eval_metrics_path):
            print(f"{model_dir} already evaluated — skipping ⚠️")
        else:  
            
            model = load_model(model_path)
            
            for inp_type, eval_inp_names in self.eval_sets.items():
                for eval_inp_name in eval_inp_names:
                    
                    if verbose: print(f'Evaluate {model_dir} on {inp_type}/{eval_inp_name}')

                    glob_err, loc_err = self._eval_single(
                        model, self.mat_name, inp_type, eval_inp_name,
                    )

                    self.eval_metrics[eval_inp_name] = {
                        'global' : glob_err.dictionary,
                        'local' : loc_err.dictionary
                    }
        
            with open(eval_metrics_path, 'w') as f:
                json.dump(self.eval_metrics, f)

    
    def _eval_single(
        self,
        model:EP_NN, 
        mat_name:str, inp_type:str, inp_name:str, 
        save_plot:bool=False, plot_path:str=None
    ) -> tuple[ErrorMetrics,ErrorMetrics]:


        eps_list, sig_list = load_responses(
            mat_name,inp_type,inp_name,
            data_path='data'
        )
        
        y_list = data_to_tensor(sig_list)
        u_list = data_to_tensor(eps_list)

        glob_err = model.glob_err(
            y_list,u_list,
            save_plot=save_plot,
            path=plot_path
        )

        loc_err = model.loc_err(
            y_list,u_list,
        )

        return glob_err, loc_err

    
    
        
if __name__ == '__main__':
    
    task2()


