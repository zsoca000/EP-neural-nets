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


def find_best_model(
    k,p,q,incr,network,
    mat_name,inp_name,
    seeds
) -> tuple[EP_NN,np.float32]:

    best_test_err = float('inf')
    best_model = None

    for i,seed in enumerate(seeds):
        model = EP_NN(k,p,q,incr=incr,network=network,seed=seed)

        log = (
            f'Train {model.name} model, '
            f'for {mat_name} on {inp_name} '
            f'{i+1}/{len(seeds)} 🔄'
        )
        print(log, end='\r')

        test_err = train_model(
            model,
            mat_name = mat_name,
            inp_name = inp_name,
            config_path='models/train_config.json',
            verbose=False,
        )
        
        log = log.replace(
            '🔄',
            f'✅ - Test Error: {test_err:.8f}'
        )
        print(log, end='\n')

        if best_test_err > test_err:
            best_test_err = test_err
            best_model = model
    
    return best_model, best_test_err



def task1_train():

    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'
    
    seeds = [42, 56, 17, 83, 29, 64, 90, 11, 75, 38]
    
    iter = [
        (MLP , False, list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        (MLP , True , list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        (LSTM, False, list(product([0,2,3,5],[2,3,5,8],[1,2]))),
        (LSTM, True, list(product([2,3,5],[2,3,5,8],[1,2]))),
    ]
    

    num_runs = len(iter) * sum([len(iter[-1]) for i in range(len(iter))])

    sum_time = 0.0
    count = 0

    for network, incr, search_space in iter:
        for k,p,q in search_space:

            parent_folder = osp.join('metrics',mat_name, train_inp_name)
            if not osp.exists(parent_folder): os.makedirs(parent_folder)

            prefix = (
                f"{network.__name__}-"
                f"{'incr' if incr else 'dir'}-"
                f"{k}-{p}-{q}"
            )

            exists = any(
                os.path.isdir(os.path.join(parent_folder, f)) 
                and f.startswith(prefix)
                for f in os.listdir(parent_folder)
            )
            
            # If there is no trained model yet
            if not exists:
                
                tic = time.time()
                
                # Find best fit with random restart
                best_model, best_test_err = find_best_model(
                    k,p,q,incr,network, # Surrogate model
                    mat_name, # Behaviour
                    train_inp_name, # Data
                    seeds=seeds
                )

                # Where to save
                save_folder = osp.join(parent_folder, best_model.name)

                # Create the folder
                if not osp.exists(save_folder):
                    os.makedirs(save_folder)
                    
                # Save the best model
                best_model.save(
                    osp.join(save_folder,'model.pth')
                )


                toc = time.time()
                sum_time += toc - tic
                count += 1
                avg_time = sum_time / count

                print(
                    f"Estimated remaining time ({count} / {num_runs}):", 
                    hhmmss(avg_time * (num_runs - count))
                )


def task1_eval():
    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'

    eval_sets = {
        'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
        'random': ['bl_ms_42_200','gp_42_200','rw_42_200']
    }

    model_folder = osp.join('metrics',mat_name,train_inp_name)

    print(osp.exists('metrics/isotropic-swift/pd_ms_42_200/MLP-dir-3-3-3-11/eval_metrics.json'))

    for model_name in tqdm(os.listdir(model_folder)):
        
        eval_metrics_path = osp.join(
            model_folder,
            model_name, 
            'eval_metrics.json'
        )

        model_path = osp.join(
            model_folder, 
            model_name,
            'model.pth'
        )

        if osp.exists(eval_metrics_path):
            print(f'{model_path} model is already evaluated!')
        else:  
            model = load_model(model_path)
            eval_metrics = {}

            for inp_type, eval_inp_names in eval_sets.items():

                for eval_inp_name in eval_inp_names:

                    glob_err, loc_err = eval_model(
                        model, 
                        mat_name, inp_type, eval_inp_name,
                        save_plot=False
                    )

                    eval_metrics[eval_inp_name] = {
                        'global' : {
                            'MSE_rel' : glob_err.MSE_rel,
                            'MAE_rel' : glob_err.MAE_rel,
                            'MAE' : glob_err.MAE,
                        },
                        'local' : {
                            'MSE_rel' : loc_err.MSE_rel,
                            'MAE_rel' : loc_err.MAE_rel,
                            'MAE' : loc_err.MAE,
                        }
                    }
        
        
        
            with open(eval_metrics_path, 'w') as f:
                json.dump(eval_metrics, f)           

                      
def task2():
    
    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'
    k,p,q = (2,5,3)
    incr = True
    network = MLP
    seed = 75
    eval_inp_names = [
        'amplitude','cyclic','impulse','piecewise','resolution'
    ]
    

    # Init the model
    model = EP_NN(k,p,q,incr=incr,network=network,seed=seed)
    
    # Load responses
    eps_list, sig_list = load_responses(
        mat_name,'random',train_inp_name,data_path='data'
    )

    # Convert to tensor
    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)        

    # Split
    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

    # Define init test data
    y_tests, u_tests = [y_test], [u_test]
    for eval_inp_name in eval_inp_names:
        eps_list, sig_list = load_responses(
            mat_name,'static',eval_inp_name ,data_path='data'
        )
        y_tests.append(data_to_tensor(sig_list))
        u_tests.append(data_to_tensor(eps_list))
    
    
    model.fit(
        epochs=500,
        y_train=y_train, u_train=u_train,
        y_val=y_val,u_val=u_val,
        y_tests=y_tests,u_tests=u_tests,
        config_path='models/train_config.json',
        verbose=True,
    )

    model.save(
        osp.join(
            'metrics',mat_name, train_inp_name, 
            model.name,'model.pth'
        )
    )


def task3():
    """Check large parameter models"""

    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'
    seed = 42
    iter = [
        (LSTM, False, list(product([5],[16],[4]))),
    ]

    for network, incr, search_space in iter:
        for k,p,q in search_space:
            model = EP_NN(k,p,q,incr=incr,network=network,seed=seed)
            
            print

            test_err = train_model(
                model,
                mat_name = mat_name,
                inp_name = train_inp_name,
                config_path='models/train_config.json',
                verbose=True,
            )

            print('test_err',test_err)

                
            save_folder =  osp.join('metrics',mat_name,train_inp_name,model.name)
            if not osp.exists(save_folder): 
                os.makedirs(save_folder)
                model.save(osp.join(save_folder,'model.pth'))



if __name__ == "__main__":

    task3()
