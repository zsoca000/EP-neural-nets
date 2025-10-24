from models.ep_nn import EP_NN, load_model
from models.networks import LSTM, MLP
from data.materials import load_responses
from models.utils import data_to_tensor, hhmmss
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import os.path as osp
from itertools import product
import time

def train_model(model:EP_NN, mat_name, inp_name, config_path, verbose=False): 
    eps_list, sig_list = load_responses(mat_name,'random',inp_name,data_path='data')

    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)

    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

    model.fit(
        epochs=500,
        y_train=y_train, u_train=u_train,
        y_val=y_val,u_val=u_val,
        config_path=config_path,
        verbose=verbose,
    )

    return model.evaluate(y_test,u_test).item()


def eval_model(model:EP_NN, mat_name, inp_type, inp_name, save_plot=False):
    
    eps_list, sig_list = load_responses(
        mat_name,inp_type,inp_name,
        data_path='data'
    )

    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)

    return model.evaluate(
        y_list,u_list,
        save_plot=save_plot,
        path=osp.join(f'tmp/{inp_name}.png') # TODO: finish path
    ).item()


def find_best_model(
    k,p,q,incr,network,
    mat_name,inp_name,
    seeds
) -> EP_NN:

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
    
    return best_model


def task1(seeds=range(42,67)):

    mat_name = 'isotropic-swift'
    inp_name = 'pd_ms_42_200'

    cases = [
        (MLP , list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        (LSTM, list(product([0,2,3,5],[2,3,5,8],[1,2])))
    ]
    
    num_runs = 2 *  (len(cases[0][1]) + len(cases[1][1]))

    sum_time = 0.0
    count = 0

    for incr in [False, True]:
        for network, search_space in cases:
            for k,p,q in search_space:
                
                
                parent_folder = osp.join('metrics',mat_name, inp_name)
                if not osp.exists(parent_folder): os.makedirs(parent_folder)

                prefix = (
                    f"{network.__name__}-"
                    f"{'incr' if incr else 'dir'}-"
                    f"{k}-{p}-{q}"
                )

                exists = any(
                    os.path.isdir(os.path.join(parent_folder, f)) and f.startswith(prefix)
                    for f in os.listdir(parent_folder)
                )
                
                # If there is no trained model yet
                if not exists:
                    
                    tic = time.time()
                    
                    # Find best fit with random restart
                    best_model = find_best_model(
                        k,p,q,incr,network, # Surrogate model
                        mat_name, # Behaviour
                        inp_name, # Data
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

                    sum_time = toc - tic
                    count += 1
                    avg_time = sum_time / count

                    
                    print(
                        f"Estimated remaining time ({count} / {num_runs}):", 
                        hhmmss(avg_time * (num_runs - count))
                    )

                


if __name__ == "__main__":
    
    task1(
        seeds=[42, 56, 17, 83, 29, 64, 90, 11, 75, 38]
    )


    
    
    # model = EP_NN(k, p, q, incr=True, network=MLP, seed=42)

    # test_err = train_model(
    #     model,
    #     mat_name = mat_name,
    #     inp_name = train_inp_name,
    #     config_path='models/train_config.json',
    # )

    # print(f"{test_err*100} %")


    # eval_list = []
    # for inp_name in ['amplitude', 'cyclic', 'impulse', 'piecewise']:
    #     eval_list += [
    #         eval_model(model,mat_name,'static',inp_name,save_plot=True)
    #     ]
    #     print(f'{inp_name}: {eval_list[-1]*100:02f} %')

    # for inp_name in ['bl_ms_42_200', 'gp_42_200', 'pd_ms_42_200', 'rw_42_200']:
    #     eval_list += [
    #         eval_model(model,mat_name,'random',inp_name,save_plot=False)
    #     ]
    #     print(f'{inp_name}: {eval_list[-1]*100:02f} %')

    # print(f'\nMean: {np.mean(eval_list)*100:02f} %')