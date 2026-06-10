"""
EP-Neural-Nets: Task Orchestrator for Elastoplastic Surrogate Modeling
This module orchestrates the data generation, hyperparameter sweep (grid search),
cross-material training, and final evaluation of neural network surrogates 
for 1D rate-independent elastoplasticity.
Key Functions:
--------------
* generate_data()               - Generates stochastic (RW, GP, PD-MS, BL-MS) and 
                                critical loading histories, and computes reference 
                                analytical stress responses (formerly task0).
* run_grid_search()             - Executes the hyperparameter sweep across 184 network 
                                configurations (AR-MLP, LSTM) on Swift hardening (formerly task1).
* run_cross_material_training() - Trains the optimized surrogate model (AR-MLP 3-5-3) 
                                across all six constitutive models (formerly task2).
* eval_all()                    - Evaluates all trained configurations on all evaluation 
                                datasets for cross-generator robustness analysis.
Usage:
------
Run this script directly to execute the default main task (currently eval_all):
    python tasks.py
"""

import time
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import product

import data.generators as gen
from data.data_set import DataSet
from data.materials import hardening
from configs.materials import materials

from train import Trainer
from eval import Evaluator
from utils.time import hhmmss


ROOT_DIR = Path("")

SAVE_DIR = ROOT_DIR / 'metrics'
CONFIG_DIR = ROOT_DIR / 'configs'
DATA_DIR = ROOT_DIR / 'data'

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
STATES_DIR = DATA_DIR / "states"

def generate_data():
    """ Create all of the input sets, and calculate the responses for all materials and all input sets."""
    
    train_dir = INPUT_DIR / 'train'
    eval_dir = INPUT_DIR / 'eval'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_DIR / "generators.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_gen_classes = [
        gen.BaselineMultisine,
        gen.PowerDecayMultisine,
        gen.GaussianProcess,
        gen.RandomWalk,
    ]
    
    critical_gen_classes = [
        gen.Amplitude,
        gen.Cyclic,
        gen.Impulse,
        gen.Resolution,
        gen.PieceWise,
    ]

    data_sets = {
        'train' : [],
        'eval'  : {
            'rand' : [],
            'crit' : [],
        },
    }

    # Generate input sets for train (seed=42)
    for GenClass in random_gen_classes:
        generator = GenClass(config=config, seed=42)
        dataset_name = generator.save_signals(n=200, folder_path=train_dir)
        data_sets['train'].append(dataset_name)

    # Generate input sets for evaluation 1 (seed=56)
    for GenClass in random_gen_classes:
        generator = GenClass(config=config, seed=56)
        dataset_name = generator.save_signals(n=20, folder_path=eval_dir)
        data_sets['eval']['rand'].append(dataset_name)

    # Generate input sets for evaluation 2
    for GenClass in critical_gen_classes:
        generator = GenClass(config=config)
        dataset_name = generator.save_signals(folder_path=eval_dir)
        data_sets['eval']['crit'].append(dataset_name)

    with open(DATA_DIR / 'data_sets.yaml', 'w') as file:
        yaml.dump(data_sets, file, default_flow_style=False, sort_keys=False)

    # Calculate responses for all materials and all input sets
    for mat_name, mat in materials.items():
        for dir in [train_dir, eval_dir]:
            
            state_folder = STATES_DIR / mat_name / dir.name
            out_folder = OUTPUT_DIR / mat_name / dir.name

            state_folder.mkdir(parents=True, exist_ok=True)
            out_folder.mkdir(parents=True, exist_ok=True)
            
            for file in dir.iterdir():
                inp_name = file.stem
                log = f'Response of {mat_name} for {inp_name}'
                
                eps_list = np.load(file, allow_pickle=True)


                sig_list, gamma_list, eps_p_list, alpha_list = [], [], [], []

                for eps in tqdm(eps_list, desc=log):
                    
                    sig, gamma, eps_p, alpha = hardening(
                        eps.astype(np.float32),
                        mat['E'], mat['dalpha'], mat['Y']
                    )
                    
                    sig_list.append(sig)
                    gamma_list.append(gamma)
                    eps_p_list.append(eps_p)
                    alpha_list.append(alpha)


                sig_list = np.array(sig_list, dtype=object)
                gamma_list = np.array(gamma_list, dtype=object)
                eps_p_list = np.array(eps_p_list, dtype=object)
                alpha_list = np.array(alpha_list, dtype=object)

                states_list = np.stack([gamma_list, eps_p_list, alpha_list], axis=-1)

                np.save(
                    out_folder / file.name,
                    sig_list, allow_pickle=True,
                )
                np.save(
                    state_folder / file.name,
                    states_list, allow_pickle=True,
                )

def data_set_names():
    """Load the names of the generated input sets from the yaml file."""
    data_dict_path = DATA_DIR / 'data_sets.yaml'
    if not data_dict_path.exists():
        raise FileNotFoundError(f"{data_dict_path} not found. Please run generate_data() to generate the input sets.")
    
    with open(DATA_DIR / 'data_sets.yaml', 'r') as file:
        return yaml.safe_load(file)
    

def run_grid_search():


    # Observed material and input set
    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'

    # Eval set names
    eval_inp_names = data_set_names()['eval']['crit']
    eval_inp_names += data_set_names()['eval']['crit']

    
    train_set = DataSet(mat_name, 'train', train_inp_name)
    
    eval_sets = [
        DataSet(mat_name, 'eval', inp_name)
        for inp_name in eval_inp_names
    ]
    
    # Trainer config
    trainer = Trainer(
        dataset=train_set, 
        config_path='configs/train.yaml',
    )

    # Evaluator config
    evaluator = Evaluator(datasets=eval_sets)
    
    # Model space
    model_space = [
        ('MLP' , 'dir',  list(product([  2,3,5,8],[2,3,5,8],[  2,3,5,8]))),
        ('MLP' , 'incr', list(product([  2,3,5,8],[2,3,5,8],[  2,3,5,8]))),
        ('LSTM', 'dir',  list(product([0,2,3,5  ],[2,3,5,8],[1,2      ]))),
        ('LSTM', 'incr', list(product([  2,3,5  ],[2,3,5,8],[1,2      ]))),
    ]
    
    num_runs = sum(len(params) for _, _, params in model_space)

    sum_time = 0.0
    count = 0

    for network_name, mode, search_space in model_space:
        for k,p,q in search_space:
            
            tic = time.time()

            # Seek the best k,p,q model by train
            trainer.find_best_model(
                k,p,q,mode,network_name,
                seeds = [42, 56, 17, 83, 29, 64, 90, 11, 75, 38],
            )
            
            model_dir = trainer.save(save_dir=SAVE_DIR)
            evaluator.evaluate(model_dir, overwrite=False)

            toc = time.time()
            sum_time += toc - tic
            count += 1
            avg_time = sum_time / count

            print(
                f"Estimated remaining time ({count} / {num_runs}):", 
                hhmmss(avg_time * (num_runs - count))
            )


def run_cross_material_training(): 
    
    # Model params
    k,p,q,mode,network_name = 3,5,3,'incr','MLP'
    
    # Seeds for random restart
    seeds = [42, 56, 17, 83, 29, 64, 90, 11, 75, 38]    
    
    # Train sets
    train_inp_names = data_set_names()['train']
    
    # Eval sets
    eval_inp_names = data_set_names()['eval']['crit']
    eval_inp_names += data_set_names()['eval']['crit']
    eval_sets = [
        DataSet(mat_name, 'eval', inp_name)
        for inp_name in eval_inp_names
    ]

    # Targets
    mat_names = materials.keys()

    num_runs = len(train_inp_names) * len(mat_names)
    
    sum_time = 0.0
    count = 0

    for mat_name in mat_names:
        for train_inp_name in train_inp_names:
            
            train_set = DataSet(mat_name, 'train', train_inp_name)
            # Trainer config
            trainer = Trainer(
                dataset=train_set,
                config_path='configs/train.yaml',
            )

            # Evaluator config
            evaluator = Evaluator(datasets=eval_sets)
            
            tic = time.time()

            # Seek and train model
            trainer.find_best_model(
                k,p,q,mode,network_name,
                seeds=seeds,
                verbose=False,
            )
            model_dir = trainer.save(save_dir=SAVE_DIR)

            # Eval model
            evaluator.evaluate(model_dir, overwrite=False)

            toc = time.time()
            sum_time += toc - tic
            count += 1
            avg_time = sum_time / count

            print(
                f"Estimated remaining time ({count} / {num_runs}):", 
                hhmmss(avg_time * (num_runs - count))
            )

def eval_all():
    """Iterate through all trained models and evaluate them on all eval sets"""
    
    train_inp_names = data_set_names()['train']
    eval_inp_names = data_set_names()['eval']['rand']
    eval_inp_names += data_set_names()['eval']['crit']
    mat_names = materials.keys()

    for mat_name in mat_names:

        datasets = [
            DataSet(mat_name, inp_type, inp_name)
            for inp_type, names in [('eval', eval_inp_names), ('train', train_inp_names)]
            for inp_name in names
        ]

        for train_inp_name in train_inp_names:
            
            evaluator = Evaluator(datasets=datasets)
            
            model_folder = SAVE_DIR / mat_name / train_inp_name

            for model_dir in model_folder.iterdir():
                
                print(f' ******** Evaluate {model_dir} ********')
                
                evaluator.evaluate(model_dir, overwrite=True, verbose=True)
                (model_dir / 'test_eval.json').unlink(missing_ok=True)
                


if __name__ == '__main__':

    eval_all()

    