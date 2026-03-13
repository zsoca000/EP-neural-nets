import time
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import product

import data.generators as gen
from data.materials import hardening
from configs.materials import materials

from train import Trainer
from eval import Evaluator
from utils.time import hhmmss


ROOT_DIR = Path("")

DATA_DIR = ROOT_DIR / 'data'
CONFIG_DIR = ROOT_DIR / 'configs'

INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"


def task0(): # Generate input signals
    
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
        gen.Combined,
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
            for file in dir.iterdir():
                inp_name = file.stem
                log = f'Response of {mat_name} for {inp_name}'
                
                eps_list = np.load(file, allow_pickle=True)

                sig_list = np.array([
                    hardening(
                        eps.astype(np.float32),
                        mat['E'], mat['dalpha'], mat['Y']
                    )[0]
                    for eps in tqdm(eps_list, desc=log)
                ], dtype=object)

                save_folder = OUTPUT_DIR / mat_name / dir.name
                save_folder.mkdir(parents=True, exist_ok=True)

                np.save(
                    save_folder / file.name,
                    sig_list,
                    allow_pickle=True,
                )

def data_set_names(): # Load the names of the generated input sets
    data_dict_path = DATA_DIR / 'data_sets.yaml'
    if not data_dict_path.exists():
        raise FileNotFoundError(f"{data_dict_path} not found. Please run task0() to generate the input sets.")
    
    with open(DATA_DIR / 'data_sets.yaml', 'r') as file:
        return yaml.safe_load(file)
    

def task1(): # Parameter sweeping

    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'
    eval_inp_names = data_set_names()['eval']['crit']
    eval_inp_names += data_set_names()['eval']['crit']

    # Trainer config
    trainer = Trainer(
        mat_name=mat_name, 
        inp_name=train_inp_name, 
        config_path='configs/train.yaml',
    )

    # Evaluator config
    evaluator = Evaluator(
        mat_name=mat_name,
        inp_names=eval_inp_names,
    )
    
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
            
            model_dir = trainer.save(save_dir='metrics')
            evaluator.evaluate(model_dir, overwrite=False)

            toc = time.time()
            sum_time += toc - tic
            count += 1
            avg_time = sum_time / count

            print(
                f"Estimated remaining time ({count} / {num_runs}):", 
                hhmmss(avg_time * (num_runs - count))
            )


def task2():
    
    # Model params
    k,p,q,mode,network_name = 3,5,3,'incr','MLP'
    
    # Seeds for random restart
    seeds = [42, 56, 17, 83, 29, 64, 90, 11, 75, 38]    
    
    # Train sets
    train_inp_names = data_set_names()['train']
    
    # Eval sets
    eval_inp_names = data_set_names()['eval']['crit']
    eval_inp_names += data_set_names()['eval']['crit']
    
    # Targets
    mat_names = materials.keys()

    num_runs = len(train_inp_names) * len(mat_names)
    
    sum_time = 0.0
    count = 0

    for mat_name in mat_names:
        for train_inp_name in train_inp_names:
            
            # Trainer config
            trainer = Trainer(
                mat_name=mat_name, 
                inp_name=train_inp_name, 
                config_path='configs/train.yaml',
            )

            # Evaluator config
            evaluator = Evaluator(
                mat_name=mat_name,
                inp_names=eval_inp_names,
            )
            
            tic = time.time()

            # Seek and train model
            trainer.find_best_model(
                k,p,q,mode,network_name,
                seeds=seeds,
                verbose=False,
            )
            model_dir = trainer.save(save_dir='metrics')

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
    
    train_inp_names = data_set_names()['train']
    eval_inp_names = data_set_names()['eval']
    eval_inp_names += data_set_names()['eval']
    mat_names = materials.keys()


    for mat_name in mat_names:
        for train_inp_name in train_inp_names:
            
            evaluator = Evaluator(
                mat_name=mat_name,
                inp_names= eval_inp_names,
            )
            
            model_folder = Path('metrics',mat_name,train_inp_name)

            for model_dir in model_folder.iterdir():
                
                print(f' ******** Evaluate {model_dir} ********')
                
                evaluator.evaluate(model_dir, overwrite=True, verbose=True)
                (model_dir / 'test_eval.json').unlink(missing_ok=True)
                

def check_change():
    # CHECK OLD VS NEW INPUTS
    input_dir = Path("data", "input")

    train_input_dir = input_dir / 'train'
    eval_input_dir = input_dir / 'eval'
    random_input_dir = input_dir / 'random'
    static_input_dir = input_dir / 'static'

    input_changes = []
    for file in random_input_dir.iterdir():
        eps_list1 = np.load(file, allow_pickle=True)
        eps_list2 = np.load(random_input_dir / file.name, allow_pickle=True)
        input_changes += [np.any(eps_list1 != eps_list2)]

    for file in static_input_dir.iterdir():
        eps_list1 = np.load(file, allow_pickle=True)
        eps_list2 = np.load(eval_input_dir / file.name, allow_pickle=True)
        input_changes += [np.any(eps_list1 != eps_list2)]
    
    print(input_changes)

    # CHECK OLD VS NEW OUTPUTS
    output_dir = Path("data", "output")
    material_names = materials.keys()
    
    output_changes = []
    for mat_name in material_names:
        train_output_dir = output_dir / mat_name / 'train'
        eval_output_dir = output_dir / mat_name / 'eval'
        random_output_dir = output_dir / mat_name / 'random'
        static_output_dir = output_dir / mat_name / 'static'

        for file in random_input_dir.iterdir():
            eps_list1 = np.load(file, allow_pickle=True)
            eps_list2 = np.load(random_input_dir / file.name, allow_pickle=True)
            output_changes += [np.any(eps_list1 != eps_list2)]

        for file in static_input_dir.iterdir():
            eps_list1 = np.load(file, allow_pickle=True)
            eps_list2 = np.load(eval_input_dir / file.name, allow_pickle=True)
            output_changes += [np.any(eps_list1 != eps_list2)]

    print(output_changes)



if __name__ == '__main__':
    
    task0()

    