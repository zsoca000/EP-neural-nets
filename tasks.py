import time
from itertools import product

from train import Trainer
from eval import Evaluator
from utils.time import hhmmss


def task1():

    # Trainer config
    trainer = Trainer(
        mat_name='isotropic-swift', 
        inp_name='pd_ms_42_200', 
        config_path='configs/train.yaml',
    )

    # Evaluator config
    evaluator = Evaluator(
        mat_name='isotropic-swift',
        eval_sets = {
            'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
            'random': ['bl_ms_42_200','combined_42_200','gp_42_200','pd_ms_42_200','rw_42_200']
        }
    )
    
    # Model space
    model_space = [
        ('MLP' , 'dir', list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        ('MLP' , 'incr' , list(product([2,3,5,8],[2,3,5,8],[2,3,5,8]))),
        ('LSTM', 'dir', list(product([0,2,3,5],[2,3,5,8],[1,2]))),
        ('LSTM', 'incr', list(product([2,3,5],[2,3,5,8],[1,2]))),
    ]
    
    num_runs = sum(len(params) for _, _, params in model_space)

    sum_time = 0.0
    count = 0

    for network_name, mode, search_space in model_space:
        for k,p,q in search_space:
            
            tic = time.time()

            # Seek and train model
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
    k,p,q,mode,network_name = 8,5,3,'incr','dir'

    # Training sets
    train_inp_names = [
        'bl_ms_42_200',
        'combined_42_200',
        'gp_42_200',
        'pd_ms_42_200',
        'rw_42_200'
    ]

    # Material behaviour (known model)
    mat_names = [
        'isotropic-swift',
        'isotropic-linear',
        'kinematic-linear',
        'kinematic-armstrong-fredrick',
        'mixed-linear',
        'mixed-armstrong-fredrick'
    ]

    num_runs = len(train_inp_names) * len(mat_names)

    for mat_name in mat_names:
        for train_inp_name in train_inp_names:
            
            
            # Trainer config
            trainer = Trainer(
                mat_name=mat_name, 
                inp_name=train_inp_name, 
                config_path='models/train_config.json',
            )

            # Evaluator config
            evaluator = Evaluator(
                mat_name=mat_name,
                eval_sets = {
                    'static' : ['amplitude','cyclic','impulse','piecewise','resolution'],
                    'random': train_inp_name,
                }
            )
            
            tic = time.time()

            # Seek and train model
            trainer.find_best_model(k,p,q,mode,network_name)
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



if __name__ == '__main__':
    
    task1()
    task2()


            


