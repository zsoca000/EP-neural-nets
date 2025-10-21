from models.MLP_dir import MLP_dir
from models.MLP_incr import MLP_incr
from models.LSTM_dir import LSTM_dir
from data.materials import load_responses
from models.commons import data_to_tensor
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os.path as osp


def train_model(model: MLP_dir | MLP_incr | LSTM_dir, mat_name, inp_name): 
    eps_list, sig_list = load_responses(mat_name,'random',inp_name,data_path='data')

    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)

    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

    model.fit(
        epochs=500,
        y_train=y_train, u_train=u_train,
        y_val=y_val,u_val=u_val,
        batch_size=1,
        early_stopping=True,
        patience=20,
        min_delta=1e-6,
        verbose=True,
        shuffle=False,
    )

    return model.evaluate(y_test,u_test).item()


def eval_model(model: MLP_dir | MLP_incr | LSTM_dir, mat_name, inp_type, inp_name, save_plot=False):
    
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


if __name__ == "__main__":
    
    mat_name = 'isotropic-swift'
    train_inp_name = 'pd_ms_42_200'
    
    # model = MLP_dir(2,8,3,seed=42)
    model = LSTM_dir(1,6,1,seed=42)

    test_err = train_model(
        model,
        mat_name = mat_name,
        inp_name = train_inp_name,
    )

    print(f"{test_err*100} %")


    eval_list = []
    for inp_name in ['amplitude', 'cyclic', 'impulse', 'piecewise']:
        eval_list += [
            eval_model(model,mat_name,'static',inp_name,save_plot=True)
        ]
        print(f'{inp_name}: {eval_list[-1]*100:02f} %')

    for inp_name in ['bl_ms_42_200', 'gp_42_200', 'pd_ms_42_200', 'rw_42_200']:
        eval_list += [
            eval_model(model,mat_name,'random',inp_name,save_plot=False)
        ]
        print(f'{inp_name}: {eval_list[-1]*100:02f} %')

    print(f'\nMean: {np.mean(eval_list)*100:02f} %')