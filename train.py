from models.MLP_incr import load_model, MLP_incr
import torch
import os
import os.path as osp
from sklearn.model_selection import train_test_split
import numpy as np
from data.plastic import materials


def copy_model(model_path):
    model_old = load_model(model_path)

    model_new = MLP_incr(model_old.k,model_old.p,model_old.q)
    model_new.load_state_dict(model_old.state_dict())

    return model_new

def prepare_data(name):
    data_path = osp.join("data","data-sets",name)
    y_list = torch.tensor(np.load(osp.join(data_path,'y_list.npy')),dtype=torch.float32).unsqueeze(-1)
    u_list = torch.tensor(np.load(osp.join(data_path,'u_list.npy')),dtype=torch.float32).unsqueeze(-1)
    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

    u_bench = torch.tensor(np.load(osp.join(data_path,'u_benchmark.npy')),dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
    y_bench = torch.tensor(np.load(osp.join(data_path,'y_benchmark.npy')),dtype=torch.float32).unsqueeze(-1).unsqueeze(0)
    
    Y = (y_train, y_val, y_test, y_bench)
    U = (u_train, u_val, u_test, u_bench)
    
    return Y, U

def train_one(epochs,k,p,q,name_list):
    for name in name_list:
        
        Y, U = prepare_data(name)
        y_train, y_val, y_test, y_bench = Y
        u_train, u_val, u_test, u_bench = U

        model = MLP_incr(k,p,q)
        
        model.fit(epochs,y_train,u_train,y_val,u_val)

        model_path = osp.join('metrics','models',f'MLP_incr_{k}-{p}-{q}',name)
        if not osp.exists(model_path): 
            os.makedirs(model_path)
        
        model.save_eval_plot(
            y_test,u_test,
            path=osp.join(model_path,f'{epochs}.png')
        )
        model.save_model(osp.join(model_path,f'{epochs}.pth'))

def train_fine_tune(best_model_path, epochs, name_list):
    for name in name_list:
    
        Y, U = prepare_data(name)
        y_train, y_val, y_test, y_bench = Y
        u_train, u_val, u_test, u_bench = U

        # empty copy of the best model
        model = copy_model(best_model_path)

        model.fit(epochs,y_train,u_train,y_val,u_val)

        model_path = osp.join('metrics','models',f'MLP_incr_tuned_{model.k}-{model.p}-{model.q}',name)
        if not osp.exists(model_path): 
            os.makedirs(model_path)
        
        model.save_eval_plot(
            y_test,u_test,
            path=osp.join(model_path,f'{epochs}.png')
        )
        model.save_model(osp.join(model_path,f'{epochs}.pth'))

def train_multiple(n,epochs,k,p,q,name_list):
    for name in name_list:
    
        Y, U = prepare_data(name)
        y_train, y_val, y_test, y_bench = Y
        u_train, u_val, u_test, u_bench = U


        # initialize and train multiple times
        best_avg_error = float('inf')
        best_model = None
        for i in range(n):
            print(f'Training the {i+1}. {name} model!')
            model_tmp = MLP_incr(k,p,q)
            model_tmp.fit(
                epochs,y_train,u_train,
                y_val,u_val,
                early_stopping=True,
                patience=20,
                min_delta=1e-5,
                verbose=False
            )

            test_error = model_tmp.evaluate(y_test,u_test)
            bench_error = model_tmp.evaluate(y_bench,u_bench)
            avg_error = 0.5 * (test_error + bench_error)

            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_model = model_tmp

        
        model_path = osp.join('metrics','models',f'MLP_incr_rand_{best_model.k}-{best_model.p}-{best_model.q}',name)
        
        if not osp.exists(model_path):
            os.makedirs(model_path)

        best_model.save_eval_plot(
            y_test, u_test,
            path=osp.join(model_path, 'model.png')
        )
        best_model.save_model(osp.join(model_path, 'model.pth'))


if __name__ == '__main__':

    # best_model_path = osp.join('metrics','models','MLP_incr_7-8-3','kinematic-linear','150.pth')

    epochs = 200
    n = 50
    k,p,q = 7,8,3

    train_multiple(n,epochs,k,p,q,materials)