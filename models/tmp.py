import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.commons import (
    LSTM, MLP, MinMaxScaler, data_to_tensor
)
from sklearn.model_selection import train_test_split
from data.materials import load_responses

class EP_NN:
    
    def __init__(self,k,p,q,incr,network: LSTM | MLP,seed=42):
        self.k,self.p,self.q=k,p,q
        self.incr = incr
        self.model = MLP(2*k+1,1,p,q) if network == MLP else LSTM(k+1,1,p,q)
        self.seed = seed
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', 
            factor=0.5, patience=7, 
            min_lr=1e-6
        )
        self.dy_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.u_scaler = MinMaxScaler()
        self.loss_function = nn.MSELoss()
        self.epochs = 0
        self.train_losses = []
        self.val_losses = []
    

    def fit(
        self,epochs, # epoch
        y_train,u_train,y_val,u_val, # data
        batch_size=32,
        early_stopping=False, patience=15, min_delta=0.0, # early stopping
        shuffle=False,
        verbose=True
    ):
        
        self.y_scaler = MinMaxScaler(y_train)
        self.u_scaler = MinMaxScaler(u_train)
        self.dy_scaler = MinMaxScaler(torch.diff(y_train, dim=1))

        input_train, output_train = self.preprocess(
            y_train,u_train,shuffle=shuffle
        )
        input_val, output_val = self.preprocess(
            y_val,u_val,shuffle=shuffle
        )

        n_train = len(input_train)
        n_val = len(input_val)

        # For early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            
            # Training Phase
            self.model.train()
            train_loss = 0.0
            for i in range(0, n_train, batch_size):
                
                batch_in = input_train[i:i+batch_size]
                batch_out_t = output_train[i:i+batch_size]
                batch_out_p = self.model(batch_in)

                loss = self.loss_function(batch_out_p, batch_out_t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= (n_train / batch_size)
            
            # Validation Phase
            self.model.eval()
            val_loss  = 0.0
            with torch.no_grad():
                for i in range(0, n_val, batch_size):
                   
                    batch_in = input_val[i:i + batch_size]
                    batch_out_t = output_val[i:i + batch_size]
                    batch_out_p = self.model(batch_in)
                    
                    val_loss += self.loss_function(batch_out_p, batch_out_t).item()
        
            val_loss /= (n_val / batch_size)
            self.scheduler.step(val_loss)

            # Update training history
            self.epochs += 1
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Logging
            if verbose: 
                print(
                    f'Epoch {epoch+1}/{epochs},', 
                    f'Loss: {train_loss},',
                    f'Val Loss: {val_loss},', 
                    f'LR: {self.scheduler.get_last_lr()[0]}'
                )
            
            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self.model.state_dict()  # Save best weights
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)  # Restore best weights
                    break

        
    def preprocess(self,y,u,shuffle=False):
        seq_len = u.shape[1]
        
        y = self.y_scaler.transform(y)
        u = self.u_scaler.transform(u)
        
        u_prev = u.unfold(1, self.k, 1)[:, :-1, 0, :]
        y_prev = y.unfold(1, self.k, 1)[:, :-1, 0, :]

        u_next = u[:, self.k:seq_len, 0:1]
        y_next = y[:, self.k:seq_len, 0:1]

        # Design output depending on model type
        if self.incr: 
            dy = self.dy_scaler.transform(torch.diff(y, dim=1))
            dy_next = dy[:, self.k-1:seq_len-1, 0:1]
            output = dy_next
        else:
            output = y_next

        # Design input depending on model type
        if isinstance(self.model, MLP):
            input = torch.cat([y_prev,u_prev,u_next],axis=-1)
        else: 
            input = torch.cat([u_prev,u_next],axis=-1)

        if shuffle:
            torch.manual_seed(self.seed)
            perm = torch.randperm(input.shape[0])
            input, output = input[perm], output[perm]
        
        return input.to(self.device), output.to(self.device)


    def predict(self,y0,u):
        # Eval mode
        self.model.eval()
        
        # scale the incoming data
        y0 = self.y_scaler.transform(y0)
        u = self.u_scaler.transform(u)
    
        # calculate the time evolution
        y_pred = y0
        for i in range(u.shape[1]-self.k):
            u_prev = u[:,i:i+self.k]
            u_next = u[:,i+self.k].unsqueeze(-1)
            
            # Decide input depending on model type
            if isinstance(self.model, MLP):
                y_prev = y_pred[:,i:i+self.k]
                input  = torch.cat([y_prev,u_prev,u_next],axis=-1)
            else:
                input = torch.cat([u_prev,u_next],axis=-1)

            # Decide output depending on model type
            with torch.inference_mode():
                if self.incr:
                    dy = self.model(input)
                    dy = self.dy_scaler.inverse_transform(dy)
                    y_last = self.y_scaler.inverse_transform(
                        y_pred[:,-1].unsqueeze(-1)
                    )
                    y_next = self.y_scaler.transform(y_last + dy)
                else:
                    y_next = self.model(input)
            y_pred = torch.cat([y_pred,y_next],axis=-1)
        
        # rescale the output
        return self.y_scaler.inverse_transform(y_pred)


    def evaluate(self,y_test,u_test,save_plot=False,path=''):
        y_pred = self.predict(
            y_test[:,:self.k,0], # y0
            u_test[:,:,0] # u
        ).detach().cpu().numpy()
        y_true = y_test.detach().squeeze(-1).cpu().numpy()
        if save_plot:
            self.save_eval_plot(y_true,y_pred,path)
        return ((
            self.y_scaler.transform(y_true) - \
            self.y_scaler.transform(y_pred))**2
        ).mean()


    def save_eval_plot(self, y_true, y_pred, path):

        n = y_true.shape[0]
        cols = min(5, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).flatten()

        for i, ax in enumerate(axes):
            if i < n:
                y_t = y_true[i].squeeze()
                y_p = y_pred[i].squeeze()
                error = ((self.y_scaler.transform(y_t) - self.y_scaler.transform(y_p))**2).mean()
                ax.plot(y_t / 1e6, color='black', lw=3)
                ax.plot(y_p / 1e6, color='red', lw=2, ls='--')
                ax.set_title(f'MSE = {error:.2e}', fontsize=8)
            else:
                ax.axis('off')

        total_error = ((self.y_scaler.transform(y_true) - self.y_scaler.transform(y_pred))**2).mean()
        plt.suptitle(f'MSE = {total_error:.2e}', fontsize=14)
        # plt.tight_layout()
        plt.savefig(path)
        plt.close()


    def save(self, path):
        torch.save({
            'seed': self.seed,
            'incr': self.incr,
            'network': MLP if isinstance(self.model, MLP) else LSTM,
            'k': self.k, 'p': self.p, 'q': self.q,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'dy_scaler': self.dy_scaler.state_dict(),
            'y_scaler': self.y_scaler.state_dict(),
            'u_scaler': self.u_scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
        }, path)
        print(f'Model saved to {path}!')


def load_model(path):
    checkpoint = torch.load(path)
    model = EP_NN(
        checkpoint['k'], checkpoint['p'], checkpoint['q'],
        incr=checkpoint['incr'], 
        network=checkpoint['network'],
        seed=checkpoint['seed'],
    )
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.dy_scaler.load_state_dict(checkpoint['dy_scaler'])
    model.y_scaler.load_state_dict(checkpoint['y_scaler'])
    model.u_scaler.load_state_dict(checkpoint['u_scaler'])
    model.train_losses = checkpoint['train_losses'] if 'train_losses' in checkpoint else []
    model.val_losses = checkpoint['val_losses'] if 'val_losses' in checkpoint else []
    model.epochs = checkpoint['epochs'] if 'epochs' in checkpoint else 0
    return model


if __name__ == "__main__":

    mat_name = 'isotropic-swift'
    inp_name = 'pd_ms_42_200'

    eps_list, sig_list = load_responses(mat_name,'random',inp_name,data_path='data')

    y_list = data_to_tensor(sig_list)
    u_list = data_to_tensor(eps_list)

    y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
    y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)


    model = EP_NN(2,8,3, incr=False, network=MLP, seed=42)


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


    