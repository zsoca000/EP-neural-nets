import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.commons import MLP, MinMaxScaler


class MLP_dir:
    def __init__(self,k,p,q,seed=42,weight_decay=1e-2):
        self.k,self.p,self.q=k,p,q
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.model = MLP(2*k+1,1,p,q,seed=self.seed).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', 
            factor=0.5, patience=7, 
            min_lr=1e-6
        )
        self.loss_function = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0

    def fit(
        self,epochs, # epoch
        y_train,u_train,y_val,u_val, # data
        early_stopping=False, patience=15, min_delta=0.0, # early stopping
        shuffle=False,
        verbose=True
    ):
        
        # define the scalers
        self.y_scaler = MinMaxScaler(y_train)
        self.u_scaler = MinMaxScaler(u_train)

        input_train, output_train = self.preprocess(
            y_train,u_train,shuffle=shuffle
        )
        input_val, output_val = self.preprocess(
            y_val,u_val,shuffle=shuffle
        )

        # For early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            for input, output_true in zip(input_train,output_train):
                output_pred = self.model(input)
                loss = self.loss_function(output_pred, output_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Validation Phase
            self.model.eval()
            val_loss  = 0
            with torch.no_grad():
                for input, output_true in zip(input_val,output_val):
                    val_outputs = self.model(input)
                    val_loss += self.loss_function(val_outputs, output_true).item()
            val_loss /= input_val.shape[0]
            self.scheduler.step(val_loss)

            # Update training history
            self.epochs += 1
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)

            if verbose: 
                print(
                    f'Epoch {epoch+1}/{epochs},', 
                    f'Loss: {loss.item()},',
                    f'Val Loss: {val_loss},', 
                    f'LR: {self.scheduler.get_last_lr()[0]}'
                )
            
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


    def predict(self,y0,u):
        self.model.eval()
        # scale the incoming data
        y0 = self.y_scaler.transform(y0)
        u = self.u_scaler.transform(u)
    
        # calculate the time evolution
        y_pred = y0
        for i in range(u.shape[1]-self.k):
            u_prev = u[:,i:i+self.k]
            u_next = u[:,i+self.k].unsqueeze(-1)
            y_prev = y_pred[:,i:i+self.k]
            input  = torch.cat([y_prev,u_prev,u_next],axis=-1)
            with torch.inference_mode():
                y_next = self.model(input)
            y_pred = torch.cat([y_pred,y_next],axis=-1)
        
        # rescale the output
        return self.y_scaler.inverse_transform(y_pred)

    def preprocess(self,y,u,shuffle=False):

        assert y.shape[1] == u.shape[1]

        y = self.y_scaler.transform(y)
        u = self.u_scaler.transform(u)

        seq_len = u.shape[1]

        u_prev = torch.stack([u[:, i:i+self.k, 0] for i in range(seq_len-self.k)], axis=-1)
        u_prev = np.transpose(u_prev, (0, 2, 1))

        u_next = torch.stack([u[:, i+self.k, 0] for i in range(seq_len-self.k)], axis=-1)
        u_next = u_next.unsqueeze(-1)   

        y_prev = torch.stack([y[:, i:i+self.k, 0] for i in range(seq_len-self.k)], axis=-1)
        y_prev = np.transpose(y_prev, (0, 2, 1))

        y_next = torch.stack([y[:, i+self.k, 0] for i in range(seq_len-self.k)], axis=-1)
        y_next = y_next.unsqueeze(-1)

        input = torch.cat([y_prev,u_prev,u_next],axis=-1)
        output = y_next
        if shuffle:
            torch.manual_seed(self.seed)
            perm = torch.randperm(input.shape[0])
            input, output = input[perm], output[perm]
        
        return input.to(self.device), output.to(self.device)

    def evaluate(self,y_test,u_test):
        y_pred = self.predict(y_test[:,:self.k,0],u_test[:,:,0])
        y_true = y_test.squeeze()
        return ((self.y_scaler.transform(y_true) - self.y_scaler.transform(y_pred))**2).mean()

    def save_eval_plot(self,y_test,u_test,path):
        y_pred = self.predict(y_test[:,:self.k,0],u_test[:,:,0])
        y_true = y_test.squeeze()

        fig, axes = plt.subplots(6,5,figsize=(12,12),sharex=True,sharey=True)
        
        for i,ax in enumerate(axes.flatten()):
            error = ((self.y_scaler.transform(y_true[i]) - self.y_scaler.transform(y_pred[i]))**2).mean()
            ax.plot(y_true[i]/1e6,color='black',lw=3)
            ax.plot(y_pred[i].detach().numpy()/1e6,color='red',lw=2,ls='--')
            ax.set_title(f'MSE = {error}',fontsize=8)
            

        error = ((self.y_scaler.transform(y_true) - self.y_scaler.transform(y_pred))**2).mean()
        
        plt.suptitle(f'MSE = {error}', fontsize=16)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'y_scaler': {
                'x_min': self.y_scaler.x_min,
                'x_max': self.y_scaler.x_max
            } if self.y_scaler else None,
            'u_scaler': {
                'x_min': self.u_scaler.x_min,
                'x_max': self.u_scaler.x_max
            } if self.u_scaler else None,
            'k': self.k,
            'p': self.p,
            'q': self.q,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
        }, path)
        print(f'Model saved to {path}!')


def load_model(path):
    checkpoint = torch.load(path)
    model = MLP_dir(checkpoint['k'], checkpoint['p'], checkpoint['q'])
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Reconstruct scalers
    if checkpoint['y_scaler'] is not None:
        model.y_scaler = MinMaxScaler.__new__(MinMaxScaler)
        model.y_scaler.x_min = checkpoint['y_scaler']['x_min']
        model.y_scaler.x_max = checkpoint['y_scaler']['x_max']
    else:
        model.y_scaler = None
    if checkpoint['u_scaler'] is not None:
        model.u_scaler = MinMaxScaler.__new__(MinMaxScaler)
        model.u_scaler.x_min = checkpoint['u_scaler']['x_min']
        model.u_scaler.x_max = checkpoint['u_scaler']['x_max']
    else:
        model.u_scaler = None
    model.train_losses = checkpoint['train_losses'] if 'train_losses' in checkpoint else []
    model.val_losses = checkpoint['val_losses'] if 'val_losses' in checkpoint else []
    model.epochs = checkpoint['epochs'] if 'epochs' in checkpoint else 0
    return model

    
    
    


    
    