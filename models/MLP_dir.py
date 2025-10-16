import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data.plastic import materials
from commons import MLP, MinMaxScaler


class MLP_dir(MLP):
    def __init__(self,k,p,q):
        super(MLP_dir, self).__init__(2*k+1,1,p,q)
        self.k,self.p,self.q=k,p,q
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0
        
    def fit(
        self,epochs,y_train,u_train,y_val,u_val, 
        early_stopping=False, patience=10, min_delta=0.0, verbose=True
    ):
        
        # define the scalers
        self.y_scaler = MinMaxScaler(y_train)
        self.u_scaler = MinMaxScaler(u_train)


        input_train, output_train = self.preprocess(y_train,u_train)
        input_val, output_val = self.preprocess(y_val,u_val)


        for epoch in range(epochs):
            # Training Phase
            self.train()
            for input, output_true in zip(input_train,output_train):
                output_pred = self(input)
                loss = self.loss_function(output_pred, output_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Validation Phase
            self.eval()
            val_loss  = 0
            with torch.no_grad():
                for input, output_true in zip(input_val,output_val):
                    val_outputs = self(input)
                    val_loss += self.loss_function(val_outputs, output_true).item()
            val_loss /= input_val.shape[0]
            
            self.epochs += 1
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')

    def predict(self,y0,u):
        
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
            y_next = self(input)
            y_pred = torch.cat([y_pred,y_next],axis=-1)
        
        # rescale the output
        return self.y_scaler.inverse_transform(y_pred)

    def preprocess(self,y,u):

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

        return input, output

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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
    model.load_state_dict(checkpoint['model_state_dict'])
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


if __name__ == '__main__':
    
    k,p,q = 2,8,2
    epochs = 150
    
    for name in materials.keys():
    
        data_path = osp.join("data","data-sets",name)

        y_list = torch.tensor(np.load(osp.join(data_path,'y_list.npy')),dtype=torch.float32).unsqueeze(-1)
        u_list = torch.tensor(np.load(osp.join(data_path,'u_list.npy')),dtype=torch.float32).unsqueeze(-1)

        y_train, y_tmp, u_train, u_tmp = train_test_split(y_list, u_list, test_size=0.3, random_state=42)
        y_val, y_test, u_val, u_test = train_test_split(y_tmp, u_tmp, test_size=0.5, random_state=42)

        model = MLP_dir(k,p,q)
        model.fit(epochs,y_train,u_train,y_val,u_val)
        
        # SAVE
        model_path = osp.join('metrics','models',f'MLP_dir_{k}-{p}-{q}',name)
        if not osp.exists(model_path): 
            os.makedirs(model_path)
        
        model.save_eval_plot(
            y_test,u_test,
            path=osp.join(model_path,f'{epochs}.png')
        )
        model.save_model(osp.join(model_path,f'{epochs}.pth'))
    

    # video_from_folder('frames',fps=15)
    
    


    
    