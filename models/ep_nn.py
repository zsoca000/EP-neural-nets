import math
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.networks import LSTM, MLP
from models.utils import MinMaxScaler, ErrorMetrics



class EP_NN:
    
    def __init__(self,k,p,q,incr, network: LSTM | MLP,seed=42,):
        
        if incr and not k:
            raise ValueError(f'Incremental model with k=0 cannot be created')
        
        self.k,self.p,self.q=k,p,q
        self.incr = incr
        self.model = MLP(2*k+1,1,p,q,seed=seed) if network == MLP else LSTM(k+1,1,p,q,seed=seed)
        self.seed = seed
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.dy_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.u_scaler = MinMaxScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min'
        )
        self.loss_function = nn.MSELoss()
        self.epochs = 0
        self.train_losses = []
        self.val_losses = []
    

    def calc_batch_size(self, seq_len):
        if isinstance(self.model,LSTM):
            return 1
        elif isinstance(self.model,MLP):
            return seq_len - self.k
        

    def fit(
        self, 
        epochs, # epoch
        y_train,u_train,y_val,u_val, # data
        config_path='train_config.json', # train config path
        verbose=True # logging
    ):
        
        
        # Set the batch size for training
        batch_size = self.calc_batch_size(y_train.shape[1])

        # Load config
        with open(config_path,'r') as f:
            config = json.load(f)

        # Define the optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
        )
        # Define the scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', 
            factor=config['scheduler']['factor'], 
            patience=config['scheduler']['patience'], 
            min_lr=config['scheduler']['min_lr'], 
        )

        # Define the scalers
        self.y_scaler = MinMaxScaler(y_train)
        self.u_scaler = MinMaxScaler(u_train)
        self.dy_scaler = MinMaxScaler(torch.diff(y_train, dim=1))

        # Perprocess the data
        input_train, output_train = self.preprocess(y_train,u_train)
        input_val, output_val = self.preprocess(y_val,u_val)

        # The number of train and validation samples
        n_train = len(input_train)
        n_val = len(input_val)

        # For early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Train
        for epoch in range(epochs):
            
            # Shuffle the data before train
            if config['shuffle']:
                input_train, output_train = self.shuffle(
                    input_train, output_train
                )

            # --- Training Phase ---
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
            
            # --- Validation Phase ---
            self.model.eval()
            val_loss  = 0.0
            with torch.no_grad():
                # Calculate validation result for each epoch
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
                    f'Epoch {epoch + 1:>{len(str(epochs))}}/{epochs} | '
                    f'Loss: {train_loss:.10f} | '
                    f'Val Loss: {val_loss:.10f} | '
                    f'LR: {self.scheduler.get_last_lr()[0]:.6f}'
                )
            
            # Early stopping check
            if "early_stopping" in config:
                if val_loss < best_val_loss - config["early_stopping"]["min_delta"]:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self.model.state_dict()  # Save best weights
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= config["early_stopping"]["patience"]:
                    if verbose: print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)  # Restore best weights
                    break

    
    def shuffle(self, input, output):
        torch.manual_seed(self.seed)
        perm = torch.randperm(input.shape[0])
        return input[perm], output[perm]
        

    def preprocess(self,y,u): 

        y_tmp = self.y_scaler.transform(y) # (n_samples, seq_len, :)
        u_tmp = self.u_scaler.transform(u) # (n_samples, seq_len, :)


        if self.incr:
            # Calc diff
            dy = torch.diff(y, dim=1) # (n_samples, seq_len-1, :)
            dy = self.dy_scaler.transform(dy) # (n_samples, seq_len-1, :) 
            output = dy[:,self.k-1:,:] # (n_samples, seq_len-k, :) TODO: should be (n_samples, seq_len-k, :, 1) 
        else:
            output = y_tmp[:, self.k:, :] # (n_samples, seq_len-k, :) TODO: should be (n_samples, seq_len-k, :, 1) 

        u_next = u_tmp[:, self.k:, :] # (n_samples, seq_len-k, :) 

        # Sliding window
        u_prev = u_tmp.unfold(1, self.k, 1)[:, :-1, 0, :] # (n_samples, seq_len-k, k) TODO: should be (n_samples, seq_len-k, :, k) 
        y_prev = y_tmp.unfold(1, self.k, 1)[:, :-1, 0, :] # (n_samples, seq_len-k, k) TODO: should be (n_samples, seq_len-k, :, k) 

        # u_prev_OLD = torch.tensor(
        #     [[[u[i:self.k+i,:]] for i in range(len(u)-self.k)] for u in u_tmp]
        # )

        # y_prev_OLD = torch.tensor(
        #     [[[y[i:self.k+i,:]] for i in range(len(y)-self.k)] for y in y_tmp]
        # )

        # print(u_prev_OLD.shape, y_prev_OLD.shape)
        # print(u_prev.shape, y_prev.shape)
        # print(torch.any(u_prev == u_prev_OLD), torch.any(y_prev == y_prev_OLD))

        # Design input depending on model type
        if isinstance(self.model, MLP):
            input = torch.cat([y_prev,u_prev,u_next],dim=-1) # (n_samples, seq_len-k, 2k+1) TODO: should be (n_samples, seq_len-k, :, 2k+1)
            input = input.reshape(-1, input.shape[-1]) # (n_samples*(seq_len-k), 2k+1) TODO: should be (n_samples*(seq_len-k), :, 2k+1)
            output = output.reshape(-1, output.shape[-1]) # (n_samples*(seq_len-k), 2k+1) TODO: should be (n_samples*(seq_len-k), :, 2k+1)
        else: 
            input = torch.cat([u_prev,u_next],dim=-1) # (n_samples, seq_len-k, k+1) TODO: should be (n_samples, seq_len-k, :, k+1) 
            
        return input.to(self.device), output.to(self.device)


    def predict(self,y_init,u):
        """
        Get data in original scale
        """
        # Eval mode
        self.model.eval()
        
        # Remove last dim 
        # TODO: what if we originally have 2 features?
        # y_init = y_init[:,:] # (:,k,1)
        # u = u[:,:] # (:,:,1)

        # scale the incoming data
        y_init = self.y_scaler.transform(y_init) # (:,k,:)
        u = self.u_scaler.transform(u) # (:,k,:)
        
        # calculate the time evolution
        y_pred = y_init # (:,k,:)
        for i in range(u.shape[1]-self.k):
            u_prev = u[:,i:i+self.k,:] # (:,k,:)
            u_next = u[:,i+self.k:i+self.k+1,:] # (:,1,:)
            
            # Decide input depending on model type
            if isinstance(self.model, MLP):
                y_prev = y_pred[:,i:i+self.k,:] # (:,k,:)
                input  = torch.cat([y_prev,u_prev,u_next], dim=1) # (:,2k+1,:)
            elif isinstance(self.model, LSTM):
                input = torch.cat([u_prev,u_next], dim=1) # (:,k+1,:)
            
            # Hard to decide what is the feature as seq len became feature
            input = input.transpose(1, 2)  # (:,k+1,:) -> (:,:,k+1) 
            
            
            # Decide output depending on model type
            with torch.inference_mode():
                if self.incr:
                    # prediction
                    dy = self.model(input) # (:,:,1)
                    dy = dy.transpose(1, 2) # (:,1,:)
                    # scale to MPa
                    dy = self.dy_scaler.inverse_transform(dy) # (:,1,:)
                    # Last pred value
                    y_last = y_pred[:,-1:,:] # (:,1,:)
                    # scale to MPa
                    y_last = self.y_scaler.inverse_transform(y_last) # (:,1,:)
                    # step
                    y_next = y_last + dy # (:,1,:)
                    # rescale to rel y scale
                    y_next = self.y_scaler.transform(y_next)  # (:,1,:)
                else:
                    # prediction
                    y = self.model(input) # (:,:,1)
                    y = y.transpose(1, 2) # (:,1,:)
                    y_next = y
            
            y_pred = torch.cat([y_pred,y_next],dim=1)  # (:,k+i,:)
        

        # print('  * input:', input.shape)
        # print('  * y_pred:', y_pred.shape)

        return self.y_scaler.inverse_transform(y_pred) # rescale to MPa 


    def glob_err(self,y_test:torch.tensor,u_test:torch.tensor,save_plot=False,path=''):
        
        # We use just the initial values for inference
        y_init = y_test[:,:self.k,:]  # (n_samples, k, 1)

        if isinstance(self.model, MLP):

            # Tensors to the device for the inference
            y_init = y_init.to(self.device)
            u_test = u_test.to(self.device)
            
            # Inference, by predicting from initial values
            y_pred = self.predict(y_init,u_test).detach().cpu().numpy()
        
        else:
            input_test, _ = self.preprocess(y_test,u_test)

            # Inference
            with torch.inference_mode():
                output_test = self.model(input_test) # (n_samples, seq_len-k, 1)
                output_test = output_test.detach().cpu()
                
            if not self.incr:
                # Rescale to MPa
                y = self.y_scaler.inverse_transform(output_test)
                y_pred = torch.cat([y_init, y], dim=1)
            else:
                y_pred = y_init # (n_samples, k, 1)
                for i in range(output_test.shape[1]): # seq_len-k
                    # Get the i th increment
                    dy = output_test[:,i:i+1,:] # (n_samples, 1, 1)
                    # Rescale to MPa
                    dy = self.dy_scaler.inverse_transform(dy) # (n_samples, 1, 1)
                    y_next = y_pred[:,-1:,:] + dy
                    y_pred = torch.cat([y_pred,y_next],dim=1)

        if save_plot: self.save_eval_plot(y_test, y_pred, path=path)

        # Transform back to the network scale
        y_test = self.y_scaler.transform(y_test)
        y_pred = self.y_scaler.transform(y_pred)
        
        # Return error metrics
        return ErrorMetrics(y_test,y_pred,self.y_scaler)


    def loc_err(self,y_test,u_test):
        # Preprocess
        input_test, output_test_t = self.preprocess(y_test,u_test)
        
        # Inference
        with torch.inference_mode():
            output_test_p = self.model(input_test)

        # Return error metrics
        return ErrorMetrics(
            output_test_t,output_test_p,
            self.dy_scaler if self.incr else self.y_scaler
        )


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
            'epochs': self.epochs,
        }, path)
        print(f'{self.name} saved to {path}!')


    @property
    def name(self):
        return "-".join([
            self.network_name,
            self.output_type_name,
            self.architecture_name,
            str(self.seed)
        ])
        
    @property
    def network_name(self) -> str:
        return self.model.__class__.__name__

    @property
    def output_type_name(self) -> str:
        return 'incr' if self.incr else 'dir'

    @property
    def architecture_name(self) -> str:
        return f"{self.k}-{self.p}-{self.q}"

    @property
    def num_params(self):
        return sum(
            p.numel() 
            for p in self.model.parameters() 
            if p.requires_grad
        )


def load_model(path:str) -> EP_NN:
    
    # Load the checkpoint of the model
    checkpoint = torch.load(
        path, 
        map_location=torch.device('cpu'), 
        weights_only=False,
    )
    
    # Initialize the model
    model = EP_NN(
        checkpoint['k'], checkpoint['p'], checkpoint['q'],
        incr=checkpoint['incr'], 
        network=checkpoint['network'],
        seed=checkpoint['seed'],
    )

    # Reset the params and the history of the model
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


