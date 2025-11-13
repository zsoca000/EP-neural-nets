import torch
import yaml
import json
import torch.nn as nn
from pathlib import Path

from models.models import SeqLSTM, SeqMLP, get_model
from data.materials import load_responses
from sklearn.model_selection import train_test_split

class ModelNotLoadedError(RuntimeError):
    pass


class Trainer:
    
    def __init__(self, mat_name, inp_name, config_path='configs/train.yaml',data_dir='data'):
        
        self.mat_name = mat_name
        self.inp_name = inp_name
        self.data_dir = data_dir
        self.model = None

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
    

    def set_model(self,model:SeqMLP|SeqMLP):
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer']['weight_decay'],
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', 
            factor=self.config['scheduler']['factor'], 
            patience=self.config['scheduler']['patience'], 
            min_lr=self.config['scheduler']['min_lr'], 
        )

        self.loss_function = nn.MSELoss()
        self.epochs = 0
        self.train_losses = []
        self.val_losses = []

      
    def load_data(self):
        
        u_list, y_list = load_responses(
            self.mat_name, 'random', self.inp_name,
            data_dir=self.data_dir
        )

        # Split the incoming data
        y_train, y_tmp, u_train, u_tmp = train_test_split(
            y_list, u_list, test_size=0.3, random_state=42
        )
        y_val, self.y_test, u_val, self.u_test = train_test_split(
            y_tmp, u_tmp, test_size=0.5, random_state=42
        )

        # Fit the model's preprocessor on the training data
        self.model.preprocessor.fit(u_train,y_train)

        # Preprocess the training data
        input_train, output_train = self.model.preprocessor.transform(
            u_train, y_train
        ) # (S, T-k, k*Fy+(k+1)*Fu), (S, T-k, Fy)

        # Preprocess the validation data
        input_val, output_val = self.model.preprocessor.transform(
            u_val,y_val
        ) # (S, T-k, k*Fy+(k+1)*Fu), (S, T-k, Fy)

        # Put them on device
        self.input_train = input_train.to(self.device)
        self.output_train = output_train.to(self.device)
        self.input_val = input_val.to(self.device)
        self.output_val = output_val.to(self.device)


    def train(self, model:SeqMLP|SeqLSTM, epochs, verbose=True):

        # Load the data and the model
        self.set_model(model)
        self.load_data()
        

        # For early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Train
        for epoch in range(epochs):
            
            # Train and val
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            
            # Updates
            self.scheduler.step(val_loss)
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
            if "early_stopping" in self.config:
                if val_loss < best_val_loss - self.config["early_stopping"]["min_delta"]:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self.model.state_dict()  # Save best weights
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.config["early_stopping"]["patience"]:
                    if verbose: print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)  # Restore best weights
                    break
    

    def train_epoch(self):
        
        self.model.train()
        train_loss = 0.0
        n_train = len(self.input_train)
        
        for i in range(n_train):
            
            out_t = self.output_train[i]
            out_p = self.model(self.input_train[i])

            loss = self.loss_function(out_p, out_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / n_train


    @torch.no_grad()
    def val_epoch(self):
        
        self.model.eval()
        n_val = len(self.input_val)
        val_loss  = 0.0

        for i in range(n_val):
        
            out_t = self.output_val[i]
            out_p = self.model(self.input_val[i])
            
            loss = self.loss_function(out_p, out_t)
            val_loss += loss.item()
        
        return val_loss / n_val
    
    
    def load(self,load_path):
        # Load the checkpoint of the model
        state_dict = self.load_state_dict_from_path(load_path)
        self.load_state_dict(state_dict)

    
    @staticmethod
    def load_state_dict_from_path(load_path: str | Path) -> dict:
        return torch.load(
            load_path, 
            map_location=torch.device('cpu'), 
            weights_only=False,
        )

    
    def load_state_dict(self,state_dict):

        # If we use the new implementation, we can check
        if "mat_name" in state_dict and "inp_name" in state_dict:
            
            if self.mat_name != state_dict["mat_name"]:
                raise RuntimeError(
                    f"Trainer was defined for '{self.mat_name}', "
                    f"but tried to load a state for '{state_dict['mat_name']}'."
                )
            if self.inp_name != state_dict["inp_name"]:
                raise RuntimeError(
                    f"Trainer was defined for '{self.inp_name}', "
                    f"but tried to load a state for '{state_dict['inp_name']}'."
                )
        
        # Init the model based on the state dict
        model = self.load_model(state_dict)

        # Set the model
        self.set_model(model)

        # Reload the trainer
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        self.train_losses = state_dict['train_losses'] if 'train_losses' in state_dict else []
        self.val_losses = state_dict['val_losses'] if 'val_losses' in state_dict else []
        self.epochs = state_dict['epochs'] if 'epochs' in state_dict else 0

    
    @staticmethod
    def load_model(state_dict):
        
        # Network architecture
        k,p,q = state_dict['k'], state_dict['p'], state_dict['q']
        seed = state_dict['seed']
        # Mode
        if 'mode' in state_dict:
            # New implementation
            mode = state_dict['mode']
        elif 'incr' in state_dict:
            # Old implementation
            mode = 'incr' if state_dict['incr'] else 'true'
        
        # Network name
        if 'network_name' in state_dict:
            # New implementation
            network_name = state_dict['network_name']
        elif 'network' in state_dict:
            # Old implementation
            network_name = state_dict['network'].__name__
        
        # Init an empty model
        model = get_model(k,p,q,mode,network_name,seed)

        # Reload the model
        model.load_state_dict(state_dict['model_state_dict'])
        model.preprocessor.dy_scaler.load_state_dict(state_dict['dy_scaler'])
        model.preprocessor.y_scaler.load_state_dict(state_dict['y_scaler'])
        model.preprocessor.u_scaler.load_state_dict(state_dict['u_scaler'])

        return model


    def save(self, save_dir='metrics'):
        
        if self.model is None: 
            raise ModelNotLoadedError("Model not loaded. Please load it before try to save.")
        
        save_dir = Path(save_dir, self.mat_name, self.inp_name, self.model.name)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), save_dir / 'model.pth')
        print(f'{self.model.name} saved to {save_dir / 'model.pth'}!')
        
        self.load_data()
        with open(save_dir / 'test_eval.json', 'w') as f:
            json.dump(self.test_eval,f)

        return save_dir

    
    @property
    def test_eval(self):
        return {
            'global' : self.model.glob_err(self.y_test,self.u_test).dictionary,
            'local' : self.model.loc_err(self.y_test,self.u_test).dictionary,
        }


    def state_dict(self):
        return {
            'mat_name': self.mat_name,
            'inp_name': self.inp_name,
            'seed': self.model.seed,
            'incr': self.model.incr,
            'network_name': self.model.network_name,
            'k': self.model.k, 'p': self.model.p, 'q': self.model.q,
            'model_state_dict': self.model.state_dict(),
            'dy_scaler': self.model.preprocessor.dy_scaler.state_dict(),
            'y_scaler': self.model.preprocessor.y_scaler.state_dict(),
            'u_scaler': self.model.preprocessor.u_scaler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
        }
        

    def find_best_model(self, k,p,q, mode, network_name, seeds, verbose=False, check_dir='metrics'):
        
        exists, load_dir = self.exists(k,p,q, mode, network_name, check_dir)

        if exists:
            print(f'{load_dir} already exists - loading instead of training ⚠️')
            self.load(Path(load_dir,'model.pth'))

        else:
            best_test_err = float('inf')
            best_state = None

            for i,seed in enumerate(seeds):
                
                model = get_model(k,p,q,mode,network_name,seed)

                log = (
                    f'Train {model.name} model, '
                    f'for {self.mat_name} on {self.inp_name} '
                    f'{i+1}/{len(seeds)} 🔄'
                )
                print(log, end='\r')
                
                self.train(
                    model,
                    epochs=500, # hardcoded
                    verbose=verbose,
                )

                test_err = model.glob_err(self.y_test,self.u_test)
                test_err = test_err.MSE_rel

                log = log.replace(
                    '🔄', f'✅ - Test Error: {test_err:.8f}'
                )
                print(log, end='\n')

                if best_test_err > test_err:
                    best_test_err = test_err
                    best_state = self.state_dict()
            
            self.load_state_dict(best_state)

    
    def exists(self, k,p,q, mode, network_name, check_dir='metrics'):

        parent_folder = Path(check_dir ,self.mat_name, self.inp_name)
        
        # Try to find file with: net_name-mode-k-p-q
        prefix = f'{network_name}-{mode}-{k}-{p}-{q}'
        
        folder_path = None
        for f in parent_folder.iterdir():
            if f.is_dir() and f.name.startswith(prefix):
                folder_path = f
                break

        return folder_path is not None, folder_path


    
if __name__ == '__main__':
    
    trainer = Trainer('isotropic-swift','pd_ms_42_200')
    
    trainer.find_best_model(
        5,2,2,'dir','MLP',
        seeds=[42, 56, 17, 83, 29],
        check_dir='tmp2'
    )

    trainer.save(save_dir='tmp2')

