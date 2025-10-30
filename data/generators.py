import numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def load_params(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


class BaselineMultisine:

    def __init__(self,config_path,seed=42):
        self.seed = seed
        self.params = load_params(config_path)
        self.rng = np.random.default_rng(seed)
        


    def pick(self,param,size=1):
        if isinstance(param, list):
            return self.rng.uniform(param[0], param[1], size=size)
        return param

    
    def generate_t(self):
        dt = float(self.pick(self.params['general']['dt']))
        t_end = float(self.pick(self.params['general']['t_end']))
        return np.arange(0, t_end, dt)


    def generate_baseline(self,t):
        t_step = self.pick(self.params['baseline']['t_step'])
        step_times = np.arange(0, t[-1] + t_step, t_step)
        step_amplitudes = self.pick(self.params['baseline']['amplitude'],size=len(step_times))
        return np.interp(t, step_times, step_amplitudes)

    
    def generate_multisine(self, t):
        
        freq_min = float(self.pick(self.params['multisine']['freq_min']))
        freq_max = float(self.pick(self.params['multisine']['freq_max']))
        n_freqs = int(self.pick(self.params['multisine']['freq_max']))
        amplitudes = self.pick(self.params['multisine']['amplitude'],size=n_freqs)
        phases = self.pick(self.params['multisine']['phase'],size=n_freqs)

        freqs = np.linspace(freq_min, freq_max, n_freqs)

        multisine = np.zeros_like(t)
        for i in range(n_freqs):
            multisine += amplitudes[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

        return multisine

    @property
    def signal(self):
        t = self.generate_t()
        baseline = self.generate_baseline(t)
        multisine = self.generate_multisine(t)
        sgn = baseline + multisine
        return sgn - sgn[0]


    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'bl_ms_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )
            

class PowerDecayMultisine:
    
    def __init__(self,config_path,seed=42):
        self.seed = seed
        self.params = load_params(config_path)
        self.rng = np.random.default_rng(seed)


    @property
    def signal(self): 

        T = self.params['len_seq']
        N = self.params['n_freqs']
        p = self.params['power']
        u_max = self.params['amplitude']

        k = np.linspace(0, T, T)  
        a = self.rng.standard_normal(N)
        u = np.zeros_like(k)
        for n in range(1, N + 1):
            u += (a[n - 1] / n**p) * np.sin(2*np.pi*n*k/N)
        u = np.array(u)
        u = u/(np.max(u) - np.min(u))*u_max
        return u
    

    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'pd_ms_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )


class GaussianProcess:
    def __init__(self, config_path, seed=42):
        self.seed = seed
        self.params = load_params(config_path)
        self.rng = np.random.default_rng(seed)


    
    def get_random_points(self):
        n_cp = int(self.params['n_cp'])
        amplitude = float(self.params['amplitude'])
        y = self.rng.uniform(-amplitude, amplitude, size=n_cp)
        X = np.arange(n_cp).reshape(-1, 1)
        return  X, y

    @property
    def signal(self):
        # Get random point
        X, y= self.get_random_points()
        
        # Fit GP
        kernel = RBF(length_scale=1.0) # + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X, y)

        # Predict == interpolate points
        n_ts = int(self.params['n_ts'])
        X_pred = np.linspace(X.min(), X.max(), n_ts).reshape(-1, 1)
        y_pred, _ = gpr.predict(X_pred, return_std=True)

        return y_pred - y_pred[0]

    def save_signals(self,n,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path,f'gp_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)],dtype=object),
            allow_pickle=True
        )


class RandomWalk:
    def __init__(self, config_path, seed=42):
        self.seed = seed
        self.params = load_params(config_path)
        self.rng = np.random.default_rng(seed)

    @property
    def signal(self):
        n_steps = int(self.params['n_steps'])
        deps_max = float(self.params['deps_max'])      # max change per step
        smooth_kernel = float(self.params['smooth_kernel'])  # smoothing sigma

        # 1. Draw random increments
        d_eps = self.rng.uniform(-0.5, 0.5, size=n_steps)

        # 2. Envelope term every 10 increments
        n_env = n_steps // 10 + 1
        log_delta = self.rng.uniform(-3, -1, size=n_env)
        delta = 10 ** log_delta
        envelope = np.repeat(delta, 10)[:n_steps]
        d_eps = d_eps * envelope

        # 3. Capped random walk
        eps = np.zeros(n_steps)
        for t in range(1, n_steps):
            eps[t] = np.clip(
                eps[t-1] + d_eps[t],
                eps[t-1] - deps_max,
                eps[t-1] + deps_max
            )

        # 4. Smooth with Gaussian filter
        from scipy.ndimage import gaussian_filter1d
        eps_smooth = gaussian_filter1d(eps, sigma=smooth_kernel)

        return eps_smooth - eps_smooth[0]

    def save_signals(self, n, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        np.save(
            osp.join(folder_path, f'rw_{self.seed}_{n}.npy'),
            np.array([self.signal for _ in range(n)], dtype=object),
            allow_pickle=True
        )


class Amplitude:

    def __init__(self, config_path):
         self.params = load_params(config_path)
    
    def save_signals(self, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        eps_list = []
        for eps_max in np.linspace(
            self.params['amplitude']['min'],
            self.params['amplitude']['max'],
            self.params['amplitude']['n']
        ):
            t = np.linspace(
                0, 
                2*np.pi, 
                self.params['n_ts']
            )
            eps_list += [np.sin(t) * eps_max]

        np.save(
            osp.join(folder_path, f'amplitude.npy'),
            np.array(eps_list, dtype=object),
            allow_pickle=True
        )
            

class Cyclic:
    def __init__(self, config_path):
         self.params = load_params(config_path)
    
    def save_signals(self,folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)
        t = np.linspace(
            0, self.params['n_cycles']*2*np.pi,
            self.params['n_ts']
        )
        eps = np.sin(t) * self.params['amplitude']
    
        np.save(
            osp.join(folder_path, f'cyclic.npy'),
            eps.reshape(1, -1).astype(object),
            allow_pickle=True
        )


class Impulse:
    def __init__(self, config_path):
         self.params = load_params(config_path)
    

    def save_signals(self, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)

        mu, sigma = self.params['mu'], self.params['sigma']
        t = np.linspace(
            mu - 8*sigma, 
            mu + 8*sigma, 
            self.params['n_ts']
        ) 
        eps = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - mu) / sigma)**2)/2
        # t -= mu - 8*sigma

        np.save(
            osp.join(folder_path, f'impulse.npy'),
            eps.reshape(1, -1).astype(object),
            allow_pickle=True
        )


class Resolution:
    def __init__(self, config_path):
        self.params = load_params(config_path)
    

    def save_signals(self, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)

        eps_list = []
        for n_ts in self.params['n_ts']:
            t = np.linspace(0, 2*np.pi, n_ts)
            eps_list += [np.sin(t) * self.params['amplitude']]

        np.save(
            osp.join(folder_path, f'resolution.npy'),
            np.array(eps_list, dtype=object),
            allow_pickle=True
        )
    

class PieceWise:
    def __init__(self,config_path):
        self.config = load_params(config_path)
    
    def save_signals(self,folder_path=''):
        points = self.config['points']
        n_ts = self.config['n_ts']
        
        np.save(
            osp.join(folder_path, f'piecewise.npy'),
            np.concatenate([
                np.linspace(points[i],points[i+1],n_ts)
                for i in range(len(points)-1)
            ]).reshape(1, -1).astype(object),
            allow_pickle=True
        )

        
# GENERATORS = {
#     'random' : {
#         'bl_ms' : BaselineMultisine(config_path=osp.join(CONFIG_PATH,'bl_ms.json'),seed=SEED),
#         'pd_ms' : PowerDecayMultisine(config_path=osp.join(CONFIG_PATH,'pd_ms.json'),seed=SEED),
#         'gp' : RandomWalk(config_path=osp.join(CONFIG_PATH,'gp.json'),seed=SEED),
#         'rw' : GaussianProcess(config_path=osp.join(CONFIG_PATH,'rw.json'),seed=SEED),
#     },
#     'static' : {
#         'amplitude' : Amplitude(config_path=f'{CONFIG_PATH}/amplitude.json'),
#         'cyclic' : Cyclic(config_path=f'{CONFIG_PATH}/amplitude.json'),
#         'impulse' : Impulse(config_path=f'{CONFIG_PATH}/amplitude.json'),
#         'resolution' : Resolution(config_path=f'{CONFIG_PATH}/amplitude.json'),
#     }
# }

GENERATORS = {
    'random' : {
        'bl_ms' : BaselineMultisine,
        'pd_ms' : PowerDecayMultisine,
        'rw' : RandomWalk,
        'gp' : GaussianProcess,
    },
    'static' : {
        'amplitude' : Amplitude,
        'cyclic' : Cyclic,
        'impulse' : Impulse,
        'resolution' : Resolution,
        'piecewise' : PieceWise,
    }
}

class InputsSignals:
    
    def __init__(self,data_path):    
        self.data_path = data_path
        self.u_list = np.load(self.data_path, allow_pickle=True)
        self.lengths = [len(u) for u in self.u_list]
        self.name = os.path.splitext(os.path.basename(data_path))[0]

    def print_summary(self,lens=True,points=True):
        
        print(f"Data path: {self.data_path}")
        print(f"# of sequencnes: {len(self.u_list)}")
        print()
        
        if lens:
            print(f"Lengths:")
            print(f"\t* Min: {np.min(self.lengths):.2f}")
            print(f"\t* Max: {np.max(self.lengths):.2f}")
            print(f"\t* Mean : {np.mean(self.lengths):.2f}")
            print(f"\t* Std: {np.std(self.lengths):.2f}")
            print()
        
        if points:
            print(f"Data:")
            print(f"\t* Min: {np.concatenate(self.u_list).min():.2f}")
            print(f"\t* Max: {np.concatenate(self.u_list).min():.2f}")
            print(f"\t* Mean : {np.concatenate(self.u_list).mean():.2f}")
            print(f"\t* Std: {np.concatenate(self.u_list).std():.2f}")
            print()
            print()


    def plot_samples(self,ax,num_samples=5,c='#E63946'):
        if len(self.u_list) < num_samples: 
            num_samples = len(self.u_list)
        for i,u in enumerate(self.u_list[:num_samples]):
            ax.plot(u,label=f'Sample {i+1}',alpha=(i+1)/num_samples,c=c)
        # ax.legend()
        ax.grid()
    
    def plot_samples_ftt(self,ax,num_samples=5):
        if len(self.u_list): num_samples=1
        for i,u in enumerate(self.u_list[:num_samples]):
            X = np.fft.fft(u)
            freqs = np.fft.fftfreq(len(u))
            mask = freqs >= 0
            ax.plot(freqs[mask],np.abs(X[mask]),label=f'Sample {i+1}')
        ax.grid()
    


def comp_datasets_fft(data_set_list: list[InputsSignals],num_samples=5):
    num = len(data_set_list)
    fig, ax = plt.subplots(num,figsize=(8,3*num), dpi=150,sharex=True,sharey=True)

    for i in range(num):
        data_set_list[i].plot_samples_ftt(ax=ax[i],num_samples=num_samples)
        ax[i].set_ylabel(f'{data_set_list[i].name}')
    ax[-1].set_xlabel('Freq')
    


if __name__ == '__main__':
    
    n = 200
    seed = 42
    input_path = osp.join('data','input')
    config_path = osp.join('data','config')

    for name, G in GENERATORS['random'].items():
        generator = G(
            config_path=osp.join(config_path,'random',f'{name}.json'), 
            seed=seed
        )
        generator.save_signals(
            n=n,
            folder_path=osp.join(input_path,'random')
        )
    
    for name, G in GENERATORS['static'].items():
        generator = G(
            config_path=osp.join(config_path,'static',f'{name}.json'), 
        )
        generator.save_signals(
            folder_path=osp.join(input_path, 'static')
        )