import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os.path as osp
from pathlib import Path
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



class RandomGenerator:
    
    name = 'base'
    
    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def signal(self):
        raise NotImplementedError

    def save_signals(self, n, folder_path=''):
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f"{self.name}_{self.seed}_{n}.npy",
            np.array([self.signal for _ in range(n)], dtype=object),
            allow_pickle=True
        )


class BaselineMultisine(RandomGenerator):
    
    name = 'bl_ms'
    
    def __init__(self,config,seed=42):
        super().__init__(config['random'][self.name], seed)
    
    
    @property
    def signal(self):
        t = self.generate_t()
        baseline = self.generate_baseline(t)
        multisine = self.generate_multisine(t)
        sgn = baseline + multisine
        return sgn - sgn[0]
    

    def pick(self,param,size=1):
        if isinstance(param, list):
            return self.rng.uniform(param[0], param[1], size=size)
        return param

    
    def generate_t(self):
        dt = float(self.pick(self.config['general']['dt']))
        t_end = float(self.pick(self.config['general']['t_end']))
        return np.arange(0, t_end, dt)


    def generate_baseline(self,t):
        t_step = self.pick(self.config['baseline']['t_step'])
        step_times = np.arange(0, t[-1] + t_step, t_step)
        step_amplitudes = self.pick(self.config['baseline']['amplitude'],size=len(step_times))
        return np.interp(t, step_times, step_amplitudes)

    
    def generate_multisine(self, t):
        
        freq_min = float(self.pick(self.config['multisine']['freq_min']))
        freq_max = float(self.pick(self.config['multisine']['freq_max']))
        n_freqs = int(self.pick(self.config['multisine']['freq_max']))
        amplitudes = self.pick(self.config['multisine']['amplitude'],size=n_freqs)
        phases = self.pick(self.config['multisine']['phase'],size=n_freqs)

        freqs = np.linspace(freq_min, freq_max, n_freqs)

        multisine = np.zeros_like(t)
        for i in range(n_freqs):
            multisine += amplitudes[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])

        return multisine


class PowerDecayMultisine(RandomGenerator):
    
    name = 'pd_ms'
    
    def __init__(self,config,seed=42):
        super().__init__(config['random'][self.name], seed)


    @property
    def signal(self): 

        T = self.config['len_seq']
        N = self.config['n_freqs']
        p = self.config['power']
        u_max = self.config['amplitude']

        k = np.linspace(0, T, T)  
        a = self.rng.standard_normal(N)
        u = np.zeros_like(k)
        for n in range(1, N + 1):
            u += (a[n - 1] / n**p) * np.sin(2*np.pi*n*k/N)
        u = np.array(u)
        u = u/(np.max(u) - np.min(u))*u_max
        return u
    

class GaussianProcess(RandomGenerator):
    
    name = "gp"
    
    def __init__(self, config, seed=42):
        super().__init__(config['random'][self.name], seed)


    @property
    def signal(self):
        # Get random point
        X, y= self.get_random_points()
        
        # Fit GP
        kernel = RBF(length_scale=1.0) # + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr.fit(X, y)

        # Predict == interpolate points
        n_ts = int(self.config['n_ts'])
        X_pred = np.linspace(X.min(), X.max(), n_ts).reshape(-1, 1)
        y_pred, _ = gpr.predict(X_pred, return_std=True)

        return y_pred - y_pred[0]

    
    def get_random_points(self):
        n_cp = int(self.config['n_cp'])
        amplitude = float(self.config['amplitude'])
        y = self.rng.uniform(-amplitude, amplitude, size=n_cp)
        X = np.arange(n_cp).reshape(-1, 1)
        return  X, y

    
class RandomWalk(RandomGenerator):
    
    name = 'rw'
    
    def __init__(self, config, seed=42):
        super().__init__(config['random'][self.name], seed)

    @property
    def signal(self):
        n_steps = int(self.config['n_steps'])
        deps_max = float(self.config['deps_max'])      # max change per step
        smooth_kernel = float(self.config['smooth_kernel'])  # smoothing sigma

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


class Combined(RandomGenerator):
    
    name = 'combined'
    
    def __init__(self, config, seed=42):

        super().__init__(config, seed)

        self.child_gens = [
            BaselineMultisine, 
            PowerDecayMultisine, 
            GaussianProcess, 
            RandomWalk
        ]
    
    def save_signals(self,n,folder_path=''):
        
        signals = []

        for GEN in self.child_gens:
            generator = GEN(config=self.config, seed=self.seed)
            signals.extend([generator.signal for _ in range(n//4)])
        
        self.rng.shuffle(signals)

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f'{self.name}_{self.seed}_{n}.npy',
            np.array(signals,dtype=object),
            allow_pickle=True
        )



class Amplitude:

    name = 'amplitude'
    
    def __init__(self, config):
        self.config = config['static'][self.name]
    
    def save_signals(self, folder_path=''):
        
        eps_list = []
        for eps_max in np.linspace(
            self.config['amplitude']['min'],
            self.config['amplitude']['max'],
            self.config['amplitude']['n']
        ):
            t = np.linspace(
                0, 
                2*np.pi, 
                self.config['n_ts']
            )
            eps_list += [np.sin(t) * eps_max]

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f'{self.name}.npy',
            np.array(eps_list, dtype=object),
            allow_pickle=True
        )
            

class Cyclic:
    
    name = 'cyclic'
    
    def __init__(self, config):
        self.config = config['static'][self.name]
    
    def save_signals(self,folder_path=''):

        t = np.linspace(
            0, self.config['n_cycles']*2*np.pi,
            self.config['n_ts']
        )
        eps = np.sin(t) * self.config['amplitude']

        
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f'{self.name}.npy',
            eps.reshape(1, -1).astype(object),
            allow_pickle=True
        )


class Impulse:

    name = 'impulse'
    
    def __init__(self, config):
        self.config = config['static'][self.name]
    

    def save_signals(self, folder_path=''):
        
        mu, sigma = self.config['mu'], self.config['sigma']
        t = np.linspace(
            mu - 8*sigma, 
            mu + 8*sigma, 
            self.config['n_ts']
        ) 
        eps = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t - mu) / sigma)**2)/2
        # t -= mu - 8*sigma

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        np.save(
            folder_path / f'{self.name}.npy',
            eps.reshape(1, -1).astype(object),
            allow_pickle=True
        )


class Resolution:

    name = 'resolution'
    
    def __init__(self, config):
        self.config = config['static'][self.name]
    

    def save_signals(self, folder_path=''):
        if not osp.exists(folder_path): os.makedirs(folder_path)

        eps_list = []
        for n_ts in self.config['n_ts']:
            t = np.linspace(0, 2*np.pi, n_ts)
            eps_list += [np.sin(t) * self.config['amplitude']]

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f'{self.name}.npy',
            np.array(eps_list, dtype=object),
            allow_pickle=True
        )
    

class PieceWise:
    
    name = 'piecewise'
    
    def __init__(self,config):
        self.config = config['static'][self.name]
    
    def save_signals(self,folder_path=''):
        points = self.config['points']
        n_ts = self.config['n_ts']
        
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        np.save(
            folder_path / f'{self.name}.npy',
            np.concatenate([
                np.linspace(points[i],points[i+1],n_ts)
                for i in range(len(points)-1)
            ]).reshape(1, -1).astype(object),
            allow_pickle=True
        )


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

    def plot_histogram(self,ax, bins_time=100, bins_value=100, vmin=None, vmax=None,cmap=None):
        n_signals, n_timepoints = self.u_list.shape
        
        times = np.arange(n_timepoints)

        X = np.repeat(times[None, :], n_signals, axis=0).ravel()
        Y = self.u_list.ravel()

        H, xedges, yedges = np.histogram2d(
            X, Y,
            bins=[bins_time, bins_value],
        )

        H = H.T  # value bins on vertical axis
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax.imshow(
            H,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            vmin=vmin, vmax=vmax,
            cmap=cmap,
            # norm=LogNorm(vmin=1, vmax=H.max())
        )

        # plt.colorbar(im, ax=ax).set_label("counts")

    


RANDOM_GEN_REGISTRY = [
    BaselineMultisine,
    PowerDecayMultisine,
    GaussianProcess,
    RandomWalk,
    Combined
]

STATIC_GEN_REGISTRY = [
    Amplitude,
    Cyclic,
    Impulse,
    Resolution,
    PieceWise,
]

if __name__ == '__main__':
    
    # Defines
    n = 200
    seed = 42
    config_path = Path("configs","generators.yaml")
    input_dir  = Path('data','input')

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Generate random data sets
    for Generator in RANDOM_GEN_REGISTRY:
        generator = Generator(config=config,seed=seed)
        generator.save_signals(n=n, folder_path=Path(input_dir,'random'))

    # Generate static data sets
    for Generator in STATIC_GEN_REGISTRY:
        generator = Generator(config=config)
        generator.save_signals(folder_path=Path(input_dir,'static'))

    