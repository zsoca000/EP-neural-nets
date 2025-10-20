from scipy.optimize import root
from numpy import sign
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern"})

# Linear elastic
E = 2e9

# Linear plastic
H_iso = 750e6
H_kin = 400e6

# Swift
a = 200e6
n = 0.2
eps0 = 0.0003
Y0 = a*eps0**n * 1.5

# Armstrong-Frederick
b = 10

# The material laws
MATERIALS = {
    'isotropic-swift': {
        'E' : E,
        'Y' : lambda λ : a*(eps0+λ)**n,
        'dalpha': lambda alpha, deps_p, dλ : 0,
        'color': '#000aff',
    },
    'isotropic-linear': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha': lambda alpha, deps_p, dλ : 0,
        'color': '#ff0000',
    },
    'kinematic-linear': {
        'E' : E,
        'Y' : lambda λ : Y0,
        'dalpha': lambda alpha, deps_p, dλ : H_kin*deps_p,
        'color': '#00b507',
    },
    'kinematic-armstrong-fredrick': {
        'E' : E,
        'Y' : lambda λ : Y0,
        'dalpha' : lambda alpha, deps_p, dλ : H_kin*deps_p-dλ*b*alpha,
        'color': '#ff9a01',
    },
    'mixed-linear': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha': lambda alpha, deps_p, dλ : H_kin*deps_p,
        'color': '#9501ff',
    },
    'mixed-armstrong-fredrick': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha' : lambda alpha, deps_p, dλ : H_kin*deps_p-dλ*b*alpha,
        'color': '#00c6c3',
    }
}


def HARDENING(eps,E,dalpha_F,Y_F):
    sig = np.zeros_like(eps)
    eps_p = np.zeros_like(eps)
    λ = np.zeros_like(eps)
    Y = np.ones_like(eps)*Y_F(0)
    alpha = np.zeros_like(eps)

    for i in range(eps.shape[0]-1):
        deps = (eps[i+1] - eps[i])
        
        sig_trial = sig[i] + E*deps
        
        if abs(sig_trial - alpha[i]) - Y[i] <= 0: 
            deps_p = 0
            dλ = 0
        else:
            def eq(deps_p):
                dλ = deps_p*sign(sig_trial)
                ret  = abs(sig_trial - alpha[i] - dalpha_F(alpha[i],deps_p,dλ))
                ret -= E*dλ
                ret -= Y_F(λ[i]+dλ)
                return ret
            
            deps_p = (root(eq,1e-8).x)[0]
            dλ = deps_p*sign(sig_trial)
    
        eps_p[i+1] = eps_p[i] + deps_p
        λ[i+1] = λ[i] + dλ
        Y[i+1] = Y_F(λ[i+1])
        alpha[i+1] = alpha[i] + dalpha_F(alpha[i],deps_p,dλ)
        sig[i+1] = sig_trial - E*deps_p
    
    return sig, alpha, Y



def plot_responses(eps_list, sig_list, color_list=None):

    fig = plt.figure(figsize=(8, 4),dpi=250)
    gs = GridSpec(2, 2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    
    n = len(eps_list)
    if color_list is None: 
        color_list=['red']*n
        alpha_list=[(i+1)/n for i in range(n)]
    
    for i in range(n):
        ax1.plot(eps_list[i],sig_list[i],lw=1.5,alpha=alpha_list[i],c=color_list[i])
        ax2.plot(eps_list[i],lw=1.5,alpha=alpha_list[i],c=color_list[i])
        ax3.plot(sig_list[i],lw=1.5,alpha=alpha_list[i],c=color_list[i])

    ax1.set_xlabel(r'$\varepsilon$', fontsize=18)
    ax2.set_xlabel(r'$n$', fontsize=18)
    ax3.set_xlabel(r'$n$', fontsize=18)
    ax1.set_ylabel(r'$\sigma$', fontsize=18)
    ax2.set_ylabel(r'$\varepsilon$', fontsize=18)
    ax3.set_ylabel(r'$\sigma$', fontsize=18)
    plt.tight_layout()
    plt.show()


def load_responses(mat_name,inp_type,inp_name,data_path=''):
    return (
        np.load(
            osp.join(data_path,'input',inp_type,f'{inp_name}.npy'),
            allow_pickle=True
        ),
        np.load(
            osp.join(data_path,'output',mat_name,inp_type,f'{inp_name}.npy'),
            allow_pickle=True
        )
    )


if __name__ == '__main__':
    input_path = osp.join('data','input')
    output_path = osp.join('data','output')

    for mat_name, mat in MATERIALS.items():
        for inp_type in ['static','random']:
            for file in os.listdir(osp.join(input_path,inp_type)):
                inp_name,_ = os.path.splitext(file)
                log = f'Response of {mat_name} for {inp_name}'
                eps_list = np.load(
                    osp.join(input_path,inp_type,file),
                    allow_pickle=True
                )

                # Calc responses
                sig_list = np.array([
                    HARDENING(
                        eps.astype(np.float32),
                        mat['E'],mat['dalpha'],mat['Y']
                    )[0]
                    for eps in tqdm(eps_list,desc=log)
                ],dtype=object)

                # Where to save the response?
                save_folder = osp.join(
                    output_path,
                    mat_name,
                    inp_type,
                )
                if not osp.exists(save_folder):
                    os.makedirs(save_folder)

                # Save response
                np.save(
                    osp.join(save_folder,file),
                    sig_list,
                    allow_pickle=True,
                )


 