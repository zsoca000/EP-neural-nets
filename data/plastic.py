from scipy.optimize import root
from numpy import sign
import numpy as np
import matplotlib.pyplot as plt

# Linear elastic
E = 1e4

# Linear plastic
H_iso = 400
H_kin = 400

# Swift
a = 100
n = 0.1
eps0 = 5e-4
Y0 = a*eps0**n

# Armstrong-Frederick
b = 10

# The material laws
materials = {
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

# The hardening rule
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