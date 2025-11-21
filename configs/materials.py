# Linear elastic params
E = 2e9

# Linear plastic params
H_iso = 750e6
H_kin = 400e6

# Swift params
a = 200e6
n = 0.2
eps0 = 0.0003
Y0 = a*eps0**n * 1.5

# Armstrong-Frederick params
b = 10

# The material laws
materials = {
    'isotropic-swift': {
        'E' : E,
        'Y' : lambda λ : a*(eps0+λ)**n,
        'dalpha': lambda alpha, deps_p, dλ : 0,
        'color': '#000aff',
        'label': 'Isotropic Swift'
    },
    'isotropic-linear': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha': lambda alpha, deps_p, dλ : 0,
        'color': '#ff0000',
        'label': 'Isotropic Linear'
    },
    'kinematic-linear': {
        'E' : E,
        'Y' : lambda λ : Y0,
        'dalpha': lambda alpha, deps_p, dλ : H_kin*deps_p,
        'color': '#00b507',
        'label': 'Kinematic Linear',
    },
    'kinematic-armstrong-fredrick': {
        'E' : E,
        'Y' : lambda λ : Y0,
        'dalpha' : lambda alpha, deps_p, dλ : H_kin*deps_p-dλ*b*alpha,
        'color': '#ff9a01',
        'label': 'Kinematic AF',
    },
    'mixed-linear': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha': lambda alpha, deps_p, dλ : H_kin*deps_p,
        'color': '#9501ff',
        'label': 'Mixed Linear',
    },
    'mixed-armstrong-fredrick': {
        'E' : E,
        'Y' : lambda λ : (Y0/H_iso+λ)*H_iso,
        'dalpha' : lambda alpha, deps_p, dλ : H_kin*deps_p-dλ*b*alpha,
        'color': '#00c6c3',
        'label': 'Mixed AF',
    }
}