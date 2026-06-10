

MAT_MAP = {
            'isotropic-linear': r'm_1',
            'isotropic-swift': r'm_2',
            'kinematic-linear': r'm_3', 
            'kinematic-armstrong-fredrick': r'm_4',
            'mixed-linear': r'm_5',
            'mixed-armstrong-fredrick': r'm_6',
        }

def dataset_latex(inp_type, inp_name, mat_name=None):
    
    # Alap karakter meghatározása (E vagy T)
    base_char = 'E' if inp_type == 'eval' else 'T'
    
    # Bemenet nevének tisztítása (alsó index)W
    clean_inp = inp_name.split('_')[0]
    if len(clean_inp) > 2:
        clean_inp = clean_inp[:3]
        
    # Ha van megadva mat_name, hozzávesszük a felső indexet
    if mat_name is not None:
        # Ha a mat_name benne van a szótárban, kicseréli, ha nincs, marad az eredeti
        mat_id = MAT_MAP.get(mat_name, mat_name)
        
        return rf'$\mathcal{{{base_char}}}^{{{mat_id}}}_{{\mathrm{{{clean_inp}}}}}$'
    
    # Ha nincs mat_name, felső index nélkül térünk vissza
    return rf'$\mathcal{{{base_char}}}_{{\mathrm{{{clean_inp}}}}}$'



def eval_metric_latex(
    inp_type, 
    inp_name,
    mat_name=None,
    err_measure=None, 
    err_type=None, 
):

    dataset = dataset_latex(inp_type, inp_name, mat_name)[1:-1]

    if err_measure is None and err_type is None:
        return rf'$J\left({{{dataset}}}\right)$'
    
    
    measure = 'NMSE' if err_measure == 'MSE_rel' else err_measure
    loc_glob = 'loc' if err_type == 'local' else 'glob'
    
    return rf'$J_{{\mathrm{{{measure}}}}}^{{\mathrm{{{loc_glob}}}}}\left({{{dataset}}}\right)$'

    



def dataset_str(name, seed=True, num=True):
    if isinstance(name, list):
        return [dataset_str_single(x, seed=seed, num=num) for x in name]
    elif isinstance(name, str):
        return dataset_str_single(name, seed=seed, num=num)


def dataset_str_single(name, seed=True, num=True):
    words = name.split('_')
    num_val = words[-1]
    seed_val = words[-2]
    if len(words) == 4:
        ret = words[0].upper() + '-' + words[1].upper()
    else:
        ret = words[0].upper()
    
    ret += f'-{seed_val}' if seed else ''
    ret += f'\n({num_val})' if num else ''
    
    return ret

