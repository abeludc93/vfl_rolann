# %%
# Comentarios:
# El flujo de trabajo es equivalente al de ROLANN + BCD.
# Habría que hacer funciones equivalentes tipo fit_lr_models y fit_vfl_lr_models
# O incluso hacer una mas abstracta que valga para las dos

# %%
from helpers import fit_lr_models, fit_lrvfl_models

# Centralizado
param_grid = dict(lam=lam, tol=tol, max_iter=max_iter)
mode = 'ovr'
lr_model = fit_lr_models(X_train, y_train, mode=mode, **param_grid)

#%% VFL
lr_params = {'lam': lam, 'tol': tol, 'max_iter': max_iter}
vfl_params = {'p': 3,
              'lr': .5,
              'method': 'gd',
              'taylor':False,
              'w_lr': lr_model[0].w_,
              'test_data': (X_test.T, y_test)
            }
lrvfl_models = fit_lrvfl_models(X_train, y_train, mode='regression', **(lr_params|vfl_params))

# %% Grid Search
from helpers import run_grid_search_lr

param_grid = {
    'lam': [1e-3, 1e-1],                    # l2 regularization
    'tol': [1e-5],                          # tolerance
    'max_iter': [50],                       # max iters

    'p': [3, 5],                            # node config
    'lr': [0.5, 0.75],                      # learning rates
    'method': ['gd'],                       # modes
    'taylor': [False]                 # taylor aprox.
}

lst_exps = run_grid_search_lr(X_train, y_train, X_test, y_test, param_grid, mode='regression')
# Resultados GS - ejemplo 1er experimento
res_lr, res_vfl = lst_exps[0]       # tupla ([mdl_lr], [mdl_vfl])
# Resultados GS - ejemplo de como coger todos los mdl_lr y mdl_vfl de cada experimento
lst_lr, lst_vfl = list(zip(*lst_exps))

#%% [GRIDSEARCH]
from helpers import run_grid_search

param_grid = {
    'f_act': ['logs'],
    'lam': [0.001],                         # regularization

    'p': [2, 3],                            # node config
    'mode_bcd': ['parallel', 'sequential'], # modes
    'lr': [0.5, 0.75, 1.0],                 # learning rates
    'tol': [1e-5],                          # tolerance
    'max_iter': [50]                        # max iterations
}

lst_exps = run_grid_search(X_train, y_train, X_test, y_test, param_grid, mode='binary')
# Resultados GS - ejemplo 1er experimento
res_rolann, res_bcd = lst_exps[0]       # tupla ([mdl_rolann], [mdl_bcd])
# Resultados GS - ejemplo de como coger todos los mdl_rolann y mdl_bcd de cada experimento
lst_rolann, lst_bcd = list(zip(*lst_exps))