#!/usr/bin/env python3
"""
tests/test_run.py
~~~~~~~~~~~~~~~~~~~~~~~~
Showcase of helper functions for easier model fitting (`fit_*_models`)
and multiple VFL scenario settings (`run_grid_search*`).
"""
#%% [0. SYSPATH]
# To avoid using relative imports (requires setting up modules)
# we update PYTHONPATH to include the root directory (cwd parent)
import sys
from pathlib import Path

if __file__ is None:
    p = Path.cwd().parent
else:
    p = Path(__file__).parents[1]
sys.path.insert(0, str(p))

#%% [1. TOY DATASET]
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m    = 10000, 10

# Clasificacion
X, y = make_classification(n_samples=n, n_features=m, n_classes=2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

#%% [2. FIT MODELS]
from helpers import fit_rolann_models, fit_lr_models, fit_bcd_models, fit_lrvfl_models

# [CENTRALIZED ROLANN]
rolann_params = {'f_act':   'tanhip',
                 'lam':     1e-3}

# mode: 'binary'|'regression'|'ovr'
mdls_rolann = fit_rolann_models(X_train, y_train, mode='binary', **rolann_params)

# [CENTRALIZED LR]
lr_params = {'lam':             1e-3}   # automatic handling of `C` and `alpha` equivalents

# mode: 'auto' (for binary clasif.)|'regression'|'ovr'
mdls_lr  = fit_lr_models(X_train, y_train, mode='auto', **lr_params)

print("[fit_rolann_models()]")
print(f"Return Type: {type(mdls_rolann)} (len:{len(mdls_rolann)})")
print(f"\t one model per output: {type(mdls_rolann[0])}")

print("[fit_lr_models()]")
print(f"Return Type: {type(mdls_lr)} (len:{len(mdls_lr)})")
print(f"\t always contains one single model ever (even w/ mode='ovr'): {type(mdls_lr[0])}")

#%% [BCD_ROLANN]
bcd_params = {
    # Params VFL split
    'p':        [2, 2, 6],
    # Params FIT
    'max_iter': 100,
    'lr':       .75,
    'tol':      1e-4 ,
    'init':     'zero',
    'mode_bcd': 'parallel',
    # Params BCD Coord
    'w_rolann':   mdls_rolann[0].w_,
    'test_data': (X_test.T, y_test),
    'bcd_impl': 'normal',               # BCD_ROLANN impls: 'normal'|'par'|'fhe'|'sec'
    }

# mode: 'binary'|'regression'|'ovr'|'multi'
mdls_bcd = fit_bcd_models(X_train, y_train, mode='binary', **(rolann_params|bcd_params))
bcd_clients, bcd_coords = mdls_bcd

print("\n[fit_bcd_models()]")
print(f"Return Type: {type(mdls_bcd)} (len:{len(mdls_bcd)})")
print("tuple with list of coordinators and clients")
print("element in tuple -> element in list")
print(f"[0] {type(bcd_coords)} -> {type(bcd_coords[0])} ...")
print(f"[1] {type(bcd_clients)} -> {type(bcd_clients[0])} ...")

#%% [VFL_LR]
vfl_params = {
    # Params VFL split
    'p':        [2, 2, 6],
    # Params FT
    'max_iter': 500,
    'lr':       .75,
    'tol':      1e-4,
    'method':   'gd',
    'taylor':   False,
    'init':     'zero',
    'reg_bias': False,
    # Params LRCoordinator
    'w_lr':     mdls_lr[0].w_,
    'test_data':(X_test.T, y_test)
    }

mdls_vfl = fit_lrvfl_models(X_train, y_train, mode='binary', **(lr_params|vfl_params))
vfl_clients, vfl_coords = mdls_vfl

print("\n[fit_lrvfl_models()]")
print(f"Return Type: {type(mdls_vfl)} (len:{len(mdls_vfl)})")
print("tuple with list of coordinators and clients")
print("element in tuple -> element in list")
print(f"[0] {type(vfl_coords)} -> {type(vfl_coords[0])} ...")
print(f"[1] {type(vfl_clients)} -> {type(vfl_clients[0])} ...")

#%% [2. GRID SEARCH]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch

# Enables to use a param_grid and automatically run combinatorial experiments thereof.
# Dictionary with {key: lst_values}

# [BCD_ROLANN]
bcd_grid = {
    'p':        [5, [2, 2, 6]],             # VFL config (5 equal nodes, 3 nodes with 2-2-6 vars)
    'f_act':    ['logs', 'tanhip'],         # act fns
    'lam':      [0.001],                    # l2 regularization
    'mode_bcd': ['parallel', 'sequential'], # modes ('sequential' automatically sets lr=1)
    'lr':       [0.25],                     # learning rates
    'tol':      [1e-4],                     # tolerance
    'max_iter': [100]                       # max iterations
}   # TOTAL: 2^3 = 8 RUNS

# Results: a list of tuples ([mdl_rolann], [mdl_bcd_coords]), one for each run
exps_bcd = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='binary')

#%% [VFL_LR]
vfl_grid = {
    'p':        [5, [2, 2, 6]],     # VFL config (5 equal nodes, 3 nodes with 2-2-6 vars)
    'lam':      [0.001],            # l2 regularization
    'method':   ['gd', 'bfgs'],     # modes
    'taylor':   [False, True],      # taylor aprox.
    'lr':       [0.25],             # learning rates
    'tol':      [1e-4],             # tolerance
    'max_iter': [500],              # max iters
}   # TOTAL: 2^3 = 8 RUNS

# Results: a list of tuples ([mdl_lr], [mdl_vfllr_coords]), one for each run
exps_vfl = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='auto')

# [MERGE]
df_bcd = merge_df_gridsearch(exps_bcd, name='example_bcd_grid')
df_vfl = merge_df_gridsearch(exps_vfl, name='example_vfl_grid')

#%% [PLOT]
from plotters import plot_sns, SNS_PARAMS

plot_sns(df_bcd, tol=bcd_grid['tol'], title="BCD_GRID", **SNS_PARAMS|{'hue': 'f_act'})
plot_sns(df_vfl, tol=vfl_grid['tol'], title="LR_GRID", **SNS_PARAMS|{'hue':'taylor'})
# For futher filtering, it is advised to use
# df.query("col1 == @var1 and col2 > @var2 or col3 in @lst3")

#%%
