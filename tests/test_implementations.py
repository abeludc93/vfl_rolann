#!/usr/bin/env python3
"""
tests/test_par.py
~~~~~~~~~~~~~~~~~~~~~~~~
Showcase of the BCD_ROLANN parallel implementation with
the `joblib` library for parallelized computations.
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
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Clasificacion
X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
sc = StandardScaler().fit(X_train)
X_train, X_test = sc.transform(X_train), sc.transform(X_test)
print(f"Muestras: {X.shape[0]} | Variables: {X.shape[1]}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

#%% [2. BCD ROLANN PAR]
from helpers import run_grid_search
from helpers import total_time
import tenseal as ts

# [TenSEAL]
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

# [BCD_ROLANN]
bcd_grid = {
    'p':        [2, 10],                        # VFL config
    'f_act':    ['logs'],                       # act fns
    'lam':      [0.001],                        # l2 regularization
    'mode_bcd': ['parallel'],                   # modes ('sequential' automatically sets lr=1)
    'lr':       [0.25],                         # learning rates
    'tol':      [0],                            # tolerance
    'max_iter': [100],                          # max iterations
    'bcd_impl': ['normal', 'sec', ],# BCD ROLANN implementations
    'ctx':      [context],                      # TenSeal context for fhe impl

}   # TOTAL: 4*4 = 16 RUNS

# Results: a list of tuples ([mdl_rolann], [mdl_bcd_coords]), one for each run
exps_bcd = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='ovr')

# We collect as an example the results from the first output: class 0
df_lst  = []
for _, bcd in exps_bcd:
    df = bcd[0].get_results(name=f"{bcd[0].__class__.__name__}_0")
    df_lst.append(df)
df_bcd = pd.concat(df_lst)
df_bcd = df_bcd.assign(total_time=total_time(df_bcd))
w_rolann = exps_bcd[0][0][0].w_
w_bcd = df_bcd.query("p == 2").w_i.values[-1]

#%% [3. PLOTTERS]
from plotters import plot_sns, SNS_PARAMS, plot_im_split, plot_im_weights

plot_sns(df_bcd, plot_log=False, **SNS_PARAMS|{'x': 'iter', 'y': 'total_time', 'hue': 'name'})
plot_sns(df_bcd.query('name not in "BCDCoordinator_fhe_0"'), plot_log=False,
         **SNS_PARAMS|{'x': 'iter', 'y': 'total_time', 'hue': 'name'})

p_dict = {'1': list(range(0,32)), '2': list(range(32,64))}
plot_im_split(X[0, :].reshape(8, 8), lst_idx=list(p_dict.values()), title="VFL split-image")
plot_im_weights(w_rolann, w_bcd, p_dict, imgsize=(8,8), title='ROLANN vs. BCD ROLANN')

#%%
"""
NOTES:
1.  With this little dataset setup, we can't overcome the overhead of the joblib scheduling
    parallelization on the BCDCoordinator_par, BCDCoordinator_sec and BCDCoordinator_fhe
    implementations. Nonetheless, with bigger datasets and matrixes (such as a n=50k, m=10k),
    this extra setup time will matter much less and actually be advantageous.

2. SecureAgg vs CKKS (FHE): they address two different security scenarios (threat models):
    a.  SecAgg offers privacy-on-transit, that it, comms are masked but its sum is the correct
        real output (and the coordinator can see it as such).
    b.  FHE offers total-privacy, as the coordinator operates entirely on encripted data and cant
        see its outputs.
In summary, SecureAgg is more similar to a kind of better differential privacy.
"""