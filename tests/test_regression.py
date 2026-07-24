#!/usr/bin/env python3
"""
tests/test_regression.py
~~~~~~~~~~~~~~~~~~~~~~~~
Equivalent to test_classification.py but for a regression task.
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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m  = 10000, 10
# Regresion
X, y = make_regression(n_samples=n, n_features=m, bias=1, noise=0.05,n_targets=1, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# Normalización
scaler      = StandardScaler().fit(X_train)                 # ROLANN works better w/ scaled data
yscaler     = StandardScaler(with_std=False).fit(y_train)   # otherwise bias dominates the sol. (centering)
X_train     = scaler.transform(X_train)
X_test      = scaler.transform(X_test)
y_train     = yscaler.transform(y_train)
y_test      = yscaler.transform(y_test)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

#%% [2. ROLANN]
from sklearn.metrics import mean_absolute_error
from rolann import ROLANN

lam = 0.001                         # l2 reg
f_act = 'linear'                    # regression task

mdl = ROLANN(f_act=f_act, lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_                   # optim. weights
y_pred = mdl.predict(X_test)
mse = mean_absolute_error(y_test, y_pred)
print(f"ROLANN mse: {mse:.2f}")

#%% [2. VFL SPLIT]
from helpers import split_vfl_X

p = [2, 2, 6]  # -> 3 clientes con 2, 2, y 6 variables
X_lst_T = split_vfl_X(X_train.T, p)
num_nodos = len(X_lst_T)

#%% [3. BCD ROLANN]
from bcd import BCDClient, BCDCoordinator

# [CLIENTS & COORD]
lst_clients = [BCDClient(id=i+1,        # clients should be numbered starting from 1
                         X=X_lst_T[i],  # local data matrix (m_i, n)
                         f_act=f_act,   # act fn
                         lam=lam,       # l2 reg
                         debug=True)    # save client results (norm_grad, etc.)
                         for i in range(num_nodos)]

coord = BCDCoordinator(clients=lst_clients,             # list of clients 
                       d=y_train,                       # real/desired outputs
                       f_act=f_act,                     # act fn
                       test_data=(X_test.T, y_test),    # test_data tuple for scoring (X transposed!)
                       w_rolann=w_rolann)               # optim 'centralized' weights (ROLANN)

# [BASIC FIT PARAMS]
max_iter    = 50        # max iter
tol         = 1e-4      # stop criteria
lr          = 0.75      # learning_rate
init        = 'zero'    # init global inputs as 'random' or 'zero' (former is better for privacy)
heur        = False     # disable heuristics for convergence guarantees in 'parallel' mode

# [PARALLEL MODE]: (lr=lr) (parallel)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel', init=init, heur=heur)
df_par = coord.get_results(name='ex_par_run')
df_par

# Check results dataframes columns for more info
print(df_par.head())

#%% [4. Ridge]
from sklearn.linear_model import Ridge

mdl_lr  = Ridge(alpha=lam, tol=tol, max_iter=max_iter)
mdl_lr.fit(X_train, y_train)
w_lr    = np.hstack([mdl_lr.intercept_, mdl_lr.coef_])
mse_lr  = mean_absolute_error(y_test, mdl_lr.predict(X_test))
print(f"RIDGE mse: {mse_lr:.2f}")

#%% [5. VFL_LR]
from lr import RidgeClient, RidgeCoordinator

# [CLIENTS & COORD]
clients_lr = [RidgeClient(id=i+1,          # clients should be numbered starting from 1
                          X=X_lst_T[i],    # local data matrix
                          lam=lam,         # l2 reg directly, equiv to alpha
                          debug=True)      # save client results
                    for i in range(num_nodos)]

coord_lr = RidgeCoordinator(clients=clients_lr,            # client list
                            y=y_train,                     # real/desired outputs
                            test_data=(X_test.T, y_test),  # test data, (X transposed!)
                            w_lr=w_lr.T)                   # optim weights (centralized) as col vector

# [BASIC FIT PARAMS]: almost identical to BCD_ROLANN
max_iter    = 500       # max iter
tol         = 1e-4      # stop criteria
lr          = 0.75      # learning_rate
init        = 'zero'    # init global inputs as 'random' or 'zero' (former is better for privacy)
taylor      = False     # use taylor approx for the act fn (enable FHE comms)
reg_bias    = True      # reg bias term (sim to fit_intercept in `LogisticRegression`)

# [GRADIENT DESCENT]
coord_lr.fit(max_iter=max_iter, tol=tol, lr=lr, reg_bias=reg_bias, method='gd', taylor=taylor) 
df_gd = coord_lr.get_results(name='ex_gd')

#%% [4. PLOTTERS]
from plotters import plot_sns, plot_weights_comparison, SNS_PARAMS

df_merged = pd.concat([df_par, df_gd])

# [PLOT CONVERGENCE]
plot_sns(df_merged,
         tol=tol,
         title='Plotting multiple fit runs',
         **SNS_PARAMS|{'hue': 'name'}    # override/add params as needed
         )
plot_sns(df_merged,
         tol=tol,
         title='Plotting multiple fit runs',
         **SNS_PARAMS|{'hue': 'name', 'y':'score_test'}    # override/add params as needed
         )

# [PLOT WEIGHTS]
plot_weights_comparison(w_rolann, coord.w_, label_w1='ROLANN', label_w2='BCD')
plot_weights_comparison(w_lr, coord_lr.w_, label_w1='Ridge', label_w2='RidgeVFL')
plot_weights_comparison(df_par.w_i.values[-1], df_gd.w_i.values[-1],
                        label_w1='BCD ROLANN',
                        label_w2='VFL RIDGE',
                        title='Weights comparison VFL algorithms')

# %%
