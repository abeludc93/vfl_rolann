#!/usr/bin/env python3
"""
tests/test_classification.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example of a binary classification task with ROLANN/BCD_ROLANN
and LogisticRegression/VFL_LR for a synthethic toy-dataset.
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
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m    = 10000, 10
n_red   = 2
weights = [.5, .5]

# Clasificacion
X, y = make_classification(n_samples=n, n_features=m, n_classes=2, weights=weights, n_informative=m-n_red, n_redundant=n_red, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) # ROLANN works better w/ scaled data, otherwise bias dominates the sol.
X_test = scaler.transform(X_test)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")
print(f"TRAIN BAL: {np.unique_counts(y_train).values} -> {np.unique_counts(y_train).counts}")

#%% [2. ROLANN]
from sklearn.metrics import accuracy_score
from rolann import ROLANN
from helpers import score2label

lam = 0.001                         # l2 reg
f_act = 'logs'                      # logsig -> clasif

mdl = ROLANN(f_act=f_act, lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_                   # optim. weights
y_pred = score2label(mdl.predict(X_test), f_act=f_act)
acc = accuracy_score(y_test, y_pred)
print(f"ROLANN acc: {acc:.2f}")

#%% [2. VFL SPLIT]
from helpers import split_vfl_X
# División de matriz de datos en 'p' clientes/nodos
# IMP! Es necesario mandar X traspuesta, de dimensiones (m, n)
# La lista devuelta contiene p matrices locales X_i     (m_i, n)
p = [2, 2, 6]   # -> 3 clientes con 2, 2, y 6 variables
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

# [SEQUENTIAL MODE]: (lr=1) (sequential)
coord.fit(max_iter=max_iter, tol=tol, lr=1, mode='sequential', init=init, heur=heur)
df_seq = coord.get_results(name='ex_seq_run')   # returns fit history as dataframe
df_seq

# [PARALLEL MODE]: (lr=lr) (parallel)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel', init=init, heur=heur)
df_par = coord.get_results(name='ex_par_run')
df_par

# Check results dataframes columns for more info
print(df_seq.head(), df_par.head())

#%% [4. PLOTTERS]
from plotters import plot_sns, plot_weights_comparison, SNS_PARAMS
from helpers import total_time, communication_costs
# It is advised to name the results from different fit runs, merge 
# them into a single dataframe, and let seaborn deal with the plotting
df_merged = pd.concat([df_seq, df_par])

# [PLOT_SNS EXAMPLES]
plot_sns(df_seq,
         plot_log=True,
         tol=tol,
         title='Plotting individual fit runs - SEQ',
         **SNS_PARAMS)

plot_sns(df_merged,
         tol=None,                                      # tol applies to diff_Z values
         title='Plotting multiple fit runs',
         **SNS_PARAMS|{'y': 'diff_w', 'style': 'name'}  # override/add params as needed
         )

plot_sns(df_par,
         plot_log=False,
         tol=None,
         title='Plotting metrics - PAR',
         **SNS_PARAMS|{'y': 'score_test'})

plot_sns(df_par.assign(total_time=total_time(df_par),
					   commcostMB=communication_costs(max(df_par.iter), len(p), n, out='megabytes')),
         plot_log=False,
         tol=None,
         title='Plotting extra info through aux fns in `helpers.py`',
         **SNS_PARAMS|{'x':'total_time', 'y':'commcostMB'})

# [PLOT WEIGHTS]
plot_weights_comparison(w_rolann, coord.w_, title='Weights comparison vs. ROLANN')
plot_weights_comparison(df_seq.w_i.values[-1], df_par.w_i.values[-1],
                        label_w1='BCD SEQ',
                        label_w2='BCD PAR',
                        title='Weights comparison vs. modes')

#%% [5. HEURISTIC EXAMPLE]
from helpers import compute_max_lr_parallel

# Compute max learning rate of convergence for the `parallel` mode
# given the VFL split `p` and ROLANN hyperparams `lam` and `f_act`
max_lr = compute_max_lr_parallel(X_train, p, lam, f_act)
lr_divergence = min(float(f"{max_lr+.1:.2f}"), 1)
print(f"Max learning rate for parallel mode: {max_lr:.2f}")
print(f"\tGuaranteed divergence w/ lr: {lr_divergence}")

# [DIVERGENCE]
coord.fit(max_iter=max_iter, tol=tol, mode='parallel', lr=lr_divergence, heur=False)
df_divergence = coord.get_results(name='ex_divergence')

# [GUARANTEED CONVERGENCE THROUGH HEURISTICS]
coord.fit(max_iter=max_iter, tol=tol, lr=lr_divergence, mode='parallel', heur=True)
df_convergence = coord.get_results(name='ex_convergence')

# [PLOT RESULTS]
plot_sns(pd.concat([df_divergence,df_convergence]),
         tol=tol,
         title="Heuristic example for `parallel` mode",
         **SNS_PARAMS|{'style':'name', 'hue':'lr', 'size':None, 'alpha':.75})

#%% [6. LOGISTIC REGRESSION]
from sklearn.linear_model import LogisticRegression

mdl_lr = LogisticRegression(C=1/lam, tol=tol, max_iter=max_iter, fit_intercept=True)
mdl_lr.fit(X_train, y_train)
w_lr = np.hstack([mdl_lr.intercept_.reshape([-1,1]), mdl_lr.coef_])
acc_lr = mdl_lr.score(X_test, y_test)
print(f"LR acc: {acc:.2f}")

#%% [7. VFL_LR]
from lr import LRClient, LRCoordinator

# [CLIENTS & COORD]
clients_lr = [LRClient(id=i+1,          # clients should be numbered starting from 1
                       X=X_lst_T[i],    # local data matrix
                       lam=lam,         # l2 reg directly, no need for reciprocal C=1/lam
                       debug=True)      # save client results
                       for i in range(num_nodos)]

coord_lr = LRCoordinator(clients=clients_lr,            # client list
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

coord_lr.fit(max_iter=max_iter, tol=tol, lr=lr, reg_bias=reg_bias, method='gd', taylor=True)
df_gd_taylor = coord_lr.get_results(name='ex_gd_taylor')

# [QUASI-NEWTON BFGS]
coord_lr.fit(max_iter=max_iter, tol=tol, lr=lr, reg_bias=reg_bias, method='bfgs', taylor=taylor)
df_bfgs = coord_lr.get_results(name='ex_bfgs')

coord_lr.fit(max_iter=max_iter, tol=tol, lr=lr, reg_bias=reg_bias, method='bfgs', taylor=True)
df_bfgs_taylor = coord_lr.get_results(name='ex_bfgs_taylor')

# [PLOT RESULTS]: in the same manner as before
df_merged_lr = pd.concat([df_gd, df_gd_taylor, df_bfgs, df_bfgs_taylor])
plot_sns(df_merged_lr, tol=tol, **SNS_PARAMS|{'hue':'name'})
plot_weights_comparison(w_rolann, w_lr, label_w1='ROLANN', label_w2='LR', title='Optim. weights')
plot_weights_comparison(coord.w_, coord_lr.w_, label_w1='BCD ROLANN', label_w2='VFL LR', title='VFL weights')

#%%
