#!/usr/bin/env python3
"""
tests/test_multi.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example of multi-output/label with ROLANN/BCD_ROLANN and
and LR/VFL_LR for both classification/regression tasks.
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

#%% [1. DATASET - MULTI-LABEL CLASSIFICATION]
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m    = 10000, 10
n_red   = 2

X, y = make_classification(n_samples=n, n_features=m, n_classes=3, n_informative=m-n_red, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")
print(f"TRAIN BAL: {np.unique_counts(y_train).values} -> {np.unique_counts(y_train).counts}")

#%% [2. ROLANN]
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from rolann import ROLANN
from helpers import score2label

# Hyperparams
lam = 0.001                         # l2 reg
f_act = 'logs'                      # logsig -> clasif

# One-hot and manual OvR:
classes = np.unique(y)
y_train_one = label_binarize(y_train, classes=classes)
y_test_one  = label_binarize(y_test, classes=classes)
mdl_lst = []

# Fit
for i in range(len(classes)):
    mdl = ROLANN(f_act=f_act, lam=lam)
    mdl.fit(X_train, y_train_one[:,i])
    mdl_lst.append(mdl)

# Score
w_rolann= np.hstack([mdl.w_ for mdl in mdl_lst])
pred    = np.hstack([mdl.predict(X_test) for mdl in mdl_lst])   # shape (2000, 3)
y_pred  = score2label(pred, labels=classes, f_act=f_act)

acc = accuracy_score(y_test, y_pred)
print(f"ROLANN acc: {acc:.2f}")

# There is a fully functional ROLANN model compatible with the `sklearn` estimator API at
# https://github.com/abeludc93/rolann that natively supports multi-label/output clasif/regression.

#%% [3. LOGISTIC REGRESSION]
from sklearn.linear_model import LogisticRegression

mdl_lr = LogisticRegression(penalty='l2', C=1/lam, multi_class='ovr')
mdl_lr.fit(X_train, y_train)
w_lr = np.hstack([mdl_lr.intercept_.reshape([-1, 1]), mdl_lr.coef_])
acc_lr = mdl_lr.score(X_test, y_test)
print(f"LR acc: {acc_lr:.2f}")

#%% [4. VFL SPLIT]
from helpers import split_vfl_X
p = 3  # -> 3 clientes con 3, 3, y 4 variables
X_lst_T = split_vfl_X(X_train.T, p)
num_nodos = len(X_lst_T)

#%% [5. BCD_ROLANN]
from helpers import predict_models, score2label
from sklearn.metrics import accuracy_score
from bcd import BCDClient, BCDCoordinator
num_outputs = len(classes)

# [CLIENTS & COORDINATOR]
clients = [BCDClient(id=i+1,          # clients should be numbered starting from 1
                     X=X_lst_T[i],    # local data matrix
                     lam=lam,         # l2 reg no need for reciprocal C
                     debug=True)      # save client history
                for i in range(num_nodos)]

coords  = [BCDCoordinator(clients=clients,                       # client list shared among coords
                          d=y_train_one[:, n],                   # real/desired outputs
                          f_act='logs',                          # act fn
                          test_data=(X_test.T, y_test_one[:, n]),# test_data
                          w_rolann=w_rolann[:, n])               # optim. weights for LR
                for n in range(num_outputs)]                     # as many coords as outputs

# [FIT PARAMS]
max_iter    = 100
lam         = 0.001
tol         = 1e-4
lr          = 0.75
init        = 'zero'
heur        = False

# [RUN]
for i, coord in enumerate(coords):
    print(f"Training BCD_ROLANN on class/output {i+1}/{num_outputs}...")
    coord.fit(max_iter=max_iter, tol=tol, lr=lr, init=init, heur=heur, mode='parallel')
dfs_bcd = [c.get_results(name=f'class_{i+1}') for i, c in enumerate(coords)]

# [SCORE]
w_vfllr     = np.hstack([c.w_ for c in coords])
pred_train  = predict_models(X_train.T, coords)
pred_test   = predict_models(X_test.T, coords)
ypred_train = score2label(pred_train, classes, f_act='logs')
ypred_test  = score2label(pred_test, classes, f_act='logs')
acc_train   = accuracy_score(y_train, ypred_train)
acc_test    = accuracy_score(y_test, ypred_test)
print(f"BCD_ROLANN acc: {acc_test:.2f}")

# [RESULTS]
dfs_bcd   = [c.get_results(name=f'output_{i}') for i, c in enumerate(coords)]


#%% [6. VFL_LR]
from lr import LRClient, LRCoordinator

# [CLIENTS & COORDINATOR]
clients_lr = [LRClient(id=i+1,          # clients should be numbered starting from 1
                       X=X_lst_T[i],    # local data matrix
                       lam=lam,         # l2 reg no need for reciprocal C
                       debug=True)      # save client history
                for i in range(num_nodos)]

coords_lr  = [LRCoordinator(clients=clients_lr,                     # client list shared among coords
                            y=y_train_one[:, n],                    # real/desired outputs
                            test_data=(X_test.T, y_test_one[:, n]), # test_data
                            w_lr=w_lr.T[:, n])                      # optim. weights for LR
                for n in range(num_outputs)]                        # as many coords as outputs

# [FIT PARAMS]
max_iter    = 500
lam         = 0.001
tol         = 1e-4
lr          = 0.75
reg_bias    = True
taylor      = False

# [RUN]
for i, coord in enumerate(coords_lr):
    print(f"Training VFL_LR on class/output {i+1}/{num_outputs}...")
    coord.fit(max_iter=max_iter, tol=tol, lr=lr, reg_bias=reg_bias, taylor=taylor, method='gd')
dfs_lr = [c.get_results(name=f'class_{i+1}') for i, c in enumerate(coords_lr)]

# [SCORE]
w_vfllr     = np.hstack([c.w_ for c in coords_lr])
pred_train  = predict_models(X_train.T, coords_lr)
pred_test   = predict_models(X_test.T, coords_lr)
ypred_train = score2label(pred_train, classes, f_act='logs')
ypred_test  = score2label(pred_test, classes, f_act='logs')
acc_train   = accuracy_score(y_train, ypred_train)
acc_test    = accuracy_score(y_test, ypred_test)
print(f"VFL_LR acc: {acc_test:.2f}")

# [RESULTS]
dfs_vfllr   = [c.get_results(name=f'output_{i}') for i, c in enumerate(coords_lr)]

#%% [7. PLOTTERS]:
from plotters import plot_sns, plot_score_global
from helpers import obtain_global_scores
import matplotlib.pyplot as plt

# This dataframe contains both OvR (individual) and global metrics per iteration 
global_bcd_scores = obtain_global_scores(dfs_bcd,                       # list of coord results
                                         train_data=(X_train.T, y_train), # train data
                                         test_data=(X_test.T, y_test),    # test data
                                         f_act='logs',                    # act fn used
                                         skmetricfn=accuracy_score,       # callable metric
                                         labels=classes)                  # class labels

global_vfllr_scores = obtain_global_scores(dfs_vfllr,                       # list of coord results
                                           train_data=(X_train.T, y_train), # train data
                                           test_data=(X_test.T, y_test),    # test data
                                           f_act='logs',                    # act fn used
                                           skmetricfn=accuracy_score,       # callable metric
                                           labels=classes)                  # class labels

# Example of subplots with seaborn capabilities through `plot_sns`
fig, axes = plt.subplots(4, 2, figsize=(10, 20))
for i in range(num_outputs):
    plot_sns(global_bcd_scores, plot_log=False, tol=None, title='BCD_ROLANN',
             **{'x':'iter', 'y':f'score_test_{i+1}', 'ax':axes[i, 0], 'legend':False})
    plot_sns(global_vfllr_scores, plot_log=False, tol=None, title='VFL_LR',
             **{'x':'iter','y':f'score_test_{i+1}', 'ax':axes[i, 1], 'legend':False})
plot_sns(global_bcd_scores, plot_log=False, tol=None, title='BCD_ROLANN global',
         **{'x':'iter','y':'score_test', 'ax':axes[3, 0], 'legend':False})
plot_sns(global_vfllr_scores, plot_log=False, tol=None, title='VFL_LR global',
         **{'x':'iter','y':'score_test', 'ax':axes[3, 1] ,'legend':False})

# Same graph plotting the different scores on the same ax
plot_score_global(global_bcd_scores, classes, title="BCD ROLANN")
plot_score_global(global_vfllr_scores, classes, title="VFL LR")


#%% [FINAL]
#   Multi-output regression is an analogous process requiring a 'linear' act. function
#   for BCD_ROLANN and instantiating VFL Ridge (RidgeClient, RidgeCoordinator) instead
#   of VFL_LR.
#   `helpers.py` contains wrapper functions for the instantiation, fitting and scoring
#   of these pairs of VFL models, namely: `fit_bcd_models`, `fit_lrvfl_models`, `run_g
#   rid_search` and `run_grid_search_lr`. See `tests/test_run.py` for an usage example.
