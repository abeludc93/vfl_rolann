#!/usr/bin/env python3
"""
tests/test_regression.py
~~~~~~~~~~~~~~~~~~~~~~~~
Equivalent to test_classification.py but for a regression task.
It also contains an example of multi-output regression and a
grid-search experiment with a toy-dataset.
"""

#%% [DATASET]
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Muestras y variables dataset
n, m = 1000, 8

# Regresion
X, y = make_regression(n_samples=n, n_features=m, n_informative=m, bias=1, noise=0.05, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

#%% [VFL] - División de matriz de datos y nodos
from helpers import split_vfl_X

# Nodos/clientes 'p':
#   si es un nº natural, dividimos equitativamente el nº de variables, m_i = m // p
#   si es una lista, dividimos según el nº de cada elemento, m_i = p_i | \sum{p_i} = m
# Es necesario mandar X de dimensiones (m, n)
p = 3   # p = [2, 3, 5]
X_lst_T = split_vfl_X(X_train.T, p)
# La lista devuelta contiene los X_i (matrices de datos locales de cada cliente) (m_i, n)

#%% [HIPERPARAMS ROLANN - BENCHMARK]
from rolann import ROLANN
from sklearn.metrics import mean_absolute_error

lam = 0.001
f_act = 'linear'    # regresion
mdl = ROLANN(f_act='linear', lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_.reshape([-1, 1])
mse = mean_absolute_error(y_test, mdl.predict(X_test))  # 0.03970

#%% [BCD ROLANN]
from bcd import BCDClient, BCDCoordinator

# Creamos 'p' clientes y coordinador
num_nodos = len(X_lst_T)
lst_clients = [BCDClient(id=i+1, X=X_lst_T[i], f_act=f_act, lam=lam) for i in range(num_nodos)]
coord = BCDCoordinator(clients=lst_clients, d=y_train, f_act=f_act, test_data=(X_test.T, y_test), w_rolann=w_rolann)

# Entrenamiento
max_iter = 20
tol = 1e-4
lr = 0.75

#%% (secuencial)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='sequential')
res_seq = coord.get_results()
res_seq

#%% (paralelo)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel')
res_par = coord.get_results()
res_par

#%% [GRAFICAS RESULTADOS]
import pandas as pd
from plotters import plot_diff, plot_score, plot_mode_diffs, plot_mode_scores

df_res = pd.concat([res_seq, res_par], ignore_index=False)  # Nunca resetear los índices: llevan la cuenta de las iters de cada experimento

#%% 1. Graficas para cada experimento individual
plot_diff(res_seq)
plot_score(res_seq)

#%% 2. Para comparar resultados según tipo de modo
plot_mode_diffs(df_res)
plot_mode_scores(df_res)

#3. Idem pero para otros hiperparams (lr, lam, etc)

#%% [GRIDSEARCH]
from helpers import run_grid_search

param_grid = {
    'f_act': ['linear'],
    'lam': [0.001],                         # regularization

    'p': [2, 3],                            # node config
    'mode_bcd': ['parallel', 'sequential'], # modes
    'lr': [0.5, 0.75, 1.0],                 # learning rates
    'tol': [1e-5],                          # tolerance
    'max_iter': [50]                        # max iterations
}

lst_exps = run_grid_search(X_train, y_train, X_test, y_test, param_grid, mode='regression')

# %%
