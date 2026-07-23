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

#=========================================================================================
#%%~~~~~~~~ REGRESION~~~~~~~~~~
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m = 1000, 8

# Clasificacion
X, y = make_regression(n_samples=n, n_features=m, n_targets=1, bias=500, n_informative=6, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")
print(f"y_max: {y_train.max():.2f}, y_min: {y_train.min():.2f}")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
Xtrain = scaler.transform(X_train)
Xtest = scaler.transform(X_test)

# Centrado de datos
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_scaler = StandardScaler(with_std=False).fit(y_train)

ytrain = y_scaler.transform(y_train)
ytest = y_scaler.transform(y_test)
print(f"SCALED | y_max: {ytrain.max():.2f}, y_min: {ytrain.min():.2f}")

# Split 
from helpers import split_vfl_X
p = [2, 2, 4]
X_lst_T = split_vfl_X(Xtrain.T, p)
num_nodos = len(X_lst_T)

#%% Ridge
from sklearn.linear_model import Ridge

max_iter = 200
tol = 1e-4
alpha = 1000
lr_mdl = Ridge(alpha=alpha, tol=tol, max_iter=max_iter)
lr_mdl.fit(Xtrain, ytrain)

lr_w = np.hstack([lr_mdl.intercept_, lr_mdl.coef_])
from sklearn.metrics import mean_absolute_error
lr_mse = mean_absolute_error(ytest, lr_mdl.predict(Xtest))

#%% VFL Ridge
from lr import RidgeClient, RidgeCoordinator
# Params
lr = 0.5
#lam_vfl = alpha/len(X_train)
# sklearn NO incluye el 1/2​ en el término del error de los datos en su f.obj.
# por eso la equivalencia sale lam_vfl = alpha/N y no 2*alpha/N
# mejor por que asi coincide con al equivalencia que hay que hacer entre LR/VFLLR
# al usar lam_vfl = 1/(C*N) = lam/N

# (el alpha de ridge es el valor de la reg l2... osea que es como nuestro lam)

#lr_clients = [RidgeClient(id=i+1, X=X_lst_T[i], lam=lam_vfl) for i in range(num_nodos)]
lr_clients = [RidgeClient(id=i+1, X=X_lst_T[i], lam=alpha) for i in range(num_nodos)]
lr_coord = RidgeCoordinator(clients=lr_clients, y=ytrain, test_data=(Xtest.T, ytest), w_lr=lr_w)

#%% Entrenamiento GD sin aprox. Taylor
lr_coord.fit(max_iter=500, tol=tol, lr=lr, method='gd', reg_bias=True)
df_lr = lr_coord.get_results()
df_lr

#%%
"""
Se implementa Ridge y RidgeVFL para tareas de regresión.
El código es casi igual (VFL) que para LR, solo es necesario ajustar la f.act y la f.obj.

Además, se ha añadido la opción de regularizar el bias en ambos caso (LR y Ridge), para obtener
resultados y pesos más parecidos a los dados por sklearn LogisticRegression y Ridge.

Cuando los datos no están normalizados ni centrados (desbalanceo en clasif. o sin escalar en regresión),
podemos ver que tanto Ridge como LogisticRegression son capaces de ajustar el modelo satisfactoriamente
ya que no penalizan el bias (no lo regularizan), por lo que puede tener una escala muy diferente al resto
de pesos.

Esto se hace automáticamente siempre que se instancien con `fit_intercept=True`. De no hacerlo,
el modelo entrenado devuelve unos pesos con el bias a 0.

En nuestra implementación VFL de LR y Ridge teniamos la regularización L2 metida tanto en el
computo de la f.objetivo como en la actualización de los pesos del cliente durante el bucle
de entrenamiento (donde se calcula el gradiente). Esto penalizaba la escala del bias de manera
que no se alcanzaban los mismos pesos finales que en el caso centralizado.

Si entrenamos con `reg_bias=False` esto no sucede; y los pesos, scores y losses son equiparables.

Volviendo a ROLANN/BCD ROLANN, esta distinción no aplica, ya que ROLANN se basa en una solución
analítica a un sist. de ecuaciones lineales subyacente derivado del desarrollo de la f. objetivo
MSEA (MSE antes de la f.act.); y donde se hace uso la factorización SVD sobre la matriz de fact.
H = X @ X.T para obtener los pesos óptimos.

Especificamente, partiendo de la solución de la ec.(6) de VFL_FedHEONN.pdf,

    w = (H H^T + lam I)^-1         H F d_bar

Se llega a la ec.(8),
    w = U(S^2 + lam I)^-1U^T       H F d_bar

Ese factor comun que se hace para meter la regularización L2 (lamI) entre U y U^T no sería
posible si decidiesemos usar una I_alt := diag(0, 1, 1, 1,...)

Por tanto, la regularización del bias es intrínseco a la solución de ROLANN/FedHEONN/BCD ROLANN.

Para mitigar los efectos negativos que esto pudiese tener, lo que hay que hacer es ser más
restrictivos que sklearn (ML empirico, out-of-the-box, funciona bien incluso sin datos centrados),
y OBLIGAR a estándarizar los datos de entrar y a CENTRAR LOS DATOS!

"""

