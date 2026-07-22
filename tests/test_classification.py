#!/usr/bin/env python3
"""
tests/test_classification.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example of a binary classification task with ROLANN/BCD_ROLANN
and also with LogisticRegression/VFL_LR for a synthethic toy-dataset.
It also contains an example of a multi-label classification task and
a grid-search experiment.
"""
#%% DATASET CLASIFICACION
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Muestras y variables dataset
n, m = 10000, 10

# Clasificacion
#X, y = make_classification(n_samples=n, n_features=m, n_classes=2, n_informative=m, n_redundant=0, n_clusters_per_class=2, random_state=1)
X, y = make_classification(n_samples=n, n_features=m, n_classes=2, weights=[0.5, 0.5], n_informative=6, n_redundant=4, random_state=1)
#X = X + 5.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")
print(f"TRAIN BAL: {np.unique_counts(y_train)}")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
Xtrain = scaler.transform(X_train)
Xtest = scaler.transform(X_test)

#%% [VFL] - División de matriz de datos y nodos
from helpers import split_vfl_X
# Nodos/clientes 'p':
#   si es un nº natural, dividimos equitativamente el nº de variables, m_i = m // p
#   si es una lista, dividimos según el nº de cada elemento, m_i = p_i | \sum{p_i} = m
# Es necesario mandar X de dimensiones (m, n)
p = [2, 2, 6]
X_lst_T = split_vfl_X(Xtrain.T, p)
num_nodos = len(X_lst_T)
# La lista devuelta contiene los X_i (matrices de datos locales de cada cliente) (m_i, n)

#%% [HIPERPAMS ROLANN - BENCHMARK]
from rolann import ROLANN
from sklearn.metrics import accuracy_score
from helpers import score2label

lam = 0.001
f_act = 'logs'    # clasificacion

mdl = ROLANN(f_act=f_act, lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_
y_pred = score2label(mdl.predict(X_test))
acc = accuracy_score(y_test, y_pred)    # 0.785

#%% [BCD ROLANN]
from bcd import BCDClient, BCDCoordinator

# Clientes y coordinador
lst_clients = [BCDClient(id=i+1, X=X_lst_T[i], f_act=f_act, lam=lam) for i in range(num_nodos)]
coord = BCDCoordinator(clients=lst_clients, d=y_train, f_act=f_act, test_data=(X_test.T, y_test), w_rolann=w_rolann)

# Entrenamiento
max_iter = 40
tol = 1e-4
lr = 0.75

#%% (secuencial)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='sequential')
res_seq = coord.get_results()
res_seq

#%% (paralelo)
#coord.reset(hard=True)  # para resetear precomputo clientes y tener en cuenta t_setup
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

#%% [TEST PARA BCD LRP] - Heuristica para learning rate del modo PARALELO
from bcd import BCDCoordinator_lrp

lr = 0.85
coord_lrp = BCDCoordinator_lrp(clients=lst_clients, d=y_train, f_act=f_act, test_data=(X_test.T, y_test), w_rolann=w_rolann)
coord_lrp.reset(hard=True)
coord_lrp.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel', lr_heuristic=True)
res_lrp = coord_lrp.get_results()
res_lrp

# Para obtener los lr por iteración válida
df_lr = res_lrp.groupby('iter', as_index=False).last()

# %% ===========================================
# Logistic Regression
from sklearn.linear_model import LogisticRegression

max_iter = 1000
tol = 0
lam = 1e-3
C = 1/lam
lr_mdl = LogisticRegression(C=C, tol=tol, max_iter=max_iter, fit_intercept=True)
lr_mdl.fit(Xtrain, y_train)

lr_w = np.hstack([lr_mdl.intercept_.reshape([-1,1]), lr_mdl.coef_])
lr_acc = lr_mdl.score(Xtest, y_test)

#%% Visualización vs. pesos ROLANN centralizado
from plotters import plot_weights_comparison
plot_weights_comparison(lr_w.T, w_rolann, label_w1='LR', label_w2='ROLANN')

# %% ==============================================================
from lr import LRClient, LRCoordinator
# Params
lr = 0.75
#lam_vfl = lam/len(X_train)     # 1/(C*len(X_train))
#lr_clients = [LRClient(id=i+1, X=X_lst_T[i], lam=lam_vfl) for i in range(num_nodos)]
lr_clients = [LRClient(id=i+1, X=X_lst_T[i], lam=lam) for i in range(num_nodos)]
lr_coord = LRCoordinator(clients=lr_clients, y=y_train, test_data=(Xtest.T, y_test), w_lr=lr_w.T)

#%% Entrenamiento GD sin aprox. Taylor
lr_coord.fit(max_iter=1000, tol=tol, lr=lr, method='gd', taylor=False, reg_bias=False) 
#df_lr = lr_coord.get_results()
#df_lr

# %% Entrenamiento GD con aprox. Taylor
lr_coord.fit(max_iter=max_iter, tol=tol, lr=lr, taylor=True)
df_lr_t = lr_coord.get_results()
df_lr_t

#%% Visualización pesos
from plotters import plot_weights_comparison
plot_weights_comparison(lr_w.T, lr_coord.w_, label_w1='LR', label_w2='VFL')

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


#%%
"""
El tipo de comunicación es el mismo que para BCD ROLANN.
Ida y vuelta de un vector por cada iteración y nodo.

El modo que hace uso de aproximaciones de Taylor está pensando exclusivamente para
su uso real en un entorno federado donde se debe aplicar algún tipo de encriptación
homomorfica, ya que esquemas como CKKS no permitirían la evaluación de una f. activación
como la sigmoide con texto cifrado o la evaluación de la f. de pérdida (negloglikelihood).

Por eso se aplican tales aproximaciones polinomicas que sí permiten hacer estos calculos
sobre texto cifrado.

Por tanto, en una implementación futura tipo LRCoordinator_fhe
"""

# %% Entrenamiento BFGS sin aprox. Taylor
lr_coord.fit(max_iter=max_iter, tol=tol, lr=lr, method='bfgs')
df_lr_bfgs = lr_coord.get_results()
df_lr_bfgs

# %% Entrenamiento BFGS con aprox. Taylor
lr_coord.fit(max_iter=max_iter, tol=tol, lr=lr, method='bfgs', taylor=True)
df_lr_bfgs_t = lr_coord.get_results()
df_lr_bfgs_t

#==============================================================
#%% Prueba rapida de clasificación multi-clase
#==============================================================
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
n, m = 800, 8
X, y = make_classification(n_samples=n, n_features=m, n_classes=3, n_informative=m, n_redundant=0, n_clusters_per_class=2,random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

# %% Logistic Regression
from sklearn.linear_model import LogisticRegression
import numpy as np

max_iter = 40
tol = 1e-4
lam = 1e-3
lr_mdl = LogisticRegression(penalty='l2', C=1/lam, tol=tol, max_iter=max_iter, multi_class='ovr')
lr_mdl.fit(X_train, y_train)
# Funciona tal cual antes: hay un aviso por deprecated en futuras versiones recomendando usar
# OneVsRestClassifier(lr_mdl) a partir de sklearn 1.7
lr_w = np.hstack([lr_mdl.intercept_.reshape([-1, 1]), lr_mdl.coef_])
lr_acc = lr_mdl.score(X_test, y_test)

# %% Logistic Regression VFL clasificación multi-clase
from lr import LRClient, LRCoordinator
from helpers import split_vfl_X

y_train = y_train.reshape([-1, 1])
y_test = y_test.reshape([-1, 1])

p = 3
# Dividimos 'X' para los distintos clientes
X_lst_T = split_vfl_X(X_train.T, p)

# Preparamos targets 'y' en caso de clasificación OvA
classes = np.unique(y_train)
assert len(classes) == len(np.unique(y_test))    # mismas clases en train que en test
y_target = np.hstack([(y_train == cls).astype(int) for cls in classes])
y_target_test = np.hstack([(y_test == cls).astype(int) for cls in classes])

#%% Clientes y coordinador
num_nodos = len(X_lst_T)
num_outputs = y_target.shape[1]

lst_clients = [LRClient(id=i+1, X=X_lst_T[i], lam=lam)
                for i in range(num_nodos)]
lst_coords = [LRCoordinator(clients=lst_clients, y=y_target[:,n], test_data=(X_test.T, y_target_test[:,n]), w_lr=lr_w.T)
                for n in range(num_outputs)]

#%% Fit - params - GD y sin aprox. Taylor
lam = 1e-3
max_iter = 40
tol = 1e-4
lr = 0.5
for i, coord in enumerate(lst_coords):
    print(f"Entrenando LR VFL para clase-salida {i+1}/{num_outputs}...")
    coord.fit(max_iter=max_iter, tol=tol, lr=lr)
dfs = [coord.get_results() for coord in lst_coords]

# %% Resultados
lr_w_vfl = np.vstack([c.w_.T for c in lst_coords])
from helpers import predict_models, score2label
from sklearn.metrics import accuracy_score

score_train = predict_models(X_train.T, lst_coords)
score_test  = predict_models(X_test.T, lst_coords)
pred_train  = score2label(score_train, classes) # habría que retocar el POS_VALUE|NEG_VALUE ya que LR usa (1,0), no hay mapeo
pred_test   = score2label(score_test, classes)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

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

#%%