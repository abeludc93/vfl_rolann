# %% [ROLANN] - CLASIFICACION - PARAMS PRINCIPALES
rolann_params = {
                    'f_act':   'logs',
                    'lam':     1e-3
                }

#%% DATASET CLASIFICACION
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Muestras y variables dataset
n, m = 1000, 8

# Clasificacion
X, y = make_classification(n_samples=n, n_features=m, n_classes=2, n_informative=m, n_redundant=0, n_clusters_per_class=2, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Muestras: {n} | Variables: {m}")
print(f"TRAIN: {X_train.shape}/{y_train.shape} | TEST: {X_test.shape}/{y_test.shape}")

# %% [ROLANN CENTRALIZADO]
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models = fit_rolann_models(X_train, y_train, mode='binary', **rolann_params)

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train, labels=None)
pred_test   = score2label(score_test, labels=None)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")
print(f"Accuracy test: {100 * acc_test:.2f} %")

#%% [BCD ROLANN]
from helpers import split_vfl_X
from bcd import BCDClient_sec, BCDCoordinator_sec

f_act   = 'logs'
lam     =  1e-3
p       = 3
X_lst_T = split_vfl_X(X_train.T, p)
num_nodos = len(X_lst_T)
w_rolann = models[0].w_

# Clientes y coordinador
lst_clients = [BCDClient_sec(id=i+1, X=X_lst_T[i], f_act=f_act, lam=lam)
               for i in range(num_nodos)]
coord = BCDCoordinator_sec(clients=lst_clients, d=y_train, f_act=f_act, test_data=(X_test.T, y_test), w_rolann=w_rolann)

# Entrenamiento
max_iter = 40
tol = 1e-4
lr = 0.75

#%% (secuencial)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='sequential')
res_seq = coord.get_results()
res_seq

#%% (paralelo)
coord.reset(hard=True)  # para resetear precomputo clientes y tener en cuenta t_setup
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel')
res_par = coord.get_results()
res_par
# %%
""" SecureAgg vs CKKS
1. Modelos de amenazas distintos:
  a. privacidad en transito (enmascaramos inputs individuales pero la agregacion da el resultado verdadero Z)
  b. privacidad total (el servidor agrega textos cifrados y no puede ver el resultado).

Al final el SecureAgg es una especie de privacidad diferencial mucho mejor.
En un escenario realista habría que ajustar tambien la máscara a la escala de los inputs Z_i de cada cliente, ya que si las máscaras aleatorias que aplica SecureAgg son ínfimas en relación a la escala del
Z_i, poco lo camuflará.
 
"""