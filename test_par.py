#%% [DATASET]
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Muestras y variables dataset
n, m = 1000, 8

# Regresion
X, y = make_regression(n_samples=n, n_features=m, n_informative=m, bias=1, noise=0.05, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Muestras: {n} | Variables: {m}")

# [VFL] - División de matriz de datos y nodos
from helpers import split_vfl_X
p = 3   # tambien acepta p = [2, 3, 5]
X_lst_T = split_vfl_X(X_train.T, p)

#%% [HIPERPARAMS ROLANN - BENCHMARK]
from rolann import ROLANN
from sklearn.metrics import mean_absolute_error

lam = 0.001
f_act = 'linear'    # regresion
mdl = ROLANN(f_act='linear', lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_
mse = mean_absolute_error(y_test, mdl.predict(X_test))  # 0.039

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

# (secuencial)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='sequential')
res_seq = coord.get_results()
res_seq

# (paralelo)
coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel')
res_par = coord.get_results()
res_par

#%% [BCD ROLANN VERSION ALTERNATIVA]:
# Operaciones eficientes y paralelización de 'precompute' y 'update_weights'
from bcd import BCDClient_par, BCDCoordinator_par

# Creamos 'p' clientes y coordinador
num_nodos = len(X_lst_T)
lst_clients_alt = [BCDClient_par(id=i+1, X=X_lst_T[i], f_act=f_act, lam=lam) for i in range(num_nodos)]
coord_alt = BCDCoordinator_par(clients=lst_clients_alt, d=y_train, f_act=f_act, test_data=(X_test.T, y_test), w_rolann=w_rolann)

# Entrenamiento (mismos params)

# (secuencial)
coord_alt.fit(max_iter=max_iter, tol=tol, lr=lr, mode='sequential')
res_seq_alt = coord_alt.get_results()
res_seq_alt

# (paralelo)
coord_alt.fit(max_iter=max_iter, tol=tol, lr=lr, mode='parallel')
res_par_alt = coord_alt.get_results()
res_par_alt

#%% [TEST EQUIVALENCIA PRECOMPUTE]
import numpy as np

c1 = lst_clients[0]
c1_alt = lst_clients_alt[0]

USVF = c1.USVF
USVF_alt = c1_alt.USVF
assert np.array_equal(USVF, USVF_alt)

#%% [TEST EQUIVALENCIA RESULTADOS]
import numpy as np

# Aproximadamente equivalentes en cuando a diff_w, diff_Z y diff_w_rolann
cols = ['diff_w', 'diff_Z', 'diff_w_rolann']
seq_diff = res_seq[cols] - res_seq_alt[cols]
par_diff = res_par[cols] - res_par_alt[cols]

seq_diff, par_diff = seq_diff.astype(float), par_diff.astype(float)
assert sum(np.linalg.vector_norm(seq_diff, axis=0)) == 0
assert sum(np.linalg.vector_norm(par_diff, axis=0)) == 0
# %%