# %% [ROLANN] - CLASIFICACION - PARAMS PRINCIPALES
f_act   = 'logs'
lam     = 1e-3  # Mejor score ROLANN con lam=1e2 -> mejor score LR con C=10000 => lam=1e-4
rolann_params = {
                    'f_act':   f_act,
                    'lam':     lam
                }

# %% [DATASET] - division 70/30
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers import load_dataset
import numpy as np

# Carga
X, y = load_dataset("datasets/clasificacion/FedHEONN/Skin.xlsx")

# Mapeamos clase positiva
y -= 1.  # usar (y == CLASE_POS).astype(int)

# División datos
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
labels = np.unique(y)

# Normalización
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(X_train_orig)

Xtrain = scaler.transform(X_train_orig)
Xtest  = scaler.transform(X_test_orig)

X_train = Xtrain.to_numpy()
X_test  = Xtest.to_numpy()
y_train = y_train_orig.to_numpy()
y_test  = y_test_orig.to_numpy()

# %% [CONDICIONAMIENTO] - matriz Hessiana: A = HH^T + \lam I
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")

# %% [ROLANN CENTRALIZADO] - BASELINE 1
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models = fit_rolann_models(X_train, y_train, mode='binary', **rolann_params)

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train, labels)
pred_test   = score2label(score_test, labels)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 92.47 %
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 92.63 %

# Pesos opt. y funcion objetivo
from rolann import compute_loss_J_w
w_rolann = models[0].w_
loss_w_rolann = compute_loss_J_w(w_rolann, X_train, y_train, lam, f_act)

# %% [BCD ROLANN SEQ] - BASELINE 2
from helpers import fit_bcd_models

# Número de nodos y params entrenamiento
bcd_params = {
                'p':        3,
                'max_iter': 100,
                'tol':      1e-6,
                'lr':       1,
                'mode_bcd': 'sequential',
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 92.47 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 92.63 %

# Resultados
df_seq = [coord.get_results() for coord in coords]
df_seq = df_seq[0]  # Unico dataframe al ser clasif. binaria -> coords[0].get_results()

# Incluimos datos sobre loss y optgap
from helpers import optimality_gap
loss                = df_seq.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
df_seq['loss']      = loss
df_seq['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)

#%%==========================================================
import matplotlib.pyplot as plt
def plot_convergencia(df_seq, df_bcd, grid, col='diff_Z', plot_tol=False, plot_log=False):
   fig, axes = plt.subplots()
   if plot_tol:    axes.axhline(grid['tol'], color='grey', linestyle='--', label='tol')
   if plot_log:    axes.set_yscale('log')

   axes.plot(df_seq.iter, df_seq[col], label='SEQ', color='k', linestyle=':')
   for bcd in df_bcd:
      axes.plot(bcd.iter, bcd[col], label=f"lr - {bcd.lr.values[0]}", alpha=0.75)

   axes.set_title('Convergencia de modo PAR con diferentes lr y modo SEQ')
   axes.set_xlabel('Iter')
   axes.set_ylabel(col)
   axes.legend()
   axes.grid()
   fig.suptitle(f"BCD nodos: {grid['p']}")
   plt.show()

#%% ======================== PRUEBAS ========================
"""
Definimos SPLITs para este este dataset:
a) 3 NODOS: una variable en cada nodo.
b) 2 NODOS: [B, G] y [R].
c) 2 NODOS: [G, R] y [B].
d) 2 NODOS: [R, B] y [G].
"""
from helpers import run_grid_search

#%% ===[a) Learning rates modo PARALELO -> convergencia hacia modo SEQ]===
rolann_grid = {k:[v] for k,v in rolann_params.items()}
param_grid  = {
        # BCD params
        'p': [3],
        'mode_bcd': ['parallel'],               # modes
        'lr': [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95],    # learning rates
        'tol': [1e-6],                          # tolerance
        'max_iter': [100],                      # max iterations
    }
exps1 = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
lst_rolann1, lst_bcd1 = list(zip(*exps1))

# Preparamos dataframes de cada experimento
from itertools import chain
lst_bcd1 = list(chain(*lst_bcd1))                       # flatten - equivalente a coger cada lst_bcd[*][0]
df_bcd1 = [coord.get_results() for coord in lst_bcd1]   # guardamos dataframe de resultados de cada coord

# Calculamos loss y optgap
from helpers import optimality_gap, total_time
for i, df in enumerate(df_bcd1):
    loss                    = df.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_bcd1[i]['loss']      = loss
    df_bcd1[i]['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)
    df_bcd1[i]['total_time']= total_time(df)

#%% ===[a) Convergencia]=== 
opts = {
    'col':       'diff_Z',
    'plot_log':  True,
    'plot_tol':  True
   }
plot_convergencia(df_seq, df_bcd1[:-2], param_grid, **opts)

#%% ===[b) GridSearch 2 Nodos -> SEQ y PAR]===
param_grid['p'] = [[2, 1]]
param_grid['lr'] = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
param_grid['mode_bcd'] = ['parallel']
exps2 = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
lst_rolann2, lst_bcd2 = list(zip(*exps2))
lst_bcd2 = list(chain(*lst_bcd2))  
df_bcd2 = [coord.get_results() for coord in lst_bcd2]

# Calculamos loss y optgap
from helpers import optimality_gap, total_time
for i, df in enumerate(df_bcd2):
    loss                    = df.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_bcd2[i]['loss']      = loss
    df_bcd2[i]['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)
    df_bcd2[i]['total_time']= total_time(df)

param_grid['lr'] = [1]
param_grid['mode_bcd'] = ['sequential']
exps2_base = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
df_seq2 = exps2_base[0][1][0].get_results()

# Incluimos datos sobre loss y optgap
from helpers import optimality_gap
loss                 = df_seq2.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
df_seq2['loss']      = loss
df_seq2['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)

#%% ===[b) Graficas Convergencia]===
opts = {
    'col':       'diff_Z',
    'plot_log':  True,
    'plot_tol':  True
   }
plot_convergencia(df_seq2, df_bcd2, param_grid, **opts)

#%% ===[c) GridSearch 2 Nodos -> SEQ y PAR]===
split_c = {0: [1, 2], 1: [0]}
param_grid['p'] = [split_c]
param_grid['lr'] = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
param_grid['mode_bcd'] = ['parallel']
exps3 = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
lst_rolann3, lst_bcd3 = list(zip(*exps3))
lst_bcd3 = list(chain(*lst_bcd3))  
df_bcd3 = [coord.get_results() for coord in lst_bcd3]

# Calculamos loss y optgap
from helpers import optimality_gap, total_time
for i, df in enumerate(df_bcd3):
    loss                    = df.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_bcd3[i]['loss']      = loss
    df_bcd3[i]['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)
    df_bcd3[i]['total_time']= total_time(df)

param_grid['lr'] = [1]
param_grid['mode_bcd'] = ['sequential']
exps3_base = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
df_seq3 = exps3_base[0][1][0].get_results()

# Incluimos datos sobre loss y optgap
from helpers import optimality_gap
loss                 = df_seq3.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
df_seq3['loss']      = loss
df_seq3['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)

#%% ===[c) Graficas Convergencia]===
opts = {
    'col':       'diff_Z',
    'plot_log':  True,
    'plot_tol':  True
   }
plot_convergencia(df_seq3, df_bcd3, param_grid, **opts)

#%% ===[d) GridSearch 2 Nodos -> SEQ y PAR]===
split_d = {0: [0, 2], 1: [1]}
param_grid['p'] = [split_d]
param_grid['lr'] = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95]
param_grid['mode_bcd'] = ['parallel']
exps4 = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
lst_rolann4, lst_bcd4 = list(zip(*exps4))
lst_bcd4 = list(chain(*lst_bcd4))  
df_bcd4 = [coord.get_results() for coord in lst_bcd4]

# Calculamos loss y optgap
from helpers import optimality_gap, total_time
for i, df in enumerate(df_bcd4):
    loss                    = df.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_bcd4[i]['loss']      = loss
    df_bcd4[i]['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)
    df_bcd4[i]['total_time']= total_time(df)

param_grid['lr'] = [1]
param_grid['mode_bcd'] = ['sequential']
exps4_base = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
df_seq4 = exps4_base[0][1][0].get_results()

# Incluimos datos sobre loss y optgap
from helpers import optimality_gap
loss                 = df_seq4.w_i.apply(func=compute_loss_J_w, X=X_train, y=y_train, lam=lam, f_act=f_act)
df_seq4['loss']      = loss
df_seq4['opt_gap']   = optimality_gap(loss, min_loss=loss_w_rolann)

#%% ===[d) Graficas Convergencia]===
opts = {
    'col':       'diff_Z',
    'plot_log':  True,
    'plot_tol':  True
   }
plot_convergencia(df_seq4, df_bcd4, param_grid, **opts)

#%%