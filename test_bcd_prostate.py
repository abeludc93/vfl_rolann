# %% [DATASET] - division entrenamiento/test dada
from sklearn.preprocessing import StandardScaler
from helpers import load_dataset
import numpy as np
import pandas as pd

# Carga
Xtrain, ytrain = load_dataset("datasets/clasificacion/LANNSVD/Prostate_train.xlsx")
Xtest, ytest = load_dataset("datasets/clasificacion/LANNSVD/Prostate_test.xlsx")
labels = np.unique(np.concat([ytrain, ytest]))
print(f"Reparto clases [1|0]: TRAIN - {ytrain.value_counts()}, TEST - {ytest.value_counts()}")

# Preprocesado: se desconoce a que corresponde cada variable ni si posee una separación natural de las mismas

# División de datos: ya viene dada

# Centrado de datos: datos de entrenamiento equilibrados

# Normalización
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(Xtrain)

Xtrain_scaled = scaler.transform(Xtrain)
Xtest_scaled  = scaler.transform(Xtest)

# Numpy
X_train = Xtrain_scaled.to_numpy()
X_test  = Xtest_scaled.to_numpy()
y_train = ytrain.to_numpy()
y_test  = ytest.to_numpy()

# %% [ROLANN HIPERPARAMS]
f_act = 'logs'
lam = 200
rolann_params = {'f_act':f_act, 'lam': lam}
rolann_grid = {k:[v] for k,v in rolann_params.items()}

# %% [CONDICIONAMIENTO]
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(Xtrain, **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
""" ROLANN
Condicionamiento A escalada: 6.72
Condicionamiento A sin escalar: 159800.79
	Reducción en razón del orden de: 2.38e+06
"""
#%% [COMENTARIOS]
""" LR
Para condicionamiento de (X^T @ X): 3.95e+21
     condicionamiento de (X^T @ D @ X): 9.07e+21        (con D=diag(0.25...))

--------------------------------------------
Matriz sistema Ax = b ROLANN
A = H @ H^T = X @ F^2 @ X^T             -> similar a la siguiente
                                        -> ROLANN calcula su inversa analiticamente 1 vez
--------------------------------------------
Hessiana de log-loss:
H_lr = X^T @ D @ X

Donde D:
diag(p_i*(1-p_i)) = diag(p*q)           -> matrices de covarianza ponderadas

La Hessiana describen la curvatura de la f. obj. (prob. optimización)
Aunque se demuestre que la Hessiana de LR también es convexa

Para el gradiente descendiente sea estable: 
$\eta \le \frac{2}{\lam_{max}(Hessiana_{LR})}$

$\lam_{max}(Hess_{LR}) \approx 126815$ 

Y el nº de iteraciones será proporcional al cond(Hessiana_lr)
"""

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from plotters import plot_sns, plot_weights_comparison
sns_params = {
    'x': 'iter',        # Eje X
    'y': 'diff_Z',      # Eje Y
    'hue': 'lr',        # Distinto color para cada Learning Rate
    'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
    'size': 'p',        # Distinto tamaño
    'linewidth': 1.5,   # etc
}

#%% [TESTING ROLANN & BCD ROLANN]
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

print(f"Accuracy train: {100 * acc_train:.2f} %")   #  86.27%
print(f"Accuracy test: {100 * acc_test:.2f} %")     #  91.18%

#%% BCD
from helpers import fit_bcd_models

bcd_params = {
                'p':        12600,
                'max_iter': 1000,
                'tol':      0,
                'mode_bcd': 'seq',
                'lr':       1,
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

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 86.27%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 91.18%

#%% [TESTING LR & VFL LR]
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

lr_params = {
    'lam': 2e4,    # esto es C=1/lam... lam=2e4 CORREGIDO
    'tol': 0,
    'max_iter': 250
}

# Fit
models_lr = fit_lr_models(X_train, y_train, mode='auto', **lr_params)

# Acc
score_lr_train = predict_models(X_train, models_lr)
score_lr_test  = predict_models(X_test, models_lr)
pred_train  = score2label(score_lr_train, labels)
pred_test   = score2label(score_lr_test, labels)

acc_lr_train   = accuracy_score(y_train, pred_train)
acc_lr_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   #  85.29%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     #  91.18%


#%% VFL LR
from helpers import fit_lrvfl_models

vfl_params = {
    'p': 2,             # probado con otros 'p' -> min lr=0.001
    'lr': 0.001,
    'max_iter': 10000,  # override lr_params (meter > 10000)
    'method': 'gd',
    'taylor': False,
    'w_lr': models_lr[0].w_,
    'test_data': (X_test.T, y_test)
}

# Fit
coords, clients = fit_lrvfl_models(X_train, y_train, **(lr_params|vfl_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 86.21%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 91.18%

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% [ESCENARIOS VFL]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch, compute_extra_results
df_all_lst = []

#%% (2) - Split balanceado - BCD ROLANN
m, m_min = X_train.shape[1], 2
p_bal = [2, 20, 200, 2000, 12600]

# BCD params
param_grid  = {
    'p': p_bal,                                     # Nodos
    'mode_bcd': ['parallel', 'sequential'],         # Modos
    'lr': [.2],                                     # Learning rates modo paralelo
    'tol': [0],                                     # Tolerancia : hasta max_ites
    'max_iter': [1000],                             # Max iteraciones
    }
print(param_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
w_rolann = exps[0][0][0].w_

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_bal')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_bal = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

#%% (2) - Split balanceado - LR VFL
lam_lr = 2e4
lr_grid = {
    'lam': [lam_lr],        # Regularización L2
    'tol': [0],
    'max_iter': [10000]
    }
vfl_grid = {
    'p': p_bal,
    'lr': [.001],
    'method': ['gd'],       # sin bfgs -> O(m^2)
    'taylor': [False]
    }

# Lanzamos experimento
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='auto')   

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_bal')
w_lr = exps_lr[0][0][0].w_

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_lr_bal = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)

# Guardamos en lista de resultados ambos experimentos
df_all_lst.extend([df_bal, df_lr_bal])


#%% (4) - Split desbalanceado - IMPORTANCIA -
# Aunque el split por importancia es mas eficiente computacionalmente, igualmente
# cacheamos los índices de las divisiones realizadas para futuras ocasiones
alpha_lst = [.1, 1, 10]
p_desbal = [682, 36, 2]

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'prostate')).is_dir():
    if (pathfile := pathdir.joinpath('importancia.pkl')).is_file():
        with open(pathfile, "rb") as f:
            dict_imps = pickle.load(f)
            save = False
else:
    pathdir.mkdir(parents=True)

if save:
    from vertibench.Splitter import ImportanceSplitter
    dict_imps = {alpha: {} for alpha in alpha_lst}

    for p in p_desbal:
        for alpha in alpha_lst:
            print(f"ImportanceSplitter para p={p}, alpha={alpha}...")
            imp_splitter = ImportanceSplitter(num_parties=p, seed=1, weights=alpha)
            alpha_idx = imp_splitter.split_indices(X_train)
            dict_imps[alpha][p] = {node: list(map(int, idx)) for node, idx in enumerate(alpha_idx)}

    with open(pathfile, "wb") as f:
        pickle.dump(dict_imps, f)

print(f"Split importancia realizado: {p_desbal} nodos con {alpha_lst} alfas")
# Debug dict_imps - reparto de columnas asociadas a alpha=0.1 y p=2
for node, cols in dict_imps[0.1][2].items():
    print(f"Node {node+1} - {list(Xtrain.columns[cols])}")

#%% Lanzamos experimentos IMPORTANCIA - BCD ROLANN
df_lst = []
for alfa in alpha_lst:
    print("Experimento importancia, alfa={alfa}. Preparando params BCD ROLANN...")
    param_grid['p'] = dict_imps[alfa]
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
    df = merge_df_gridsearch(exps, name=f'bcd_desbal_imp_{alfa}')
    df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_lst.append(df)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst)

#%% Lanzamos experimentos IMPORTANCIA - LR VFL
df_lst_lr = []
for alfa in alpha_lst:
    print("Experimento importancia, alfa={alfa}. Preparando params LRVFL...")
    vfl_grid['p'] = dict_imps[alfa]
    # Batería de pruebas
    exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='auto')   
    df_lr = merge_df_gridsearch(exps_lr, name=f'lr_desbal_imp_{alfa}')
    df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)
    df_lst_lr.append(df_lr)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst_lr)

#%% Concatenamos todos los dataframes y guardamos en un gigante DF para posterior análisis
df_credit = pd.concat(df_all_lst)
df_credit.info()
df_credit.to_excel("results/prostate/df.xlsx", index=False)

# %%
