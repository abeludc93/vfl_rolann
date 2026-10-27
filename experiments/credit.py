#!/usr/bin/env python3
"""
experiments/credit.py
~~~~~~~~~~~~~~~~~~~~~~~~
Experiment with the UCI ML - Default Credit Dataset.
+ Samples:  30000
+ Features: 23
+ Task:     binary classification
"""
#%% [0. SYSPATH]
# To avoid using relative imports (requires setting up modules)
# we update PYTHONPATH to include the root directory (cwd parent)
import sys
from pathlib import Path
SEED = 1

if __file__ is None:
    p = Path.cwd().parent
else:
    p = Path(__file__).parents[1]
sys.path.insert(0, str(p))

# %% [1. DATASET] - train/val/test split and class balance
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers import balance_train_data
import numpy as np
import pandas as pd

# Carga
default_of_credit_card_clients = fetch_ucirepo(id=350) 
X = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 
labels = np.unique(y)
print(f"Reparto clases [1|0]: {y.value_counts()}")

# Preprocesado
map_columns = {'X1': 'LIMIT_BAL',
               'X2': 'SEX',
               'X3': 'EDUCATION',
               'X4': 'MARRIAGE',
               'X5': 'AGE',
    
               'X6': 'PAY_0',
               'X7': 'PAY_2',
               'X8': 'PAY_3',
               'X9': 'PAY_4',
               'X10': 'PAY_5',
               'X11': 'PAY_6',

               'X12': 'BILL_AMT1',
               'X13': 'BILL_AMT2',
               'X14': 'BILL_AMT3',
               'X15': 'BILL_AMT4',
               'X16': 'BILL_AMT5',
               'X17': 'BILL_AMT6',
							
               'X18': 'PAY_AMT1',
               'X19': 'PAY_AMT2',
               'X20': 'PAY_AMT3',
               'X21': 'PAY_AMT4',
               'X22': 'PAY_AMT5',
               'X23': 'PAY_AMT6',
               }

X = X.rename(columns=map_columns)

num_demo_cols = ['LIMIT_BAL', 'AGE']
cat_demo_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
pay_cols = [ # 18 variables
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'] 

X['EDUCATION'] = X['EDUCATION'].replace({0: 4, 5: 4, 6: 4}) # Agrupamos 0, 5 y 6 en la categoría 4 (others)
X['MARRIAGE'] = X['MARRIAGE'].replace({0: 3})               # Agrupamos el 0 en la categoría 3 (others)

X_cat_demo = pd.get_dummies(X[cat_demo_cols], columns=cat_demo_cols, dtype=int)
X_final = pd.concat([X[pay_cols + num_demo_cols], X_cat_demo], axis=1)

# División datos: 70-30
Xtrain, Xvaltest, ytrain, yvaltest = \
    train_test_split(X_final, y, test_size=0.3, stratify=y, random_state=SEED)

Xval, Xtest, yval, ytest = \
    train_test_split(Xvaltest, yvaltest, test_size=0.5, stratify=yvaltest, random_state=SEED)

# Centrado de datos (etiquetas)
Xtrainbal, ytrainbal = balance_train_data(Xtrain, ytrain)

# Normalización (datos entrada)
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(Xtrainbal)

Xtrainbal   = scaler.transform(Xtrainbal)
Xval        = scaler.transform(Xval)
Xtest       = scaler.transform(Xtest)

X_train = Xtrainbal.to_numpy()
X_val   = Xval.to_numpy()
X_test  = Xtest.to_numpy()
y_train = ytrainbal.to_numpy()
y_val   = yval.to_numpy()
y_test  = ytest.to_numpy()

# %% [ROLANN HIPERPARAMS]
f_act = 'logs'
lam = 1e-3
rolann_params = {'f_act':f_act, 'lam': lam}
rolann_grid = {k:[v] for k,v in rolann_params.items()}

# %% [CONDICIONAMIENTO]
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
""" ROLANN
Condicionamiento A escalada: 134426.48
Condicionamiento A sin escalar: 3119082708406449.50
	Reducción en razón del orden de: 2.32e+12
"""
#%% [COMENTARIOS]
""" LR
Para condicionamiento de (X^T @ X): 3.76e+11
     condicionamiento de (X^T @ D @ X): 3.76e+11        (con D=diag(0.25...))

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

$\lam_{max}(Hess_{LR}) \approx 494512247954619.25$

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

print(f"Accuracy train: {100 * acc_train:.2f} %")   #  67.23%
print(f"Accuracy test: {100 * acc_test:.2f} %")     #  69.04%

#%% BCD
from helpers import fit_bcd_models

bcd_params = {
                'p':        23,
                'max_iter': 1000,
                'tol':      0,
                'mode_bcd': 'parallel',
                'lr':       0.2,
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

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 67.23%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 69.04%

#%% [TESTING LR & VFL LR]
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

lr_params = {
    'lam': 1e-3,    # esto es C=1/lam... lam=2e4 CORREGIDO
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

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   #  67.09%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     #  68.42%


#%% VFL LR
from helpers import fit_lrvfl_models

vfl_params = {
    'p': 23,             # probado con otros 'p' -> min lr=0.001
    'lr': 0.2,
    'max_iter': 5000,   # override lr_params (meter > )
    'method': 'bfgs',
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

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 67.09%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 68.42%

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% [ESCENARIOS VFL]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch, compute_extra_results
df_all_lst = []

#%% (1) - Split natural en 2 y 4 nodos - BCD ROLANN
feat_pay  = len(pay_cols)
feat_demo = len(num_demo_cols) + len(X_cat_demo.columns)
split2 = [feat_pay, feat_demo]
split4 = [6, 6, 6, 11]
# BCD params
param_grid  = {
    'p': [split2, split4],                          # Reparto: [[vars pago, vars demograficas]]
    'mode_bcd': ['parallel', 'sequential'],         # Modos
    'lr': [.2],                                     # Learning rates modo paralelo
    'tol': [0],                                     # Tolerancia
    'max_iter': [5000],                             # Max iteraciones
    }
print(param_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')

# (exps) es una lista de tuplas (mdls_rolann, mdls_bcd); a su vez, mdls_* son listas de n_outputs elementos (1 por clase/salida)
w_rolann = exps[0][0][0].w_

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_natural')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

#%% (1) - Split natural en 2 y 4 nodos - LR VFL
lr_grid = {
    'lam': [lam],                   # Regularización L2
    'tol': [0],
    'max_iter': [5000]
    }
vfl_grid = {
    'p': [split2, split4],
    'lr': [.2],
    'method': ['gd', 'bfgs'],
    'taylor': [False]
    }

# Lanzamos experimento - auto=binary | ovr=multi_class
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='auto')   

# Sol. optima LR
w_lr = exps_lr[0][0][0].w_

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_natural')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)

# Guardamos en lista de resultados ambos experimentos
df_all_lst.extend([df, df_lr])

#%% (2) - Split balanceado - BCD ROLANN
m, m_min = X_train.shape[1], 4
num_groups = 4
p_bal = np.geomspace(m, m_min, num_groups, dtype=int).tolist()
param_grid['p'] = p_bal # 29 atributos, 4 minimo, 5 grupos para geomspace => [29, 14, 7, 4]
print(param_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_bal')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_bal = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

#%% (2) - Split balanceado - LR VFL
vfl_grid['p'] = p_bal

# Lanzamos experimento
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='auto')   

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_bal')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_lr_bal = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)

# Guardamos en lista de resultados ambos experimentos
df_all_lst.extend([df_bal, df_lr_bal])

#%% (3) - Split desbalanceado - CORRELACION -
# Generar estas particiones es una tarea computacionalmente intensiva, por lo
# que cacheamos los índices de las divisiones realizadas para futuras ocasiones
beta_lst = [0, .5, 1]
p_desbal = p_bal[1:] + [2]    # quitamos 1era opcion p_bal=29 que corresponde a 1 variable/nodo

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'credit')).is_dir():
    if (pathfile := pathdir.joinpath('correlacion.pkl')).is_file():
        with open(pathfile, "rb") as f:
            dict_corrs = pickle.load(f)
            save = False
else:
    pathdir.mkdir(parents=True)

if save:
    from vertibench.Splitter import CorrelationSplitter
    dict_corrs = {beta: {} for beta in beta_lst}

    for p in p_desbal:
        print(f"CorrelationSplitter para p={p} ...")
        corr_splitter = CorrelationSplitter(num_parties=p, seed=1, gpu_id=0, n_jobs=1)
        corr_splitter.fit(X_train)
        for beta in beta_lst:
            print(f"\tObteniendo indices con beta={beta} ...")
            beta_idx = corr_splitter.split_indices(X_train, beta=beta)
            dict_corrs[beta][p] = {node: list(map(int, idx)) for node, idx in enumerate(beta_idx)}

    with open(pathfile, "wb") as f:
        pickle.dump(dict_corrs, f)

print(f"Split correlación realizado: {p_desbal} nodos con {beta_lst} betas")
# Debug dict_imps - reparto de columnas asociadas a beta=0 y p=4
for node, cols in dict_corrs[0][4].items():
    print(f"Node {node+1} - {list(Xtrain.columns[cols])}")

#%% Lanzamos experimentos CORRELACION - BCD ROLANN
df_lst = []
for beta in beta_lst:
    print(f"Experimento correlación, beta={beta}. Preparando params BCD ROLANN...")
    param_grid['p'] = list(dict_corrs[beta].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='binary')
    df = merge_df_gridsearch(exps, name=f'bcd_desbal_corr_{beta}')
    df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)
    df_lst.append(df)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst)

#%% Lanzamos experimentos CORRELACION - LR VFL
df_lst_lr = []
for beta in beta_lst:
    print(f"Experimento correlación, beta={beta}. Preparando params LRVFL...")
    vfl_grid['p'] = list(dict_corrs[beta].values())
    # Batería de pruebas
    exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='auto')   
    df_lr = merge_df_gridsearch(exps_lr, name=f'lr_desbal_corr_{beta}')
    df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)
    df_lst_lr.append(df_lr)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst_lr)

#%% (4) - Split desbalanceado - IMPORTANCIA -
# Aunque el split por importancia es mas eficiente computacionalmente, igualmente
# cacheamos los índices de las divisiones realizadas para futuras ocasiones
alpha_lst = [.1, 1, 10]
p_desbal = p_bal[1:] + [2]    # quitamos 1era opcion p_bal=29 que corresponde a 1 variable/nodo

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'credit')).is_dir():
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
# Debug dict_imps - reparto de columnas asociadas a alpha=0.1 y p=4
for node, cols in dict_imps[0.1][4].items():
    print(f"Node {node+1} - {list(Xtrain.columns[cols])}")

#%% Lanzamos experimentos IMPORTANCIA - BCD ROLANN
df_lst = []
for alfa in alpha_lst:
    print(f"Experimento importancia, alfa={alfa}. Preparando params BCD ROLANN...")
    param_grid['p'] = list(dict_imps[alfa].values())
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
    print(f"Experimento importancia, alfa={alfa}. Preparando params LRVFL...")
    vfl_grid['p'] = list(dict_imps[alfa].values())
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
df_credit.to_excel("results/credit/df_short2.xlsx", index=False)

# %%