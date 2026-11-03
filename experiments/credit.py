"""
#!/usr/bin/env python3
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

# División datos: 70-30 train-(val,test)
Xtrain, Xvaltest, ytrain, yvaltest = \
    train_test_split(X_final, y, test_size=0.3, stratify=y, random_state=SEED)

# División datos: 50-50 val-test
Xval, Xtest, yval, ytest = \
    train_test_split(Xvaltest, yvaltest, test_size=0.5, stratify=yvaltest, random_state=SEED)

# Centrado de datos (etiquetas)
Xtrainbal, ytrainbal = balance_train_data(Xtrain, ytrain)

# Normalización (datos entrada)
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(Xtrain)

Xtrain      = scaler.transform(Xtrain)
Xtrainbal   = scaler.transform(Xtrainbal)
Xval        = scaler.transform(Xval)
Xtest       = scaler.transform(Xtest)

# Numpy
X_train     = Xtrain.to_numpy()
X_train_bal = Xtrainbal.to_numpy()
X_val       = Xval.to_numpy()
X_test      = Xtest.to_numpy()
y_train     = ytrain.to_numpy()
y_train_bal = ytrainbal.to_numpy()
y_val       = yval.to_numpy()
y_test      = ytest.to_numpy()

# %% [ROLANN HIPERPARAMS]
from sklearn.model_selection import ParameterGrid
f_act = ['tanhip','logs']
lam = [1e-3, 1e-2, 1e-1, 1]
rolann_params = {'f_act':f_act, 'lam': lam}
rolann_grid = list(ParameterGrid(rolann_params))

# %% [CONDICIONAMIENTO]
from helpers import check_condition
for params in rolann_grid:
    k_A = check_condition(X_train, **params)
    k_A_bal = check_condition(X_train_bal, **params)
    print(f"Cond: {k_A:10.2f} Cond_bal: {k_A_bal:10.2f} | {params}")
# ↑ reg = ↓ cond , bal = ↓ cond

#%% [TESTING ROLANN]
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# (He probado a usar X_train_bal pero los resultados son peores)

# Fit
print(f"ROLANN test params: {rolann_grid[-1]}")
models = fit_rolann_models(X_train, y_train, mode='binary', **rolann_grid[-1])

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train, labels)
pred_test   = score2label(score_test, labels)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   #  79.92%
print(f"Accuracy test: {100 * acc_test:.2f} %")     #  80.09%
w_rolann = models[0].w_

#%% [TESTING BCDROLANN]
from helpers import fit_bcd_models

bcd_params = {
                'p':        2,
                'max_iter': 100,
                'tol':      0,
                'mode_bcd': 'sequential',
                'lr':       1,
}
test_data = {
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
print(f"BCDROLANN test params: {rolann_grid[-1]|bcd_params}")
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_grid[-1]|bcd_params|test_data), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 79.92%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 89.09%
df_bcd = coords[0].get_results()

#%% [TESTING LR]
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

lr_params = {
    'lam': 1,
    'tol': 0,
    'max_iter': 100
}

# Fit
print(f"LR test params: {lr_params}")
models_lr = fit_lr_models(X_train, y_train.ravel(), mode='auto', **lr_params)

# Acc
predict_models_lr = lambda X, mdls: np.hstack([mdl.predict_proba(X)[:, 1] for mdl in mdls])
score_lr_train = predict_models_lr(X_train, models_lr)
score_lr_test  = predict_models_lr(X_test, models_lr)
pred_train  = score2label(score_lr_train, labels)
pred_test   = score2label(score_lr_test, labels)

acc_lr_train   = accuracy_score(y_train, pred_train)
acc_lr_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   #  81.16%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     #  81.20%
w_lr = models[0].w_

#%% [TESTING LRVFL]
from helpers import fit_lrvfl_models

vfl_params = {
    'p': 2,             # probado con otros 'p' -> min lr=0.001
    'lr': 0.2,
    'max_iter': 100,   # override lr_params (meter > )
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

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 67.09%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 68.42%
df_lrvfl = coords[0].get_results()

#%% [EXTRA RESULTS]
# Columnas extras de resultados con optimalidad, costes comm, tiempo y loss.
from helpers import compute_extra_results, decision_function
from sklearn.metrics import brier_score_loss, roc_auc_score

train_params = {'X': X_train, 'y': y_train}
compute_extra_results(df_bcd, rolann=True, w_optim=w_rolann, 
                      **train_params|rolann_grid[-1])
compute_extra_results(df_lrvfl, rolann=False, w_optim=w_lr,
                      **train_params|{'lam':lr_params['lam'], 'f_act': df_lrvfl.f_act.iloc[0]}) 

df_bcd['brier_train']   = [brier_score_loss(y_train, decision_function(X_train, w, df_bcd.f_act.iloc[0]))
                           for w in df_bcd.w_i]
df_lrvfl['brier_train'] = [brier_score_loss(y_train, decision_function(X_train, w, df_lrvfl.f_act.iloc[0]))
                           for w in df_lrvfl.w_i]
df_bcd['auc_train']   = [roc_auc_score(y_train, decision_function(X_train, w, df_bcd.f_act.iloc[0]))
                           for w in df_bcd.w_i]
df_lrvfl['auc_train'] = [roc_auc_score(y_train, decision_function(X_train, w, df_lrvfl.f_act.iloc[0]))
                           for w in df_lrvfl.w_i]

#%% [PLOTTERS]
from plotters import plot_sns, plot_weights_comparison
sns_params = {
    'x': 'iter',        # Eje X
    'y': 'diff_Z',      # Eje Y
    'hue': 'lr',        # Distinto color para cada Learning Rate
    'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
    'size': 'p',        # Distinto tamaño
    'linewidth': 1.5,   # etc
    'palette': 'bright',
}
plot_sns(df_bcd, **sns_params|{'y':'opt_gap'})
plot_sns(df_lrvfl, **sns_params|{'y':'opt_gap'})
plot_sns(df_bcd, plot_log=False, **sns_params|{'y': 'brier_train'})
plot_sns(df_lrvfl, plot_log=False, **sns_params|{'y': 'brier_train'})
plot_sns(df_bcd, plot_log=False, **sns_params|{'y': 'auc_train'})
plot_sns(df_lrvfl, plot_log=False, **sns_params|{'y': 'auc_train'})

#%% [TEST FEDTREE]
from fedtree import FLClassifier

# VFL params
ft_vfl_params = {
    'n_parties': 2,                     # num nodos
    'partition': 1,                     # 0= cada nodo su Xi, 1=FedTree reparte equitativamente variables
    'mode': "vertical",                 # horizontal|vertical
    'partition_mode': "vertical",       # idem
    'propose_split': "server",          # split_points, dejar en 'server' (default)
    'merge_histogram': "server",        # idem
    'privacy_tech':"none",              # SMPC: 'none'|'he'|'dp' (en caso de 'dp', meter privacy_budget)
    'privacy_budget': 10,               # Para differential privacy
    'seed': SEED,
    #n_hori=2,                          # ?
    #n_verti=2,                         
    #alpha=100,                         
    #variance=200,                      
}
# GBDT params
ft_gbdt_params = {
    'max_depth': 6,                     # profundidad árboles
    'n_trees': 40,                      # num árboles
    'lambda_ft':1,                      # L2 reg
    'learning_rate':1,                  # learning rate
    'objective':"binary:logistic",      # f.obj: 'reg:linear'|'reg:logistic'|'binary:logistic'
    'verbose': 1,                       # 0|1|2               
    #gamma=1,                           # ?
    #min_child_weight=1,                
    #column_sampling_rate=1,
    #n_parallel_trees=1,
    #num_class=1,
    #n_device=1,
    #max_num_bin=255,
    #ins_bagging_fraction=1.0,
    #reorder_label=0,
    #bagging=0,
    #constant_h = 0.0,
    #tree_method="auto"
}
model = FLClassifier(**ft_gbdt_params|ft_vfl_params)

# Entrenamos y capturamos salida
from wurlitzer import pipes
with pipes() as (out, _):
  model.fit(X_train, y_train)
log_ft = out.read()
print(log_ft)

import re
auc_scores_ft = [float(val) for val in re.findall(r"AUC = ([0-9.]+)", log_ft)]
total_time_ft = [float(val) for val in re.findall(r"training time = ([0-9.]+)s", log_ft)][0]
iters_ft = np.arange(start=6, stop=246, step=6) # D, D*T, D

"""
# Si se ejecuta como un script de Python (no como notebook), debería de bastar esto
from contextlib import redirect_stdout
from io import StringIO
f = StringIO()
with redirect_stdout(f):
    model.fit(X_train, y_train)
log_ft = f.getvalue()
print(f"LOG: {log_ft}")
"""
score_ft_train = model.predict_proba(X_train)[:,1]  # clase positiva
score_ft_test  = model.predict_proba(X_test)[:,1]   # clase positiva

acc_ft_train = accuracy_score(y_train, (score_ft_train > .5).astype(int))
auc_ft_train = roc_auc_score(y_train, score_ft_train) 
acc_ft_test = accuracy_score(y_test, (score_ft_test > .5).astype(int))
auc_ft_test = roc_auc_score(y_test, score_ft_test) 

print(f"Accuracy train: {100 * acc_ft_train:.2f} %")   # 89.95 %
print(f"Accuracy test: {100 * acc_ft_test:.2f} %")     # 79.11 %

print(f"AUC train: {100 * auc_ft_train:.2f} %")   # 94.36 %
print(f"AUC test: {100 * auc_ft_test:.2f} %")     # 73.90 %

#%% [TEST SPLITNN]
from splitnn import SplitNNClassifier

nn_vfl_params = {
    'n_parties':            2,          # int
    'hidden_dims_clients':  (4,2),      # tuple
    'hidden_dims_server':   (6,),       # tuple
    'activation':           'relu',     # 'relu'|'tanh'|'gelu'

    'epochs':               25,         # int
    'batch_size':           64,         # int
    
    'validation_fraction':  0.0,        # float
    'restore_best_weights': True,       # bool
    'early_stopping':       False,      # bool
    'patience':             10,         # int   (if ↑ True)
    'min_delta':            0.0,        # float (if ↑ True)
}

nn_opt_params = {
    'learning_rate':        0.05,       # float
    'optimizer':            'sgd',      # 'sgd'|'adam'|'adamw'
    'momentum':             0.0,        # float
    'dropout':              0.0,        # float (0<=dropout<1)
    'weight_decay':         0.0,        # float
}

nn_misc_params = {
    'shuffle': False,
    'device': 'cpu',
    'verbose': True,
    'random_state': SEED,
}

model = SplitNNClassifier(**nn_vfl_params|nn_opt_params|nn_misc_params)

#%% 
import time
start = time.perf_counter()
model.fit(X_train, y_train)
elapsed = time.perf_counter() - start

#%%
# Modelo info: batch_history_, epoch_history_, loss_history_

score_nn_train = model.predict_proba(X_train)[:,1]
score_nn_test = model.predict_proba(X_test)[:,1]

acc_nn_train = accuracy_score(y_train, model.predict(X_train))
acc_nn_test = accuracy_score(y_test, model.predict(X_test))

auc_nn_train = roc_auc_score(y_train, score_nn_train)
auc_nn_test = roc_auc_score(y_test, score_nn_test)

print(f"Accuracy train: {100 * acc_nn_train:.2f} %")   # 89.95 %
print(f"Accuracy test: {100 * acc_nn_test:.2f} %")     # 79.11 %

print(f"AUC train: {100 * auc_nn_train:.2f} %")   # 94.36 %
print(f"AUC test: {100 * auc_nn_test:.2f} %")     # 73.90 %

#%%
model.batch_history_
model.epoch_history_
model.loss_history_


#%% ======================================================================
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