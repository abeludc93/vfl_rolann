# %% [DATASET] - OJO NO HAY TARGETS PARA TEST
from sklearn.preprocessing import StandardScaler
from helpers import load_dataset
import numpy as np
import pandas as pd

# Carga
X_train_orig, y_train_orig  = load_dataset("datasets/clasificacion/VertiBench/Gisette_train.xlsx")
X_val_orig, y_val_orig      = load_dataset("datasets/clasificacion/VertiBench/Gisette_valid.xlsx")
X_test_orig, y_test_orig    = load_dataset("datasets/clasificacion/VertiBench/Gisette_test.xlsx")

labels = np.unique(y_train_orig)
print(labels)
print(f"(TRAIN) Reparto clases [1|0]: {y_train_orig.value_counts()}")
print(f"(VALID) Reparto clases [1|0]: {y_val_orig.value_counts()}")
print(f"(TEST)  Reparto clases [1|0]: {y_test_orig.value_counts()}")

#%% [PREPROCESADO]

# Normalización (datos entrada)
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(X_train_orig)

Xtrain = scaler.transform(X_train_orig)
Xtest  = scaler.transform(X_val_orig)   # Ojo uso VALID como TEST

X_train = Xtrain.to_numpy()
X_test  = Xtest.to_numpy()
y_train = y_train_orig.to_numpy()
y_test  = y_val_orig.to_numpy()

# Mapeado de clases a 0|1
y_train = np.where(y_train == -1, 0, 1)
y_test  = np.where(y_test == -1, 0, 1)

# %% [ROLANN HIPERPARAMS]
f_act = 'logs'
lam = 10
rolann_params = {'f_act':f_act, 'lam': lam}
lr_params = {'lam': lam}

rolann_grid = {k:[v] for k,v in rolann_params.items()}
lr_grid = {k:[v] for k,v in lr_params.items()}

# %% [CONDICIONAMIENTO]
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
""" ROLANN
Condicionamiento A escalada: 1934196.35
Condicionamiento A sin escalar: 1861352563923.43
	Reducción en razón del orden de: 9.62e+07
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
    'palette': 'bright',
}

#%% [TESTING ROLANN] (1m)
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models = fit_rolann_models(X_train, y_train, mode='binary', **rolann_params)

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train)
pred_test   = score2label(score_test)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 99.75%
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 97.80%

#%% [TESTING LR]
from helpers import fit_lr_models
from sklearn.metrics import accuracy_score

# Fit
models_lr = fit_lr_models(X_train, y_train, mode='auto', **lr_params|{'tol':0,'max_iter':100})

# Acc
pred_lr_train = predict_models(X_train, models_lr)
pred_lr_test  = predict_models(X_test, models_lr)

acc_lr_train   = accuracy_score(y_train, pred_lr_train)
acc_lr_test    = accuracy_score(y_test, pred_lr_test)

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   # 100%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     # 98.10%

#%% [PLOT]
w_rolann = models[0].w_
w_lr = models_lr[0].w_
plot_weights_comparison(w_rolann, w_lr, 'ROLANN', 'LR')

#%% [TESTING BCD]
from helpers import fit_bcd_models

bcd_params = {
                'p':        5000,
                'max_iter': 1000,
                'tol':      0,
                'mode_bcd': 'parallel',
                'lr':       0.01,
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train)
pred_bcd_test   = score2label(score_bcd_test)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 99.75%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 97.80%

#%% [PLOT]
plot_weights_comparison(w_rolann, coords[0].w_, 'ROLANN', 'BCD')

#%% [TESTING VFL LR]
from helpers import fit_lrvfl_models

vfl_params = {
    'p': 1000,
    'lr': 0.1,
    'method': 'gd',

    'max_iter': 100,
    'tol': 0,
    'taylor': False,
    'w_lr': models_lr[0].w_,
    'test_data': (X_test.T, y_test)
}

# Fit
coords_vfl, clients_vfl = fit_lrvfl_models(X_train, y_train, **(lr_params|vfl_params), mode='binary')

# Acc
score_vfl_train = predict_models(X_train.T, coords_vfl)
score_vfl_test  = predict_models(X_test.T, coords_vfl)

pred_vfl_train  = score2label(score_vfl_train)
pred_vfl_test   = score2label(score_vfl_test)

acc_vfl_train   = accuracy_score(y_train, pred_vfl_train)
acc_vfl_test    = accuracy_score(y_test, pred_vfl_test)

print(f"Accuracy train: {100 * acc_vfl_train:.2f} %")   # 100%
print(f"Accuracy test: {100 * acc_vfl_test:.2f} %")     # 98.10%

#%% [PLOT]
plot_weights_comparison(w_lr, coords_vfl[0].w_, 'LR', 'VFL')

#%% VertiBench: IMPORTANCIA
from pathlib import Path
import pickle
alpha_lst = [.1, 1, 10]
p_desbal = [2, 10]

pathdir = Path.cwd().joinpath('cache', 'gisette')
pathfile1 = pathdir.joinpath('importancia.pkl')

save = True
if pathdir.is_dir():
    if pathfile1.is_file():
        with open(pathfile1, "rb") as f:
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

    with open(pathfile1, "wb") as f:
        pickle.dump(dict_imps, f)
    print(f"Split importancia realizado: {p_desbal} nodos con {alpha_lst} alfas")

#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% [ESCENARIOS VFL]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch, compute_extra_results
df_all_lst = []

bcd_grid = \
{
    'p':        [],
    'mode_bcd': ['parallel', 'sequential'],
    'lr':       [0.01],
    'max_iter': [2000],
    'tol':      [0],
    'f_act':    ['logs'],
    'lam':      [10],
}

vfl_grid = \
{
    'p':        [],
    'lr':       [0.01],   # gd funciona para cualquier lr
    'method':   ['gd','bfgs'],
    'max_iter': [5000],
    'tol':      [0],
    'taylor':   [False],
    'lam':      [10],
}

#%% (BALANCEADO - BCD ROLANN)
p_bal = [2, 10, 100, 5000]
bcd_grid['p'] = p_bal
print(bcd_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='binary')

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_bal')

# Rresultados complementarios
df_bal = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

# Guardamos resultados BAL
df_all_lst.append(df_bal)

#%% (BALANCEADO - VFL LR)
p_bal = [2, 10, 100, 5000]
vfl_grid['p'] = p_bal
print(vfl_grid)

# Lanzamos experimento
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='auto')   

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_bal')

# Resultados complementarios
df_lr_bal = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)

# Guardamos resultados BAL
df_all_lst.append(df_lr_bal)

#%% (DESBALANCEADO IMP  - BCD ROLANN)
dfs_bcd_corr = []
for alfa in alpha_lst:
    print(f"Experimento importancia, alfa={alfa} | BCD ROLANN")
    bcd_grid['p'] = list(dict_imps[alfa].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='binary')
    df = merge_df_gridsearch(exps, name=f'bcd_desbal_imp_{alfa}')
    df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)
    dfs_bcd_corr.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_bcd_corr)

#%% (DESBALANCEADO IMP  - VFL LR)
dfs_bcd_imp = []
for alfa in alpha_lst:
    print(f"Experimento importancia, alfa={alfa} | VFL LR")
    vfl_grid['p'] = list(dict_imps[alfa].values())
    # Batería de pruebas
    exps = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='auto')
    df = merge_df_gridsearch(exps, name=f'lr_desbal_imp_{alfa}')
    df = compute_extra_results(df, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)
    dfs_bcd_imp.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_bcd_imp)

#%% (EXPORTAMOS)
df_gisette = pd.concat(df_all_lst)
df_gisette.info()
df_gisette.to_excel("results/gisette/df.xlsx", index=False)

#%%