# %% [DATASET] - division entrenamiento/test dada
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from helpers import load_dataset
import numpy as np
import pandas as pd

# Carga
X, y = load_dataset("datasets/regresion/FedOnce/Superconduct.xlsx")
print(f"Salida: min - {np.min(y.values)}, max - {np.max(y.values)}\n{y.describe()}")

# División datos
X_train_orig, X_test_orig, y_train_orig, y_test_orig = \
    train_test_split(X, y, test_size=0.3, random_state=1)

#%% Centrado de datos: restamos media (target) o aplicamos yeo-johnson si hay asimetría (desbalanceo)
if abs(skew := y.critical_temp.skew()) >= 1:
    print(f"Asimétria detectada ({skew:.2f}). Transformando target...")
    scaler_y = PowerTransformer('yeo-johnson', standardize=True).set_output(transform='pandas')
else:
    print(f"Distribución simétrica. Centrando target...")
    scaler_y = StandardScaler(with_std=False).set_output(transform='pandas')

scaler_y.fit(y_train_orig)
ytrain_scaled = scaler_y.transform(y_train_orig)
ytest_scaled = scaler_y.transform(y_test_orig)
min_yscaled = np.min(ytrain_scaled)
max_yscaled = np.max(ytrain_scaled)

#%% Visualización
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.figure(figsize=(8,4))
ax1 = plt.subplot(1, 2, 1)
y_train_orig.critical_temp.hist(ax=ax1)
ax2 = plt.subplot(1, 2, 2)
stats.probplot(ytrain_scaled.critical_temp, dist="norm", plot=ax2);
ax1.set_title(f'ytrain (skew={skew:.2f})');
ax2.set_title('ytrain_scaled (probplot)');

#%% Normalización input
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(X_train_orig)

Xtrain_scaled = scaler.transform(X_train_orig)
Xtest_scaled  = scaler.transform(X_test_orig)

# Numpy
X_train = Xtrain_scaled.to_numpy()
X_test  = Xtest_scaled.to_numpy()
y_train = ytrain_scaled.to_numpy()
y_test  = ytest_scaled.to_numpy()

# %% [HIPERPARAMS]
f_act = 'linear'
lam = 1

rolann_params = {'f_act':f_act, 'lam': lam}
ridge_params = {'lam': lam}

# %% [CONDICIONAMIENTO]
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
"""
Condicionamiento A escalada: 557202.09
Condicionamiento A sin escalar: 7658840280340.87
	Reducción en razón del orden de: 1.37e+09
"""
#%% [Test ROLANN]
from rolann import ROLANN
from sklearn.metrics import root_mean_squared_error

mdl = ROLANN(f_act=f_act, lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_

y_pred_train = mdl.predict(X_train)
rmse_scaled_train = root_mean_squared_error(y_train, y_pred_train)   # 17.95

y_pred = mdl.predict(X_test)
rmse_scaled = root_mean_squared_error(y_test, y_pred)   # 17.95

print(f"RMSE TRAIN: {rmse_scaled_train:.2f}")
print(f"RMSE TEST: {rmse_scaled:.2f}")

#%% [Test Ridge]
from sklearn.linear_model import Ridge

mdl_lr = Ridge(alpha=lam)
mdl_lr.fit(X_train, y_train)
w_ridge = np.hstack([mdl_lr.intercept_, mdl_lr.coef_]).reshape(-1, 1)
mdl_lr.w_ = w_ridge


y_pred_ridge_train = mdl_lr.predict(X_train)
rmse_scaled_ridge_train = root_mean_squared_error(y_train, y_pred_ridge_train)   # 17.95

y_pred_ridge = mdl_lr.predict(X_test)
rmse_scaled_ridge = root_mean_squared_error(y_test, y_pred_ridge)   # 17.95

print(f"RMSE TEST Ridge: {rmse_scaled_ridge_train:.2f}")
print(f"RMSE TEST Ridge: {rmse_scaled_ridge:.2f}")

#%% Guardamos resultados centralizados
import pickle
metric_dic = {}
metric_dic['rolann'] = mdl
metric_dic['lr'] = mdl_lr

metric_dic['rmse_train_rolann'] = rmse_scaled_train
metric_dic['rmse_test_rolann'] = rmse_scaled

metric_dic['rmse_train_lr'] = rmse_scaled_ridge_train
metric_dic['rmse_test_lr'] = rmse_scaled_ridge

with open("./results/superconduct/mdls.pkl", "wb") as f:
    pickle.dump(metric_dic, f)

#%% [Figura predict]

# Si hubo transformación devolvemos a escala original
if abs(skew) >= 1:
    y_pred = np.clip(y_pred, min_yscaled, max_yscaled)
    y_pred_ridge = np.clip(y_pred_ridge, min_yscaled, max_yscaled)

ypred = scaler_y.inverse_transform(y_pred)
ypred_ridge = scaler_y.inverse_transform(y_pred_ridge.reshape(-1,1))

rmse = root_mean_squared_error(y_test_orig.values, ypred)
rmse_ridge = root_mean_squared_error(y_test_orig.values, ypred_ridge)
print(f"")

# Ordenamos salidas
idx = np.argsort(y_test_orig.values.ravel())
y_real_sorted = y_test_orig.values.ravel()[idx]
y_pred_sorted = ypred.ravel()[idx]
y_pred_ridge_sorted = ypred_ridge.ravel()[idx]

# Plot
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(y_real_sorted, 'k', lw=2, label='Target')
plt.plot(y_pred_sorted, 'r', alpha=0.7, label='Pred')
plt.xlabel("Salidas reales en orden creciente")
plt.ylabel("Critical Temperature")
plt.title("ROLANN")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1,2,2)
plt.plot(y_real_sorted, 'k', lw=2, label='Target')
plt.plot(y_pred_ridge_sorted, 'g', alpha=0.7, label='Pred')
plt.xlabel("Salidas reales en orden creciente")
plt.ylabel("Critical Temperature")
plt.title("Ridge")

plt.legend()
plt.grid(alpha=0.3)

#%% [PLOT]
from plotters import plot_weights_comparison
plot_weights_comparison(w_rolann, w_ridge, 'ROLANN', 'RIDGE')

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% [PLOT_SNS]
from plotters import plot_sns
sns_params = {
    'x': 'iter',        # Eje X
    'y': 'diff_Z',      # Eje Y
    'hue': 'lr',        # Distinto color para cada Learning Rate
    'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
    'size': 'p',        # Distinto tamaño
    'linewidth': 1.5,   # etc
    'palette': 'bright'
}

#%% [TESTING BCD ROLANN]
from helpers import fit_bcd_models, predict_models
from sklearn.metrics import root_mean_squared_error

bcd_params = {
                'p':        81,
                'max_iter': 2500,
                'tol':      0,
                'mode_bcd': 'sequential',
                'lr':       0.01,
                'w_rolann': [w_rolann],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='regression')

# Acc
pred_bcd_train = predict_models(X_train.T, coords)
pred_bcd_test  = predict_models(X_test.T, coords)

rmse_bcd_train = root_mean_squared_error(y_train, pred_bcd_train)
rmse_bcd_test  = root_mean_squared_error(y_test, pred_bcd_test)

print(f"RMSE train (BCD): {rmse_bcd_train:.2f}")   # 17.42
print(f"RMSE test (BCD): {rmse_bcd_test:.2f}")     # 17.95

#%% [Figuras BCD]
df = coords[0].get_results()
plot_weights_comparison(w_rolann, coords[0].w_, 'ROLANN', 'BCD')
plot_sns(df, plot_log=False, **sns_params|{'y': 'score_train'})

#%% [TESTING VFL RIDGE]
from helpers import fit_lrvfl_models

vfl_params = {
    'p': 81,            
    'max_iter': 1000,
    'tol': 0,
    'lr': 0.001,
    'method': 'bfgs',
    'taylor': False,
    'w_lr': w_ridge,
    'test_data': (X_test.T, y_test)
}

# Fit
coords_ridge, clients_ridge = fit_lrvfl_models(X_train, y_train, **(ridge_params|vfl_params), mode='regression')

# Acc
pred_ridge_train = predict_models(X_train.T, coords_ridge)
pred_ridge_test  = predict_models(X_test.T, coords_ridge)

rmse_ridge_train = root_mean_squared_error(y_train, pred_ridge_train)
rmse_ridge_test  = root_mean_squared_error(y_test, pred_ridge_test)

print(f"RMSE train (RIDGE): {rmse_ridge_train:.2f}")   # 17.54
print(f"RMSE test (RIDGE): {rmse_ridge_test:.2f}")     # 18.04

#%% [Figuras Ridge]
df = coords_ridge[0].get_results()
plot_weights_comparison(w_ridge, coords_ridge[0].w_, 'RIDGE', 'VFL')
plot_sns(df, plot_log=False, **sns_params|{'y': 'score_train'})

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% INVESTIGANDO LOSSES
def compute_loss(X, y, w, lam, reg_bias=False):
    reg_bias = False
    Xp = np.vstack([np.ones(len(X)), X.T])
    w = w.reshape(-1, 1)
    z = w.T @ Xp
    y_f, z_f = y.flatten(), z.flatten()
    loss     = np.mean((y_f - z_f)**2)
    loss_l2  = np.sum(w**2) if reg_bias else np.sum(w[1:]**2)
    sum_loss =  0.5 * (loss + lam * loss_l2)

    return loss, loss_l2, sum_loss

#%% LOSS RIDGE
loss_ridge, loss_l2_ridge, sum_loss = compute_loss(X_train, y_train, w_ridge, lam)
print(loss_ridge, loss_l2_ridge, sum_loss)

#%% LOSS VFL
df = coords_ridge[0].get_results()
loss_lst, loss_l2_lst, sum_loss_lst = [], [], []
for w_i in df.w_i.values:
    loss_iter, loss_l2_iter, sum_loss_iter = compute_loss(X_train, y_train, w_i, lam)
    loss_lst.append(loss_iter)
    loss_l2_lst.append(loss_l2_iter)
    sum_loss_lst.append(sum_loss_iter)

#%% FIGURA
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(12,4))
axes[0].plot(loss_lst)
axes[1].plot(loss_l2_lst)
axes[2].plot(sum_loss_lst)
axes[0].set_title("loss")
axes[1].set_title("loss_l2")
axes[2].set_title("sum_loss")
fig.suptitle("")

#%% LOSS ROLANN
loss_rolann, loss_l2_rolann, sum_loss_rolann = compute_loss(X_train, y_train, w_rolann, lam)
print(loss_rolann, loss_l2_rolann, sum_loss_rolann)

#%% LOSS VFL
df_rolann = coords[0].get_results()
loss_lst, loss_l2_lst, sum_loss_lst = [], [], []
for w_i in df.w_i.values:
    loss_iter, loss_l2_iter, sum_loss_iter = compute_loss(X_train, y_train, w_i, lam)
    loss_lst.append(loss_iter)
    loss_l2_lst.append(loss_l2_iter)
    sum_loss_lst.append(sum_loss_iter)

#%% VertiBench: CORRELACION e IMPORTANCIA
from pathlib import Path
import pickle
beta_lst = [0, .5, 1]
alpha_lst = [.1, 1, 10]
p_desbal = [2, 10]

pathdir = Path.cwd().joinpath('cache', 'superconduct')
pathfile1 = pathdir.joinpath('importancia.pkl')
pathfile2 = pathdir.joinpath('correlacion.pkl')

save = True
if pathdir.is_dir():
    if pathfile1.is_file():
        with open(pathfile1, "rb") as f:
            dict_imps = pickle.load(f)
            save = False
    if pathfile2.is_file():
        with open(pathfile2, "rb") as f:
            dict_corrs = pickle.load(f)
            save = False
else:
    pathdir.mkdir(parents=True)

if save:
    from vertibench.Splitter import ImportanceSplitter
    from vertibench.Splitter import CorrelationSplitter
    dict_imps = {alpha: {} for alpha in alpha_lst}
    dict_corrs = {beta: {} for beta in beta_lst}

    for p in p_desbal:
        for alpha in alpha_lst:
            print(f"ImportanceSplitter para p={p}, alpha={alpha}...")
            imp_splitter = ImportanceSplitter(num_parties=p, seed=1, weights=alpha)
            alpha_idx = imp_splitter.split_indices(X_train)
            dict_imps[alpha][p] = {node: list(map(int, idx)) for node, idx in enumerate(alpha_idx)}

    with open(pathfile1, "wb") as f:
        pickle.dump(dict_imps, f)
    print(f"Split importancia realizado: {p_desbal} nodos con {alpha_lst} alfas")

    for p in p_desbal:
        print(f"CorrelationSplitter para p={p} ...")
        corr_splitter = CorrelationSplitter(num_parties=p, seed=1, gpu_id=0, n_jobs=1)
        corr_splitter.fit(X_train)
        for beta in beta_lst:
            print(f"\tObteniendo indices con beta={beta} ...")
            beta_idx = corr_splitter.split_indices(X_train, beta=beta)
            dict_corrs[beta][p] = {node: list(map(int, idx)) for node, idx in enumerate(beta_idx)}

    with open(pathfile2, "wb") as f:
        pickle.dump(dict_corrs, f)

    print(f"Split correlación realizado: {p_desbal} nodos con {beta_lst} betas")

# %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% [ESCENARIOS VFL]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch, compute_extra_results
df_all_lst = []

bcd_grid = \
{
    'p':        [],
    'mode_bcd': ['parallel', 'sequential'],
    'lr':       [0.01],
    'max_iter': [5000],
    'tol':      [0],
    'f_act':    ['linear'],
    'lam':      [1],
}

vfl_grid = \
{
    'p':        [],
    'lr':       [0.001],
    'method':   ['gd','bfgs'],
    'max_iter': [10000],
    'tol':      [0],
    'taylor':   [False],
    'lam':      [1],
}

#%% (BALANCEADO - BCD ROLANN)
p_bal = [2, 10, 40, 81]
bcd_grid['p'] = p_bal
print(bcd_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='regression')

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_bal')

# Rresultados complementarios
df_bal = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

# Guardamos resultados BAL
df_all_lst.append(df_bal)

#%% (BALANCEADO - VFL LR)
p_bal = [2, 10, 40, 81]
vfl_grid['p'] = p_bal
print(vfl_grid)

# Lanzamos experimento
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='regression')   

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_bal')

# Resultados complementarios
df_lr_bal = compute_extra_results(df_lr, w_optim=w_ridge, rolann=False, X=X_train, y=y_train, lam=lam, f_act='linear')

# Guardamos resultados BAL
df_all_lst.append(df_lr_bal)

#%% (DESBALANCEADO CORR - BCD ROLANN)
dfs_bcd_corr = []
for beta in beta_lst:
    print(f"Experimento correlación, beta={beta} | BCD ROLANN")
    bcd_grid['p'] = list(dict_corrs[beta].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='regression')
    df = merge_df_gridsearch(exps, name=f'bcd_desbal_corr_{beta}')
    df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)
    dfs_bcd_corr.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_bcd_corr)

#%% (DESBALANCEADO IMP  - BCD ROLANN)
dfs_bcd_imp = []
for alfa in alpha_lst:
    print(f"Experimento importancia, alfa={alfa} | BCD ROLANN")
    bcd_grid['p'] = list(dict_imps[alfa].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, bcd_grid, mode='regression')
    df = merge_df_gridsearch(exps, name=f'bcd_desbal_imp_{alfa}')
    df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)
    dfs_bcd_imp.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_bcd_imp)

#%% (DESBALANCEADO CORR - VFL LR)
dfs_lr_corr = []
for beta in beta_lst:
    print(f"Experimento correlación, beta={beta} | VFL LR")
    vfl_grid['p'] = list(dict_corrs[beta].values())
    # Batería de pruebas
    exps = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='regression')
    df = merge_df_gridsearch(exps, name=f'lr_desbal_corr_{beta}')
    df = compute_extra_results(df, w_optim=w_ridge, rolann=False, X=X_train, y=y_train, lam=lam, f_act='linear')
    dfs_lr_corr.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_lr_corr)

#%% (DESBALANCEADO IMP  - VFL LR)
dfs_lr_imp = []
for alfa in alpha_lst:
    print(f"Experimento importancia, alfa={alfa} | VFL LR")
    vfl_grid['p'] = list(dict_imps[alfa].values())
    # Batería de pruebas
    exps = run_grid_search_lr(X_train, y_train, X_test, y_test, vfl_grid, mode='regression')
    df = merge_df_gridsearch(exps, name=f'lr_desbal_imp_{alfa}')
    df = compute_extra_results(df, w_optim=w_ridge, rolann=False, X=X_train, y=y_train, lam=lam, f_act='linear')
    dfs_lr_imp.append(df)

# Guardamos resultados DESBAL
df_all_lst.extend(dfs_lr_imp)

#%% (EXPORTAMOS)
df_superconduct = pd.concat(df_all_lst)
df_superconduct.info()
df_superconduct.to_excel("results/superconduct/df.xlsx", index=False)

#%%