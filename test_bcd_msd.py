# %% [DATASET] - division entrenamiento/test dada
from sklearn.preprocessing import StandardScaler, PowerTransformer
from helpers import load_dataset
import numpy as np
import pandas as pd

# Carga
Xtrain, ytrain = load_dataset("datasets/regresion/VertiBench/MSD_train.xlsx")
Xtest, ytest = load_dataset("datasets/regresion/VertiBench/MSD_test.xlsx")
target = (np.concat([ytrain, ytest]))
print(f"Salida: min - {target.min()}, max - {target.max()}")

# Preprocesado: las 12 primeras variables son promedios del timbre y las 78 siguientes covarianzas
avg_cols = list(range(12))
cov_cols = list(range(12, 90))

avg_cols_map = {f'Feat {i+1}': f'Avg {i+1}' for i in avg_cols}
cov_cols_map = {f'Feat {i+1}': f'Cov {i+1}' for i in cov_cols}
map_columns  = avg_cols_map|cov_cols_map

Xtrain = Xtrain.rename(columns=map_columns)
Xtest = Xtest.rename(columns=map_columns)

#%% Centrado de datos: restamos media (target) o aplicamos yeo-johnson si hay asimetría (desbalanceo)
if abs(skew := ytrain.year.skew()) >= 1:
    print(f"Asimétria detectada ({skew:.2f}). Transformando target...")
    scaler_y = PowerTransformer('yeo-johnson', standardize=True).set_output(transform='pandas')
else:
    print(f"Distribución simétrica. Centrando target...")
    scaler_y = StandardScaler(with_std=False).set_output(transform='pandas')

scaler_y.fit(ytrain)
ytrain_scaled = scaler_y.transform(ytrain)
ytest_scaled = scaler_y.transform(ytest)
min_yscaled = np.min(ytrain_scaled)
max_yscaled = np.max(ytrain_scaled)

#%% Visualización
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.figure(figsize=(8,4))
ax1 = plt.subplot(1, 2, 1)
ytrain.year.hist(ax=ax1)
ax2 = plt.subplot(1, 2, 2)
stats.probplot(ytrain_scaled.year, dist="norm", plot=ax2);
ax1.set_title(f'ytrain (skew={skew:.2f})');
ax2.set_title('ytrain_scaled (probplot)');

#%% Normalización input
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(Xtrain)

Xtrain_scaled = scaler.transform(Xtrain)
Xtest_scaled  = scaler.transform(Xtest)

# Numpy
X_train = Xtrain_scaled.to_numpy()
X_test  = Xtest_scaled.to_numpy()
y_train = ytrain_scaled.to_numpy()
y_test  = ytest_scaled.to_numpy()

# %% [ROLANN HIPERPARAMS]
f_act = 'linear'
lam = 1e-3
rolann_params = {'f_act':f_act, 'lam': lam}
rolann_grid = {k:[v] for k,v in rolann_params.items()}

# %% [Test ROLANN & Ridge]
from rolann import ROLANN
from sklearn.metrics import root_mean_squared_error

mdl = ROLANN(f_act=f_act, lam=lam)
mdl.fit(X_train, y_train)
w_rolann = mdl.w_
y_pred = mdl.predict(X_test)
rmse_scaled = root_mean_squared_error(y_test, y_pred)   # 0.861

if abs(skew) >= 1:  ypred = np.clip(y_pred, min_yscaled, max_yscaled)
ypred = scaler_y.inverse_transform(y_pred)
rmse = root_mean_squared_error(ytest, ypred)            # 9.576

df_pred = pd.concat([ytest, pd.DataFrame(ypred, columns=['pred'])], axis=1)
df_pred.sort_values(by='year').reset_index()[['year','pred']].plot()

#%% Ridge
from sklearn.linear_model import Ridge

mdl_lr = Ridge(alpha=lam)
mdl_lr.fit(X_train, y_train)
w_ridge = np.hstack([mdl_lr.intercept_, mdl_lr.coef_]).reshape(-1, 1)
y_pred_ridge = mdl_lr.predict(X_test)
rmse_scaled_ridge = root_mean_squared_error(y_test, y_pred_ridge)   # 0.861

if abs(skew) >= 1:  ypred = np.clip(y_pred, min_yscaled, max_yscaled)
ypred_ridge = scaler_y.inverse_transform(y_pred_ridge.reshape(-1, 1))
rmse_ridge = root_mean_squared_error(ytest, ypred_ridge)    

df_pred2 = pd.concat([ytest, pd.DataFrame(ypred_ridge, columns=['pred'])], axis=1)
df_pred2.sort_values(by='year').reset_index()[['year','pred']].plot()

# %% [CONDICIONAMIENTO]
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(Xtrain.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% [ESCENARIOS VFL]
from helpers import run_grid_search, run_grid_search_lr, merge_df_gridsearch, compute_extra_results
df_all_lst = []

#%% (1) - Split natural en 2 nodos - BCD ROLANN
split2 = [len(avg_cols), len(cov_cols)]

# BCD params
param_grid  = {
    'p': [split2],                                  # Reparto: [vars promedio, vars covarianza]]
    'mode_bcd': ['parallel', 'sequential'],         # Modos
    'lr': [.1, .3, .5, .7, .9],                     # Learning rates modo paralelo
    'tol': [1e-6],                                  # Tolerancia
    'max_iter': [250],                              # Max iteraciones
    }
print(param_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='regression')

# (exps) es una lista de tuplas (mdls_rolann, mdls_bcd); a su vez, mdls_* son listas de n_outputs elementos (1 por clase/salida)
w_rolann = exps[0][0][0].w_

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_natural')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

#%% (1) - Split natural en 2 y 4 nodos - Ridge VFL
lr_grid = {
    'lam': [lam],                               # Regularización L2
    'tol': [1e-6],
    'max_iter': [250]
    }
vfl_grid = {
    'p': [split2],
    'lr': [.1, .3, .5, .7, .9],
    'method': ['gd', 'bfgs'],
    'taylor': [False]
    }

# Lanzamos experimento - auto=binary | ovr=multi_class
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='regression')   

# Sol. optima LR
w_lr = exps_lr[0][0][0].w_

# Unimos los resultados de cada prueba en un solo dataframe
df_lr = merge_df_gridsearch(exps_lr, name='lr_natural')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)

# Guardamos en lista de resultados ambos experimentos
df_all_lst.extend([df, df_lr])

#%% (2) - Split balanceado - BCD ROLANN
m, m_min = X_train.shape[1], 2
num_groups = 4
p_bal = np.geomspace(m, m_min, num_groups, dtype=int).tolist()
param_grid['p'] = p_bal
print(param_grid)

# Lanzamos experimento
exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='regression')

# Unimos los resultados de cada prueba en un solo dataframe
df = merge_df_gridsearch(exps, name='bcd_bal')

# Calculamos resultados complementarios: loss, tiempo total, suboptimalidad y costes comm
df_bal = compute_extra_results(df, w_optim=w_rolann, X=X_train, y=y_train, lam=lam, f_act=f_act)

#%% (2) - Split balanceado - LR VFL
vfl_grid['p'] = p_bal

# Lanzamos experimento
exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='regression')   

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
p_desbal = p_bal[1:]    # quitamos 1era opcion que corresponde a 1 variable/nodo

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'msd')).is_dir():
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
# Debug dict_imps - reparto de columnas asociadas a beta=0 y p=2
for node, cols in dict_corrs[0][2].items():
    print(f"Node {node+1} - {list(Xtrain.columns[cols])}")

#%% Lanzamos experimentos CORRELACION - BCD ROLANN
df_lst = []
for beta in beta_lst:
    print(f"Experimento correlación, beta={beta}. Preparando params BCD ROLANN...")
    param_grid['p'] = list(dict_corrs[beta].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='regression')
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
    exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='regression')   
    df_lr = merge_df_gridsearch(exps_lr, name=f'lr_desbal_corr_{beta}')
    df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)
    df_lst_lr.append(df_lr)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst_lr)

#%% (4) - Split desbalanceado - IMPORTANCIA -
# Aunque el split por importancia es mas eficiente computacionalmente, igualmente
# cacheamos los índices de las divisiones realizadas para futuras ocasiones
alpha_lst = [.1, 1, 10]
p_desbal = p_bal[1:]        # quitamos 1era opcion que corresponde a 1 variable/nodo

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'msd')).is_dir():
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
    print(f"Experimento importancia, alfa={alfa}. Preparando params BCD ROLANN...")
    param_grid['p'] = list(dict_imps[alfa].values())
    # Batería de pruebas
    exps = run_grid_search(X_train, y_train, X_test, y_test, rolann_grid|param_grid, mode='regression')
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
    exps_lr = run_grid_search_lr(X_train, y_train, X_test, y_test, lr_grid|vfl_grid, mode='regression')   
    df_lr = merge_df_gridsearch(exps_lr, name=f'lr_desbal_imp_{alfa}')
    df_lr = compute_extra_results(df_lr, w_optim=w_lr, rolann=False, X=X_train, y=y_train, lam=lam)
    df_lst_lr.append(df_lr)

# Guardamos resultados experimentos
df_all_lst.extend(df_lst_lr)

#%% Concatenamos todos los dataframes y guardamos en un gigante DF para posterior análisis
df_credit = pd.concat(df_all_lst)
df_credit.info()
df_credit.to_excel("results/msd/df.xlsx", index=False)

# %%
