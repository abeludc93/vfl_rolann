# %% [ROLANN] - CLASIFICACION - PARAMS PRINCIPALES
rolann_params = {
                    'f_act':   'logs',
                    'lam':     1e-3
                }

# %% [DATASET] - division 70/30
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helpers import load_dataset
import numpy as np

# Carga
X, y = load_dataset("datasets/clasificacion/FedHEONN/DryBean.xlsx")
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
labels = np.unique(y)

# Normalización
scaler = StandardScaler()   #.set_output(transform='pandas')
scaler.fit(X_train_orig)

X_train = scaler.transform(X_train_orig)
X_test = scaler.transform(X_test_orig)
y_train = y_train_orig.to_numpy()
y_test = y_test_orig.to_numpy()

# %% [CONDICIONAMIENTO] - matriz Hessiana: A = HH^T + \lam I
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")

# %% [ROLANN CENTRALIZADO]
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models = fit_rolann_models(X_train, y_train, mode='ovr', **rolann_params)

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train, labels)
pred_test   = score2label(score_test, labels)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 90.26 %
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 90.72 %

# %% [BCD ROLANN PAR]
from helpers import fit_bcd_models

# Número de nodos y params entrenamiento
bcd_params = {
                'p':        3,
                'max_iter': 200,
                'tol':      1e-3,
                'lr':       0.25,
                'mode_bcd': 'parallel',
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='ovr')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 88.03 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 87.81 %

# Resultados
df_bcd = [coord.get_results() for coord in coords]

# %% [BCD ROLANN SEQ]
from helpers import fit_bcd_models

# Número de nodos y params entrenamiento
bcd_params = {
                'p':        3,
                'max_iter': 200,
                'tol':      1e-3,
                'lr':       1,
                'mode_bcd': 'sequential',
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords2, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='ovr')

# Acc
score_bcd_train = predict_models(X_train.T, coords2)
score_bcd_test  = predict_models(X_test.T, coords2)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 90.71 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 90.60 %

# Resultados
df_bcd2 = [coord.get_results() for coord in coords2]

# %% [Visualizaciones]
#==========================================================================================================
from plotters import plot_diff, plot_score, plot_mode_diffs, plot_mode_scores, plot_correlation_map, plot_weights_comparison, plot_communication_costs, plot_score_global, plot_time_scores
from helpers import combine_multioutput_dfs, check_correlation_weights, obtain_global_scores, communication_costs

#%% Conseguimos score global de test y train para cada iteracion
from sklearn.metrics import accuracy_score

train_data  =  X_train.T, y_train
test_data   = X_test.T, y_test
f_act       = rolann_params['f_act']
df_scores   = obtain_global_scores(df_bcd, test_data, train_data, f_act, accuracy_score, labels)

plot_score_global(df_scores, labels)

#%% Calculamos costes de comunicación para el entrenamiento
k = bcd_params['max_iter']
p = bcd_params['p'] if isinstance(bcd_params['p'], int) else len(bcd_params['p'])
n = len(X_train)
df_scores['comm'] = communication_costs(k, p, n, size_bytes=8, c=len(labels), out='megabytes')

plot_communication_costs(df_scores, out='MB')

#%% Visualización frente a tiempo
df_sum = combine_multioutput_dfs(df_bcd)
plot_time_scores(df_sum.merge(df_scores, on='iter'))

# %% Diferencias diff_w, diff_z y diff_w_rolann:

# Paralelo
[plot_diff(df, f"BCD ROLANN DIFF (clase-salida {i+1})") for i, df in enumerate(df_bcd)];

# Secuencial
[plot_diff(df, f"BCD ROLANN DIFF (clase-salida {i+1})") for i, df in enumerate(df_bcd2)];

# Sumadas - global - paralelo
df_sum = combine_multioutput_dfs(df_bcd)
plot_diff(df_sum, "BCD ROLANN GLOBAL DIFF")

# Sumadas - global - secuencial
df_sum2 = combine_multioutput_dfs(df_bcd2)
plot_diff(df_sum2, "BCD ROLANN GLOBAL DIFF")

# %% Score obtenido durante bucle

# Paralelo
[plot_score(df, f"BCD ROLANN SCORE (clase-salida {i+1})") for i, df in enumerate(df_bcd)];

# Secuencial
[plot_score(df, f"BCD ROLANN SCORE (clase-salida {i+1})") for i, df in enumerate(df_bcd2)];

# %% Matriz de correlación
plot_correlation_map(X)

# %% Comparación de pesos - soluciones finales

# Paralelo
for idx, (rolann, bcd) in enumerate(zip(models, coords)):   # Por clase-salida
    plot_weights_comparison(rolann.w_, bcd.w_, title=f"Pesos (clase-salida {idx+1}) [PAR]")

# Secuencial
for idx, (rolann, bcd) in enumerate(zip(models, coords2)):   # Por clase-salida
    plot_weights_comparison(rolann.w_, bcd.w_, title=f"Pesos (clase-salida {idx+1}) [SEQ])")

# Global - suma de magnitudes (norma l1 por componentes de w):
w_rolann = np.hstack([abs(mdl.w_) for mdl in models]).sum(axis=0)
w_bcd    = np.hstack([abs(c.w_) for c in coords]).sum(axis=0)
w_bcd2    = np.hstack([abs(c.w_) for c in coords2]).sum(axis=0)

plot_weights_comparison(w_rolann, w_bcd, title=f"Norma L1 pesos ROLANN vs. BCD [PAR]")
plot_weights_comparison(w_rolann, w_bcd2, title=f"Norma L1 pesos ROLANN vs. BCD [SEQ]")

# %% Diferencia entre pesos ROLANN y BCD en variables altamente correlacionadas
for i in range(len(models)):
    print(f"Salida-clase ({i+1}) [PAR]")
    display(check_correlation_weights(X, w_rolann=models[i].w_, w_bcd=coords[i].w_, threshold=0.95))

for i in range(len(models)):
    print(f"Salida-clase ({i+1}) [SEQ]")
    display(check_correlation_weights(X, w_rolann=models[i].w_, w_bcd=coords2[i].w_, threshold=0.95))

# %% Comparación entre modos de operación BCD [SEQ]/[PAR]
from plotters import plot_mode_diffs
import pandas as pd

for i, dfs in enumerate(zip(df_bcd, df_bcd2)):
    dfpair = pd.concat(dfs, ignore_index=False)
    plot_mode_diffs(dfpair, f"BCD ROLANN DIFF (clase-salida {i+1})")

for i, dfs in enumerate(zip(df_bcd, df_bcd2)):
    dfpair = pd.concat(dfs, ignore_index=False)
    plot_mode_scores(dfpair, f"BCD ROLANN SCORE (clase-salida {i+1})")

plot_mode_diffs([df_sum, df_sum2], "BCD ROLANN GLOBAL DIFF")
plot_mode_scores([df_sum, df_sum2], "BCD ROLANN GLOBAL SCORE")

# %% ====== VertiBench
from vertibench.Splitter import ImportanceSplitter, CorrelationSplitter

#%% Split by importance
imp_splitter = ImportanceSplitter(num_parties=4, weights=[1, 1, 1, 3], seed=1)
Xs = imp_splitter.split(X_train)

#%% Split by correlation
corr_splitter = CorrelationSplitter(num_parties=4, seed=1, gpu_id=0, n_jobs=1)
Xs2 = corr_splitter.fit_split(X_train)

# %% Map columns from split
from helpers import map_columns_from_split
Dxs = map_columns_from_split(X_train_orig, Xs)
Dxs2 = map_columns_from_split(X_train_orig, Xs2)

#%%
from helpers import build_dataframe_from_split
df_Xs = build_dataframe_from_split(X_train_orig, Xs)
df_Xs2 = build_dataframe_from_split(X_train_orig, Xs2)
