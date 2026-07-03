# %% [ROLANN] - CLASIFICACION - PARAMS PRINCIPALES
rolann_params = {
                    'f_act':   'logs',
                    'lam':     1e-3
                }

# %% [DATASET]
from helpers import load_dataset
from time import perf_counter

# Carga
t_load = perf_counter()
X_train_orig, y_train_orig = load_dataset("datasets/clasificacion/LANNSVD/AMLALL_train.xlsx")
X_test_orig, y_test_orig   = load_dataset("datasets/clasificacion/LANNSVD/AMLALL_test.xlsx")
t_load -= perf_counter()
print(f"Carga completa en {abs(t_load):.2f} (s)")

#%% Normalización
from sklearn.preprocessing import StandardScaler
import numpy as np

X_train, X_test = X_train_orig.to_numpy(), X_test_orig.to_numpy()
y_train, y_test = y_train_orig.to_numpy(), y_test_orig.to_numpy()

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

labels = np.unique(y_train)

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
models = fit_rolann_models(X_train, y_train, mode='binary', **rolann_params)

# Acc
score_train = predict_models(X_train, models)
score_test  = predict_models(X_test, models)
pred_train  = score2label(score_train, labels)
pred_test   = score2label(score_test, labels)

acc_train   = accuracy_score(y_train, pred_train)
acc_test    = accuracy_score(y_test, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 100.00 %
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 91.18 %

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
coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords)
score_bcd_test  = predict_models(X_test.T, coords)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 100.00 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 91.18 %

# Resultados
df_bcd = [coord.get_results() for coord in coords]

# %% [BCD ROLANN SEQ]
from helpers import fit_bcd_models

# Número de nodos y params entrenamiento
bcd_params = {
                'p':        [1, 7128],
                'max_iter': 50,
                'tol':      0,
                'lr':       1,
                'mode_bcd': 'sequential',
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test.T, y_test)
             }

# Fit
coords2, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords2)
score_bcd_test  = predict_models(X_test.T, coords2)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 100.00 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 91.18 %

# Resultados
df_bcd2 = [coord.get_results() for coord in coords2]

#%%
"""
Hay un comportamiento curioso cuando se aumenta el numero de nodos:

La precisión sobre test se degrada. Cuando se hace un grafico tipo stem para comparar
los pesos de ROLANN vs BCD se puede ver como coinciden excepto en el nodo que incluye el bias.

Sospecho que se da en casos en donde el bias tiene una magnitud muy superior al resto de pesos.
En ROLANN, podemos ver que el bias = -1.2, mientras que el resto de pesos son del orden de 10^-3.

En el BCD, el nodo responsable que debe incluir el bias es el que presenta más variación con respecto
a los pesos respectivos de la solución centralizada y parece no ser capaz de "equilibrarlos" correctamente.
"""