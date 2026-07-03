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
X, y = load_dataset("datasets/clasificacion/FedHEONN/MiniBoone.xlsx")

# No hace falta mapear la clase positiva: ya está dada (1, 0)

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

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 89.55 %
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 89.62 %

# Pesos opt. y funcion objetivo
from rolann import compute_loss_J_w
w_rolann = models[0].w_
loss_w_rolann = compute_loss_J_w(w_rolann, X_train, y_train, lam, f_act)

# %% [BCD ROLANN PAR]
from helpers import fit_bcd_models

# Número de nodos y params entrenamiento
bcd_params = {
                'p':        [40, 5, 5],
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

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 89.53 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 89.64 %

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
coords2, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(X_train.T, coords2)
score_bcd_test  = predict_models(X_test.T, coords2)
pred_bcd_train  = score2label(score_bcd_train, labels)
pred_bcd_test   = score2label(score_bcd_test, labels)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 89.54 %
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 89.64 %

# Resultados
df_bcd2 = [coord.get_results() for coord in coords2]

#%%
