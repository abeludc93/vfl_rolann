# %% [DATASET] - division dada
from helpers import load_dataset
from time import perf_counter

# Carga
t_load = perf_counter()
X_train_orig, y_train_orig = load_dataset("datasets/clasificacion/LANNSVD/MNIST_train.xlsx")
X_test_orig, y_test_orig   = load_dataset("datasets/clasificacion/LANNSVD/MNIST_test.xlsx")
t_load -= perf_counter()
print(f"Carga completa en {abs(t_load):.2f} (s)")   # 10min

#%% Normalización
from sklearn.preprocessing import StandardScaler
import numpy as np

X_train, X_test = X_train_orig.to_numpy(), X_test_orig.to_numpy()
y_train, y_test = y_train_orig.to_numpy(), y_test_orig.to_numpy()

scaling = False
if scaling:
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
else:
    X_train = X_train / 255
    X_test  = X_test / 255

labels = np.unique(y_train)

# %% [ROLANN] - CLASIFICACION - PARAMS PRINCIPALES
rolann_params = {
                    'f_act':   'logs',
                    'lam':     1e0
                }

# %% [CONDICIONAMIENTO] - matriz Hessiana: A = HH^T + \lam I
from helpers import check_condition

k_A = check_condition(X_train, **rolann_params)
k_A_orig = check_condition(X_train_orig.to_numpy(), **rolann_params)

print(f"Condicionamiento A escalada: {k_A:.2f}")
print(f"Condicionamiento A sin escalar: {k_A_orig:.2f}")
print(f"\tReducción en razón del orden de: {100 * k_A_orig/k_A:.2e}")
"""
Condicionamiento A escalada: 5156579.71
Condicionamiento A sin escalar: 335310383339.80
	Reducción en razón del orden de: 6.50e+06
"""

# %% [ROLANN CENTRALIZADO OVR] (2min20s)
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

print(f"Accuracy train: {100 * acc_train:.2f} %")   # 85.76 %
print(f"Accuracy test: {100 * acc_test:.2f} %")     # 86.06 %

#%% [LOGISTIC REGRESSION OVR] (1min)
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

lr_params = {
    'lam': 1e-3,
    'tol': 0.0001,      # defecto
    'max_iter': 100     # defecto
}

# Fit
models_lr = fit_lr_models(X_train, y_train, mode='ovr', **lr_params)

# Acc
pred_lr_train = predict_models(X_train, models_lr)
pred_lr_test  = predict_models(X_test, models_lr)

#... ver comentario de abajo

acc_lr_train   = accuracy_score(y_train, pred_lr_train)
acc_lr_test    = accuracy_score(y_test, pred_lr_test)

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   # 92.84%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     # 91.95%

#%% COMENTARIOS:
"""
Había un fallo con el bloque de arriba, ya que llamaba a score2label sin necesidad.
LogisticRegression, tanto para clasif. binaria como multiclase, devuelve directamente
las etiquetas predichas si se llama a `predict`, como se hace en `predict_models`.

Para hacer uso de score2labels igual que en el bloque de BCD ROLANN usar `predict_proba`:

    score_lr_train = models_lr[0].predict_proba(X_train)
    pred_lr_train  = score2label(score_lr_train, labels)
    acc_lr_train   = accuracy_score(y_train, pred_lr_train)
"""

#%% ACC POR CLASE OVR

# (ROLANN)
classes = np.unique(y_train)
for cls, mdl in zip(classes, models):
    pred = score2label(mdl.predict(X_test))
    acc  = accuracy_score((y_test == cls).astype(int), pred)
    print(f"ROLANN acc CLS:{cls} = {acc:.2f}")

# (LR)
classes = np.unique(y_train)
pred = models_lr[0].predict(X_test)
for cls in classes:
    acc  = accuracy_score((y_test == cls).astype(int), (pred == cls).astype(int))
    print(f"LR acc CLS:{cls} = {acc:.2f}")

# Dígitos más díficiles: 8, 9, 5 ...

#%% COMPARACION PESOS CENTRALIZADOS
import matplotlib.pyplot as plt
w_rolann = np.hstack([m.w_ for m in models])

fig, axes = plt.subplots(1, 2, figsize=(14,7))
pos = axes[0].imshow(w_rolann, aspect='auto', norm='log')
fig.colorbar(pos, ax=axes[0])
pos = axes[1].imshow(models_lr[0].w_, aspect='auto', norm='log')
fig.colorbar(pos, ax=axes[1])
axes[0].set_title("ROLANN")
axes[1].set_title("LR")
plt.suptitle("Pesos sol. centralizada")
plt.tight_layout()

#%%#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% PLOTTERS
from plotters import plot_sns, plot_weights_comparison
sns_params = {
    'x': 'iter',        # Eje X
    'y': 'diff_Z',      # Eje Y
    'hue': 'lr',        # Distinto color para cada Learning Rate
    'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
    'size': 'p',        # Distinto tamaño
    'linewidth': 1.5,   # etc
    'palette': 'bright'
}

#%% [CONVERSION CLASIF. BINARIA]

# MAPPING ETIQUETAS
#    y-> 0 := False := (num != 0)
#    y-> 1 := True  := (num == 0)

binary_clasif = True
target_digit  = 2
print(f"Convirtiendo a clasificación binaria, dígito objetivo: {target_digit}")
if binary_clasif:
    print(f"[Distribución previa]\nTRAIN\n{y_train_orig.value_counts()}\nTEST\n{y_test_orig.value_counts()}")

    idx_num = np.where(y_train == target_digit)[0]
    idx_rest = np.where(y_train != target_digit)[0]
    n_digits  = len(idx_num)

    np.random.seed(1)  # reproducibilidad
    idx_sample = np.random.choice(idx_rest, n_digits, replace=False)
    idx_final = np.concatenate([idx_num, idx_sample])
    np.random.shuffle(idx_final)

    # Nuevo conjunto entrenamiento
    X_train_bin = X_train[idx_final]
    y_train_bin = (y_train[idx_final] == target_digit).astype(int)

    # Convertimos test
    y_test_bin  = (y_test == target_digit).astype(int)
    X_test_bin  = X_test.copy()

    print(f"[Distribución final]\nTRAIN\n{np.unique_counts(y_train_bin)}\nTEST\n{np.unique_counts(y_test_bin)}")

#%% [TESTING ROLANN - CLASIF. BIN]
from helpers import fit_rolann_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models = fit_rolann_models(X_train_bin, y_train_bin, mode='binary', **rolann_params)

# Acc
score_train = predict_models(X_train_bin, models)
score_test  = predict_models(X_test_bin, models)
pred_train  = score2label(score_train)  # o labels = np.unique(y_train_bin)
pred_test   = score2label(score_test)   # idem

acc_train   = accuracy_score(y_train_bin, pred_train)
acc_test    = accuracy_score(y_test_bin, pred_test)

print(f"Accuracy train: {100 * acc_train:.2f} %")   #  98.30%
print(f"Accuracy test: {100 * acc_test:.2f} %")     #  97.67%

#%% [TESTING LR  - CLASIF. BIN]
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

lr_params = {
    'lam': 1e-3,
    'solver': 'sag'
    #'tol': 0, 
    #'max_iter': 3000
}

# Fit
models_lr = fit_lr_models(X_train_bin, y_train_bin, mode='auto', **lr_params)

# Acc
pred_lr_train = predict_models(X_train_bin, models_lr)
pred_lr_test  = predict_models(X_test_bin, models_lr)

acc_lr_train   = accuracy_score(y_train_bin, pred_lr_train)
acc_lr_test    = accuracy_score(y_test_bin, pred_lr_test)

print(f"Accuracy train: {100 * acc_lr_train:.2f} %")   #  98.70%
print(f"Accuracy test: {100 * acc_lr_test:.2f} %")     #  98.31%

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% SPLIT VFL
import matplotlib.pyplot as plt
from helpers import get_column_indexes

# Columnas en 2D
cols_2d = X_train_orig.columns.values.reshape(28, 28)   # 28 x 28 img.grises

# Ej: parte derecha e izquierda
izqda = cols_2d[:, :14]
drcha = cols_2d[:, 14:]

izqda_idx = get_column_indexes(X_train_orig, izqda.flatten()).tolist()
drcha_idx = get_column_indexes(X_train_orig, drcha.flatten()).tolist()

# Imagen 
img = X_train_orig.iloc[0, :].values.reshape(28, 28)
mask1 = np.zeros_like(img, dtype=bool)
mask2 = np.zeros_like(img, dtype=bool)

mask1.reshape(-1, 1)[izqda_idx] = True
mask2.reshape(-1, 1)[drcha_idx] = True

fig, axes = plt.subplots(1, 2, figsize=(6,4))
axes[0].imshow(img, cmap="grey")
axes[1].imshow(np.where(mask1, img, np.nan), cmap="Reds")
axes[1].imshow(np.where(mask2, img, np.nan), cmap="Blues")
fig.suptitle("División p=2 vertical")
fig.tight_layout()

#%% Generalizando (PLOTTERS)
from plotters import plot_im_split

plot_im_split(img, (izqda_idx, drcha_idx))
p_nath2 = {i:v for i, v in enumerate([izqda_idx, drcha_idx])}   # dict para uso en `fit_*_models`

#%% [TESTING BCD ROLANN - CLASIF. BIN]
from helpers import fit_bcd_models, split_vfl_X

bcd_params = {
                'p':        p_nath2,
                'max_iter': 1000,
                'tol':      0,
                'mode_bcd': 'sequential',
                'lr':       0.2,
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test_bin.T, y_test_bin)
             }

# Fit
coords, clients = fit_bcd_models(X_train_bin, y_train_bin, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(np.vstack(split_vfl_X(X_train_bin.T, p_nath2)), coords)  # FIXED
score_bcd_test  = predict_models(np.vstack(split_vfl_X(X_test_bin.T, p_nath2)),  coords)
pred_vfl_train  = score2label(score_bcd_train)  # ó np.unique(y_train_bal) := [0,1]
pred_bcd_test   = score2label(score_bcd_test)

acc_bcd_train   = accuracy_score(y_train_bin, pred_vfl_train)
acc_bcd_test    = accuracy_score(y_test_bin, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 98.30%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 97.67%

#%% COMENTARIOS:
"""
Ojo! Hay un fallo con las métricas que se sacan en el bloque anterior pero que no afecta
a la función `fit_bcd_models` ni a los scores que vienen en el dataframe de resultados del
coordinador.

Que sucede?
    La exactitud reportada del bloque anterior no coincide con la que debería ni con la
    vista en el dataframe de resultados del coordinador.

Explicación?
    Hay que tener en cuenta que el coordinador expone sus pesos por simple concatenación
    de los pesos individuales de cada cliente, pero como estos corresponden a variables
    disjuntas, el orden de sus componentes no coincide con el de los pesos dados por ROLANN
    en su versión centralizada.

    Esto se corrige dentro de `fit_bcd_models` y similares automáticamente ya que se "reordenan"
    los pesos w_rolann (centralizado) de manera que estén alineados con el split vertical de
    cada escenario del coordinador.
    
    Al mismo tiempo, también se alinean las columnas (variables) de Xtest y Xtrain. Entonces, la
    predicción y evaluación (scores) se hacen correctamente.

Fallo?
    En el bloque anterior, a `predict_models` se le mandan los conjuntos de datos sin alinear a
    un modelo con los pesos reordenados.

    Por la misma razon también falla la comparación visual de los pesos de la versión centralizada
    contra la versión federada del coordinador, por que no alineamos los pesos 1ero.

Código ejemplo:

    # Alineación "coordinador"
    w_r = arrange_weights_by_split(models[0].w_, p_nath2)
    plt.imshow(w_r[1:].reshape(-1,28))
    plt.imshow(coords[0].w_[1:].reshape(-1,28))

"""

#%% HELPERs
from plotters import plot_im_weights

w_rolann = models[0].w_
w_bcd = coords[0].w_

plot_im_weights(w_rolann, w_bcd, p_nath2, (28,28))  # OJO: ya se reordena w_bcd dentro de la fn

#%% [TESTING VFL LR - CLASIF. BIN]
from helpers import fit_lrvfl_models, split_vfl_X

vfl_params = {
    'p': p_nath2,

    'lam': 1e-3,
    'max_iter': 5000,
    'tol': 0,
    
    'lr': 0.2,
    'method': 'gd',
    'taylor': False,
    'w_lr': [models_lr[0].w_],
    'test_data': (X_test_bin.T, y_test_bin)
}

# Fit
coords_vfl, clients_vfl = fit_lrvfl_models(X_train_bin, y_train_bin, **(lr_params|vfl_params), mode='binary')

# Acc
score_vfl_train = predict_models(np.vstack(split_vfl_X(X_train_bin.T, p_nath2)), coords_vfl)
score_vfl_test  = predict_models(np.vstack(split_vfl_X(X_test_bin.T, p_nath2)), coords_vfl)

pred_vfl_train  = score2label(score_vfl_train)
pred_vfl_test   = score2label(score_vfl_test)

acc_vfl_train   = accuracy_score(y_train_bin, pred_vfl_train)
acc_vfl_test    = accuracy_score(y_test_bin, pred_vfl_test)

print(f"Accuracy train: {100 * acc_vfl_train:.2f} %")   # 98.41%
print(f"Accuracy test: {100 * acc_vfl_test:.2f} %")     # 97.99%

# %% PLOT WEIGHTS VFL LR
from plotters import plot_im_weights

w_lr = models_lr[0].w_
w_vfl = coords_vfl[0].w_

plot_im_weights(w_lr, w_vfl, p_nath2, (28,28))

#%% COMENTARIOS
#PESOS
from plotters import plot_weights_comparison
from helpers import disarrange_weights_by_split
plot_weights_comparison(w_lr, disarrange_weights_by_split(w_vfl, p_nath2), 'LR', 'VFL')

"""
Los pesos no dan lo mismo en este escenario, problema de equivalencia entre regularizaciones?
Le he dado bastantes vueltas y en principio la formulación teórica debe ser equivalente.
A mi favor decir que incluso cambio entre distintos solvers de LogisticRegression de sklearn, 
para los mismos parametros, salen pesos finales diferentes, así que no me voy a concentrar
demasiado en esto.

Sigue siendo un comportamiento curioso, ya que con lam_lr = 1e-3 * N y lam_vfl=1e-3
me salen los pesos casi solapados...

Puede que sea cosa de la implementación interna de cada solver... 
Puede que hagan directamente la suma -> e = p - y; grad = X.T @ e
Sin hacer promedios (como hago yo en mi log-loss) -> -np.mean(y * ln(p) + (1-y) * ln(1-p))
...

"""
# %%
# TIP:  arrange_weights_by_split    : w_centralizado |> w_vfl
#       disarrange_weights_by_split : w_vfl |> w_centralizado

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% SPLITS VFL - CONTINUACION
from itertools import chain
from plotters import plot_im_split
from helpers import get_column_indexes

# Img ejemplo
img = X_train_orig.iloc[0, :].values.reshape(28, 28)

# Columnas en 2D
cols_2d = X_train_orig.columns.values.reshape(28, 28)   # 28 x 28 img.grises

#%% (Izquierda y derecha)
izqda = cols_2d[:, :14]
drcha = cols_2d[:, 14:]

izqda_idx = get_column_indexes(X_train_orig, izqda.flatten()).tolist()
drcha_idx = get_column_indexes(X_train_orig, drcha.flatten()).tolist()
split_H2  = (izqda_idx, drcha_idx) 

p_nat_H2 = {i:v for i, v in enumerate(split_H2)}
plot_im_split(img, split_H2)

assert sorted(list(chain(*split_H2))) == list(range(0, 28*28))

#%% (Arriba y abajo)
arriba = cols_2d[:14, :]
abajo  = cols_2d[14:, :]

arriba_idx = get_column_indexes(X_train_orig, arriba.flatten()).tolist()
abajo_idx  = get_column_indexes(X_train_orig, abajo.flatten()).tolist()
split_V2  = (arriba_idx, abajo_idx) 

p_nat_V2 = {i:v for i, v in enumerate(split_V2)}   # dict para uso en `fit_*_models`
plot_im_split(img, split_V2)

assert sorted(list(chain(*split_V2))) == list(range(0, 28*28))

#%% (4 esquinas)
sup_izq = cols_2d[:14, :14]
sup_dch = cols_2d[:14, 14:]
inf_izq = cols_2d[14:, :14]
inf_dch = cols_2d[14:, 14:]

split_idx_4 = [get_column_indexes(X_train_orig, split.flatten()).tolist()
               for split in (sup_izq, sup_dch, inf_izq, inf_dch)]

p_nat_E4 = {i:v for i, v in enumerate(split_idx_4)}   # dict para uso en `fit_*_models`
plot_im_split(img, split_idx_4)

assert sorted(list(chain(*split_idx_4))) == list(range(0, 28*28))

#%% (2 cuadrados concéntricos)
pad = 6
len_pad =  28 - pad

int_cdr = cols_2d[pad:len_pad, pad:len_pad]
int_cdr_idx = get_column_indexes(X_train_orig, int_cdr.flatten()).tolist()

ext_cdr = set(cols_2d.flatten()) - set(int_cdr.flatten())
ext_cdr_idx = get_column_indexes(X_train_orig, ext_cdr).tolist()

split_idx_cdr2 = (ext_cdr_idx, int_cdr_idx)

p_nat_C2 = {i:v for i, v in enumerate(split_idx_cdr2)}   # dict para uso en `fit_*_models`
plot_im_split(img, split_idx_cdr2)

assert sorted(list(chain(*split_idx_cdr2))) == list(range(0, 28*28))

#%% (4 franjas)
pad = 7
len_pad = 28 - pad

sup_cdr = cols_2d[:pad, :]
inf_cdr = cols_2d[len_pad:, :]
izq_cdr = cols_2d[pad:len_pad, :pad]
dch_cdr = cols_2d[pad:len_pad, len_pad:]

int_cdr = cols_2d[pad:len_pad, pad:len_pad]

split_idx_F4 = [get_column_indexes(X_train_orig, split.flatten()).tolist()
                  for split in (sup_cdr, inf_cdr, izq_cdr, dch_cdr, int_cdr)]

p_nat_F4 = {i:v for i, v in enumerate(split_idx_F4)}   # dict para uso en `fit_*_models`
plot_im_split(img, split_idx_F4)

assert sorted(list(chain(*split_idx_F4))) == list(range(0, 28*28))

#%% 4 cuadrados concéntricos
# anchura de cada capa
pads = [3, 6, 9]

# cuadrados interiores sucesivos
sq1 = cols_2d[pads[0]:28-pads[0], pads[0]:28-pads[0]]
sq2 = cols_2d[pads[1]:28-pads[1], pads[1]:28-pads[1]]
sq3 = cols_2d[pads[2]:28-pads[2], pads[2]:28-pads[2]]

# sets para operaciones
all_idx = set(cols_2d.flatten())

sq1_set = set(sq1.flatten())
sq2_set = set(sq2.flatten())
sq3_set = set(sq3.flatten())

# capas tipo "cebolla"
layer0 = all_idx - sq1_set          # exterior
layer1 = sq1_set - sq2_set          # segundo anillo
layer2 = sq2_set - sq3_set          # tercer anillo
layer3 = sq3_set                    # centro

# convertir a índices reales de columnas
split_idx_cdr4 = tuple(
    get_column_indexes(X_train_orig, list(layer)).tolist()
    for layer in [layer0, layer1, layer2, layer3]
)

p_nat_C4 = {i: v for i, v in enumerate(split_idx_cdr4)}

plot_im_split(img, split_idx_cdr4)

assert sorted(chain(*split_idx_cdr4)) == list(range(28*28))

# %% (BALANCEADO - P nodos)
# Divisores de 28x28 = 784 := { 1, 2, 4, 7, 8, 14, 16, 28, 49, 56, 98, 112, 196, 392 y 784 }
from plotters import plot_im_split
len_img = 28 * 28
p_bal = [4, 7, 14]

#%% Índices adyacentes
p_bal_idx = {}
for p in p_bal:
    p_bal_idx[p] = {i: list(range((len_img//p)*i, (len_img//p)*i + (len_img//p)))
                    for i in range(p)}
for p in p_bal:
    plot_im_split(img, p_bal_idx[p].values())

#%% Índices aleatorios
p_bal_rnd = p_bal[:-1]
p_bal_rndidx = {}
for p in p_bal_rnd:
    perm = np.random.permutation(len_img)
    splits = np.array_split(perm, p)

    p_bal_rndidx[p] = {
        i: split.tolist()
        for i, split in enumerate(splits)
    }

for p in p_bal_rnd:
    plot_im_split(img, p_bal_rndidx[p].values())

#%% (DESBALANCEADO  - P nodos - VertiBench): IMPORTANCIA  (CORR TARDA MUCHO)
alpha_lst = [.1, 1, 10]
p_desbal = [2, 4]

from pathlib import Path
import pickle

# Si está guardada se carga y si no se genera
save = True
if (pathdir := Path.cwd().joinpath('cache', 'mnist')).is_dir():
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
            alpha_idx = imp_splitter.split_indices(X_train_bin)
            dict_imps[alpha][p] = {node: list(map(int, idx)) for node, idx in enumerate(alpha_idx)}

    with open(pathfile, "wb") as f:
        pickle.dump(dict_imps, f)

print(f"Split importancia realizado: {p_desbal} nodos con {alpha_lst} alfas")

#%% VISUALIZACION
for alfa in alpha_lst:
    for p in p_desbal:
        plot_im_split(img, dict_imps[alfa][p].values(), title=f"imp={alfa}, p={p}")

#%% RECOPILACION DE SPLITS VFL
p_grid_nat      = {
    2: [p_nat_H2, p_nat_V2, p_nat_C2],
    4: [p_nat_E4, p_nat_F4, p_nat_C4]
}

p_grid_bal      = [p_bal_idx, p_bal_rndidx]

p_grid_desbal   = [dict_imps[alfa] for alfa in alpha_lst]

#%% SPLITS INTERESANTES p=4
p_4s = {
    '4Esquinas': p_nat_E4,        
    '4Cuadrados': p_nat_C4,
    '4Balanceados': p_bal_idx[4],
    '10Imp': dict_imps[10][4],
    '0.1Imp': dict_imps[0.1][4]
}
for nome, p in p_4s.items():
    plot_im_split(img, p.values(), title=nome)

#%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% [BCD ROLANN - CLASIF. BIN]
from helpers import fit_bcd_models, split_vfl_X
from plotters import plot_im_weights

p_target = p_bal_idx[14]
bcd_params = {
                'p':        p_target,
                'max_iter': 1000,
                'tol':      0,
                'mode_bcd': 'sequential',
                'lr':       0.2,
                'w_rolann': [mdl.w_ for mdl in models],
                'test_data': (X_test_bin.T, y_test_bin)
             }

# Fit
coords, clients = fit_bcd_models(X_train_bin, y_train_bin, **(rolann_params|bcd_params), mode='binary')

# Acc
score_bcd_train = predict_models(np.vstack(split_vfl_X(X_train_bin.T, p_target)), coords)  # FIXED
score_bcd_test  = predict_models(np.vstack(split_vfl_X(X_test_bin.T, p_target)),  coords)
pred_vfl_train  = score2label(score_bcd_train)  # ó np.unique(y_train_bal) := [0,1]
pred_bcd_test   = score2label(score_bcd_test)

acc_bcd_train   = accuracy_score(y_train_bin, pred_vfl_train)
acc_bcd_test    = accuracy_score(y_test_bin, pred_bcd_test)

print(f"Accuracy train: {100 * acc_bcd_train:.2f} %")   # 98.30%
print(f"Accuracy test: {100 * acc_bcd_test:.2f} %")     # 97.67%

# Pesos
w_rolann = models[0].w_
w_bcd = coords[0].w_
plot_im_weights(w_rolann, w_bcd, p_target, (28,28))

#%% [VFL LR - CLASIF. BIN]
from helpers import fit_lrvfl_models, split_vfl_X

p_target = p_bal_idx[14]
vfl_params = {
    'p': p_target,

    'lam': 1e-3,
    'max_iter': 5000,
    'tol': 0,
    
    'lr': 0.2,
    'method': 'gd',
    'taylor': False,
    'w_lr': [models_lr[0].w_],
    'test_data': (X_test_bin.T, y_test_bin)
}

# Fit
coords_vfl, clients_vfl = fit_lrvfl_models(X_train_bin, y_train_bin, **(lr_params|vfl_params), mode='binary')

# Acc
score_vfl_train = predict_models(np.vstack(split_vfl_X(X_train_bin.T, p_target)), coords_vfl)
score_vfl_test  = predict_models(np.vstack(split_vfl_X(X_test_bin.T, p_target)), coords_vfl)

pred_vfl_train  = score2label(score_vfl_train)
pred_vfl_test   = score2label(score_vfl_test)

acc_vfl_train   = accuracy_score(y_train_bin, pred_vfl_train)
acc_vfl_test    = accuracy_score(y_test_bin, pred_vfl_test)

print(f"Accuracy train: {100 * acc_vfl_train:.2f} %")   # 98.41%
print(f"Accuracy test: {100 * acc_vfl_test:.2f} %")     # 97.99%

# Pesos
w_lr = models_lr[0].w_
w_vfl = coords_vfl[0].w_
plot_im_weights(w_lr, w_vfl, p_target, (28,28))

# %%
