# %% [DATASET] - division 70/30
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Carga TODO: meter de todas, generar idx para hacer combinaciones en diccionarios 'p'
n_smpl = 10000
n_feat = 100
n_inf  = 50
n_red  = 25 # Más redundancia = peor condicionamiento
n_rep  = 20
n_rnd  = n_feat - (n_inf + n_red + n_rep)

rng_idx = list(range(n_feat))
idx_inf = rng_idx[: n_inf]
idx_red = rng_idx[n_inf: n_inf+n_red]
idx_rep = rng_idx[n_inf+n_red: n_inf+n_red+n_rep]
idx_rnd = rng_idx[n_inf+n_red+n_rep:]

dataset_params = {
    'shuffle':              False,
    'n_clusters_per_class': 4,

    'n_samples':            n_smpl,
    'n_features':           n_feat,
    'n_informative':        n_inf,
    'n_redundant':          n_red,
    'n_repeated':           n_rep,
}
X, y = make_classification(**dataset_params, random_state=1)
print(f"Reparto clases [1|0]: {np.unique_counts(y)}")

# DataFrame para identificar columnas
X_df = pd.DataFrame(X, columns=[f"n_inf_{i+1}" for i in range(n_inf)] + \
                               [f"n_red_{i+1}" for i in range(n_red)] + \
                               [f"n_rep_{i+1}" for i in range(n_rep)] + \
                               [f"n_rnd_{i+1}" for i in range(n_rnd)]
                    )
y_df = pd.Series(y, name="class")

# Condicionamiento
poison = False
if poison:
    print("Envenenando dataset. Ampliando escalas. Ignoramos normalización...")
    # Empeoramos escalas
    X_df.iloc[:, :(n_inf + n_red)] = X_df.iloc[:, :(n_inf + n_red)] * 1000.0
    X_df.iloc[:, (n_inf + n_red):] = X_df.iloc[:, (n_inf + n_red):] * 0.0001

# División datos
X_train_orig, X_test_orig, y_train_orig, y_test_orig = \
    train_test_split(X_df, y_df, test_size=0.2, stratify=y, random_state=1)

if poison:
    # No normalizamos
    Xtrain = X_train_orig
    Xtest  = X_test_orig
    ytrain = y_train_orig
    ytest  = y_test_orig
else:
    print("Normalizando dataset...")
    # Normalización (datos entrada)
    scaler = StandardScaler().set_output(transform='pandas')
    scaler.fit(X_train_orig)
    
    # Formato pandas
    Xtrain = scaler.transform(X_train_orig)
    Xtest  = scaler.transform(X_test_orig)
    ytrain = y_train_orig
    ytest  = y_test_orig

# Formato numpy
X_train = Xtrain.to_numpy()
X_test  = Xtest.to_numpy()
y_train = ytrain.to_numpy()
y_test  = ytest.to_numpy()

bias                = np.ones(len(X_train)).reshape(-1,1)
X_train_bias        = np.hstack([bias , X_train])
X_train_orig_bias   = np.hstack([bias , X_train_orig.to_numpy()])

#%% PLOTTERS
import matplotlib.pyplot as plt
import seaborn as sns
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

# %% [HIPERPARAMS]
rolann_params = {
    'f_act':    'logs',
    'lam':      1e-5
}
lr_params = {
    'lam':      1e-1
}

#%% CONDICIONAMIENTO
from helpers import check_condition, check_condition_lr

kA  = check_condition(X_train_bias, **rolann_params)
kA2 = check_condition(X_train_orig_bias, **rolann_params)

print("ROLANN")
print(f"{'Cond. k(A)->':20}{kA:.2f}")
print(f"{'Cond. k(A_sc)->':20}{kA2:.2f}")

kH  = check_condition_lr(X_train_bias, **lr_params)
kH2 = check_condition_lr(X_train_orig_bias, **lr_params)

print("LR")
print(f"{'Cond. k(H)->':20}{kH:.2f}")
print(f"{'Cond. k(H_sc)->':20}{kH2:.2f}")

#%% RADIO ESPECTRAL SEGUN SPLIT
from helpers import check_spectral_radius

p = 8
mode = "sequential"
lr = 1
lr2 = 1.9

rho_M_iter = check_spectral_radius(X_train, p, **rolann_params|{'mode_bcd': mode, 'lr': lr})
rho_M_iter2 = check_spectral_radius(X_train, p, **rolann_params|{'mode_bcd': mode, 'lr': lr2})
print(f"{'Split->':40}{p}")
print(f"{'Modo->':40}{mode}")
print(f"{'lr->':40}{lr}")
print(f"{'Radio espectral | rho(M_iter)->':40}{rho_M_iter:.6f}")

#%% GRAFICA RHO VS LR
mode = "parallel"
x = np.arange(0, 2, .01)[1:]
y_par = [check_spectral_radius(X_train, p, **rolann_params|{'mode_bcd': 'parallel', 'lr': i})
         for i in x]
y_seq = [check_spectral_radius(X_train, p, **rolann_params|{'mode_bcd': 'sequential', 'lr': i})
         for i in x]
plt.subplot(1, 2, 1)
plt.plot(x, y_par, label=r'$\rho(\mathbf{M}_{JOR})$')
plt.xlabel('iter')
plt.ylabel('-')
plt.legend()
plt.title("Parallel")
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(x, y_seq, label=r'$\rho(\mathbf{M}_{SOR})$')
plt.xlabel('iter')
plt.ylabel('-')
plt.legend()
plt.title("Sequential")
plt.grid()

#%% BARRIDO RADIO ESPECTRAL
from helpers import product_grid, check_spectral_radius
grid = {
    'mode_bcd': ['sequential', 'parallel'],
    'lr': np.arange(0.1, 1, 0.1).tolist(),
    'p': np.arange(2, n_feat, 2).tolist()
}

df_spectral_radius = {'split': [], 'mode': [], 'lr': [], 'rho': []}
for params in product_grid(grid):
    rho_M_iter = check_spectral_radius(X_train, **rolann_params|params)
    df_spectral_radius['split'].append(params['p'])
    df_spectral_radius['mode'].append(params['mode_bcd'])
    df_spectral_radius['lr'].append(params['lr'])
    df_spectral_radius['rho'].append(rho_M_iter)

df_spectral_radius = pd.DataFrame.from_dict(df_spectral_radius).sort_values(by='rho', ascending=False)
df_spectral_radius

#%% GRAFICA BARRIDO ESPECTRAL
import seaborn as sns
g = sns.catplot(
    data=df_spectral_radius,
    kind="bar",
    x="split",
    y="rho",
    col="mode",
    hue="lr",
    palette='coolwarm'
)

for ax in g.axes.flat:
    ax.axhline(y=1, color="red", linestyle="--", linewidth=.5)
    ax.axhspan(0, 1, alpha=0.1)

g = sns.catplot(
    df_spectral_radius.assign(rho_minus_1=lambda x: x.rho - 1),
    kind="bar",
    x="split",
    y="rho_minus_1",
    col="mode",
    hue="lr",
    palette='coolwarm'
)

for ax in g.axes.flat:
    ax.axhline(0, color="red", ls="--", linewidth=.5)
    ax.set_yscale("symlog", linthresh=1e-2, linscale=5)

# %% [VertiBench] importancia
from vertibench.Splitter import ImportanceSplitter
alpha_lst = [.1 , .2, .3]
dict_imps = {alpha: {} for alpha in alpha_lst}
p = 2

for alpha in alpha_lst:
    print(f"ImportanceSplitter para p={p}, alpha={alpha}...")
    imp_splitter = ImportanceSplitter(num_parties=p, weights=alpha, seed=1)
    alpha_idx = imp_splitter.split_indices(X_train)
    dict_imps[alpha][p] = {node: list(map(int, idx)) for node, idx in enumerate(alpha_idx)}

for alpha in alpha_lst:
    print(f"alpha: {alpha}", end="\t")
    print([f"p_i={p_i+1}: {len(split)}" for p_i, split in dict_imps[alpha][p].items()])

#%% [TESTING ROLANN]
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

print(f"ROLANN acc train: {100 * acc_train:.2f} %")
print(f"ROLANN acc test: {100 * acc_test:.2f} %")

#%% [TESTING BCD]
from helpers import fit_bcd_models

#p_bcd = {0: list(range(0, 5)), 1: list(range(5, 10))}
#p_bcd = {0: list(range(0, 10, 2)), 1: list(range(1, 10, 2))}
#p_bcd = dict_imps[0.1][2]
p_bcd = 8
bcd_params = {
    'p':        p_bcd,
    'max_iter': 1000,
    'tol':      0,
    'mode_bcd': 'sequential',
    'lr':       1,
    'w_rolann': [mdl.w_ for mdl in models],
    'test_data': (X_test.T, y_test)
}

# Fit
with np.errstate(over='ignore'):
    coords, clients = fit_bcd_models(X_train, y_train, **(rolann_params|bcd_params), mode='binary')

# Acc
if type(p_fit := bcd_params['p']) == dict:
    from helpers import split_vfl_X
    XtrainT = np.vstack(split_vfl_X(X_train.T, p_fit))
    XtestT  = np.vstack(split_vfl_X(X_test.T, p_fit))
else:
    XtrainT = X_train.T
    XtestT  = X_test.T
score_bcd_train = predict_models(XtrainT, coords)
score_bcd_test  = predict_models(XtestT, coords)
pred_bcd_train  = score2label(score_bcd_train)
pred_bcd_test   = score2label(score_bcd_test)

acc_bcd_train   = accuracy_score(y_train, pred_bcd_train)
acc_bcd_test    = accuracy_score(y_test, pred_bcd_test)

print(f"BCD acc train: {100 * acc_bcd_train:.2f} %")
print(f"BCD acc test: {100 * acc_bcd_test:.2f} %")

df_bcd = coords[0].get_results()
df_bcd_clients = coords[0].get_client_results(name='cliente')

#%% [TESTING LR]
from helpers import fit_lr_models, score2label, predict_models
from sklearn.metrics import accuracy_score

# Fit
models_lr = fit_lr_models(X_train, y_train, mode='auto', **lr_params)

# Acc
pred_train = predict_models(X_train, models_lr)
pred_test  = predict_models(X_test, models_lr)

acc_lr_train   = accuracy_score(y_train, pred_train)
acc_lr_test    = accuracy_score(y_test, pred_test)

print(f"LR acc train: {100 * acc_lr_train:.2f} %")
print(f"LR acc test: {100 * acc_lr_test:.2f} %")

#%% [TESTING VFL LR]
from helpers import fit_lrvfl_models

vfl_params = {
    #'p':        {0: idx_inf + idx_rep, 1: idx_red + idx_rnd},
    'p':        {0: idx_inf + idx_rep, 1: idx_red + idx_rnd},
    #'p':        [n_feat - n_rnd, n_rnd],
    #'p':        dict_imps[0.1][2],
    #'p':        2,
    'lr':       1e-5,
    'tol':      0,
    'max_iter': 10000,
    'method':   'gd',
    'taylor':   False,
    'w_lr':     [models_lr[0].w_],
    'test_data':(X_test.T, y_test)
}

# Fit
coords_lr, clients_lr = fit_lrvfl_models(X_train, y_train, **(lr_params|vfl_params), mode='binary')

# Acc
if type(p_fit := vfl_params['p']) == dict:
    from helpers import split_vfl_X
    XtrainT = np.vstack(split_vfl_X(X_train.T, p_fit))
    XtestT  = np.vstack(split_vfl_X(X_test.T, p_fit))
else:
    XtrainT = X_train.T
    XtestT  = X_test.T
score_vfl_train = predict_models(XtrainT, coords_lr)
score_vfl_test  = predict_models(XtestT, coords_lr)
pred_vfl_train  = score2label(score_vfl_train)
pred_vfl_test   = score2label(score_vfl_test)

acc_vfl_train   = accuracy_score(y_train, pred_vfl_train)
acc_vfl_test    = accuracy_score(y_test, pred_vfl_test)

print(f"VFL acc train: {100 * acc_vfl_train:.2f} %")
print(f"VFL acc test: {100 * acc_vfl_test:.2f} %")

df_vfl = coords_lr[0].get_results()
df_vfl_clients = coords_lr[0].get_client_results(name='cliente')

# %% [INSPECCIONAMOS DIFERENCIAS]
import matplotlib.pyplot as plt

# Coordinador: convergencia
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle("BCD | VFL")
axes = axes.flatten()

# BCD
plot_sns(df_bcd,
         **sns_params|{'y': 'diff_Z', 'ax': axes[0]})
plot_sns(df_bcd,
         **sns_params|{'y': 'diff_w', 'ax': axes[2]})
plot_sns(df_bcd,
         **sns_params|{'y': 'diff_w_rolann', 'ax': axes[4]})
# VFL
plot_sns(df_vfl,
         **sns_params|{'y': 'diff_Z', 'ax': axes[1]})
plot_sns(df_vfl,
         **sns_params|{'y': 'diff_w', 'ax': axes[3]})
plot_sns(df_vfl,
         **sns_params|{'y': 'diff_w_lr', 'ax': axes[5]})

# Coordinador: scores
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
fig.suptitle("BCD | VFL")
axes = axes.flatten()

# BCD
plot_sns(df_bcd, plot_log=False, tol=acc_train,
         **sns_params|{'y': 'score_train', 'ax': axes[0]})
plot_sns(df_bcd, plot_log=False, tol=acc_test,
         **sns_params|{'y': 'score_test', 'ax': axes[2]})

# VFL
plot_sns(df_vfl, plot_log=False, tol=acc_lr_train,
         **sns_params|{'y': 'score_train', 'ax': axes[1]})
plot_sns(df_vfl, plot_log=False, tol=acc_lr_test,
         **sns_params|{'y': 'score_test', 'ax': axes[3]})

# Clientes: normas gradiente, residuos e inputs neto (z_i)
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
fig.suptitle("BCD | VFL")
axes = axes.flatten()

# BCD
plot_sns(df_bcd_clients, plot_log=False,
         **sns_params|{'y': 'norm_grd', 'hue': 'name', 'ax': axes[0], 'size':None, 'style':None})
plot_sns(df_bcd_clients, plot_log=False,
         **sns_params|{'y': 'norm_res', 'hue': 'name', 'ax': axes[2], 'size':None, 'style':None})
plot_sns(df_bcd_clients, plot_log=False,
         **sns_params|{'y': 'norm_inp', 'hue': 'name', 'ax': axes[4], 'size':None, 'style':None})
# VFL
plot_sns(df_vfl_clients, plot_log=False,
         **sns_params|{'y': 'norm_grd', 'hue': 'name', 'ax': axes[1], 'size':None, 'style':None})
plot_sns(df_vfl_clients, plot_log=False,
         **sns_params|{'y': 'norm_res', 'hue': 'name', 'ax': axes[3], 'size':None, 'style':None})
plot_sns(df_vfl_clients, plot_log=False,
         **sns_params|{'y': 'norm_inp', 'hue': 'name', 'ax': axes[5], 'size':None, 'style':None})

#%% DIFERENCIA EN PESOS: BCD VS LR
# BCD
from helpers import disarrange_weights_by_split
from plotters import plot_weights_comparison
w_rolann= models[0].w_
w_bcd   = coords[0].w_
if type(bcd_params['p']) == dict:
    w_bcd   = disarrange_weights_by_split(w_bcd, bcd_params['p'])
plot_weights_comparison(w_rolann, w_bcd)

#%% DIFERENCIA EN PESOS: LR VS VFL
w_lr    = models_lr[0].w_
w_vfl   =  coords_lr[0].w_
if type(vfl_params['p']) == dict:
    w_vfl   = disarrange_weights_by_split(w_vfl, vfl_params['p'])
plot_weights_comparison(w_lr, w_vfl, 'LR', 'VFL')

"""
Comparar los pesos de ROLANN vs BCD para ver si lo del SVD local vs SVD global hace mejor
al BCD ROLANN en algunos casos o si tiene algo que ver con el diff_w_rolann
"""
#%% ANALISIS RADIO ESPECTRAL: teorico vs empirico
split = bcd_params['p']
mode  = bcd_params['mode_bcd']
lr    = bcd_params['lr']
rho_M_iter = (df_spectral_radius
              .query('split == @split and mode == @mode and lr == @lr')
              .rho.values[0])
#%% RHO_EMP EMPIRICO
df_bcd = df_bcd.assign(rho_teo=rho_M_iter,
                       rho_emp=lambda f: f.diff_w / f.diff_w.shift(1),
                       rho_emp2=lambda f: f.diff_Z / f.diff_Z.shift(1))
df_bcd.loc[:100, ['rho_teo', 'rho_emp', 'rho_emp2']].plot()

#%% LR MAXIMO SEGUN SPLIT MODO 'PAR'
from helpers import compute_max_lr_parallel
floor = lambda n, precision=2: np.true_divide(np.floor(n * 10**precision), 10**precision)

max_lr = compute_max_lr_parallel(X_train, bcd_params['p'], **rolann_params)
print(f"Max LR modo PAR: {floor(max_lr)}")

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()
df = df_bcd.query("iter<100")

plot_sns(df,
         title=r'Convergencia $\Delta\mathbf{w}$',
         **sns_params|{'y':'diff_w', 'ax':axes[1]})

axes[0].plot(df.iter, df.rho_emp2, label=r'$\hat\phi = \Delta\mathbf{Z}^{k+1} / \Delta\mathbf{Z}^{k}$')
axes[0].axhline(rho_M_iter, linestyle='--', label=r"$\phi = \rho(\mathbf{M}_{JOR})$", color='red', alpha=0.6)
axes[0].legend()
axes[0].set_title(r'Estimación $\rho(\mathbf{M}_{iter})$')
axes[0].grid()

plt.suptitle(f"(BCD_ROLANN) - {mode} - {'divergente' if rho_M_iter > 1 else 'convergente'}")
plt.tight_layout()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes = axes.flatten()
df = df_bcd.query("iter<100")

sns.lineplot(df, x='iter', y='rho_emp', hue='lr', ax=axes[0], palette='bright', legend=False)
axes[0].axhline(rho_M_iter, linestyle='--', color='violet', label=r'$\rho(\mathbf{M}_{JOR}^0$)')
axes[0].axhline(rho_M_iter2, linestyle='-.', color='green', label=r'$\rho(\mathbf{M}_{JOR}^1)$')
axes[0].legend()
axes[0].set_title(r'Evolución de $\hat\phi$ con heurística')
axes[0].axvline(13, linestyle=':', alpha=0.5, color='grey')
axes[0].grid()

plot_sns(df,
         title=r'Convergecia con $\Delta\mathbf{Z}$',
         **sns_params|{'ax':axes[1]})

plt.suptitle(f"(BCD_ROLANN) - {mode} - {'divergente' if rho_M_iter > 1 else 'convergente'}")
plt.tight_layout()