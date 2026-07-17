#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Funciones de utilidad para experimentacion VFL
"""
from time import perf_counter
import itertools
import copy
# Third-party libraries
import numpy as np
import pandas as pd
from scipy import linalg as spla
from sklearn.linear_model import LogisticRegression, Ridge
# Application-modules
from bcd import BCDClient, BCDCoordinator
from lr import LRClient, LRCoordinator, RidgeClient, RidgeCoordinator, compute_loss_log, compute_loss_linear
from rolann import ROLANN, compute_loss_J_w
from activation_functions import load_act_fn, POS_VALUE, NEG_VALUE


def disarrange_weights_by_split(w_coord, p_dict):
    
    # Extraemos bias (suponemos 1er nodo) y resto de pesos
    w_coord_flt = w_coord.flatten()
    bias, weights = w_coord_flt[0], w_coord_flt[1:]

    # Máscara de pesos "originales"
    w_orig = np.empty_like(weights)
    
    # Índices de las columnas en el split vertical de `p_dict`
    split_w_idx = [idx for sublist in list(p_dict.values()) for idx in sublist]
    w_orig[split_w_idx] = weights   # las colocamos ordenadas

    # Concatenamos en vector columna
    w_ = np.concatenate([[bias], w_orig]).reshape(-1, 1)
    return w_


def arrange_weights_by_split(w, p):
    bias    = w[0]
    w_p     = split_vfl_X(w[1:], p)
    w_stack = np.vstack([bias, *w_p])
    return w_stack

def build_dataframe_from_split(df, split_lst):
    mapping = map_columns_from_split(df, split_lst)
    df_lst = [df[cols] for cols in mapping.values()]
    return df_lst


def map_columns_from_split(df, split_lst):
    # Diccionario índice subdivison -> columnas
    mapping = {}

    for i, split in enumerate(split_lst):
        cols = []
        split_cols = split.shape[1]
        for idx_col in range(split_cols):
            split_col = split[:, idx_col]
            name_col  = [col for col in df if np.array_equal(df[col], split_col)]
            if name_col:
                cols += name_col
            else:
                raise ValueError(f"No se halló correspondencia para la columna {idx_col} del split {i}")
        mapping[i] = cols

    return mapping


def get_column_indexes(df, cols):
    assert np.all([c in df.columns for c in cols])
    indexes = df.columns.get_indexer(cols)
    return indexes


def split_vfl_X(X, p):
    """
    Vertically split matrix X (m, n) with 'm' features and 'n' samples into submatrices
    (m_i, n) with 'm_i' features and 'n' samples such that sum m_i = m.

    If 'p' is an int, then it results in 'p' submatrices with equal number of m_i features.
    If 'p' is a list, then it results in len(p) submatrices with p[i] features each.
    If 'p' is a dict, then it results in len(p) submatrices with X[p[i]] features each.
    """

    m, _  = X.shape     # Dimensiones m x n
    X_lst = []          # Lista con las submatrices divididas
    
    start = 0
    if type(p) == int:
        if p > m: raise ValueError(f"({p}) no puede ser mayor que m ({m})")
        
        m_i = m // p    # Nº variables por nodo
        r   = m % p     # Resto (se las lleva el último nodo)

        for i in range(p):
            m_p = m_i + r if i == p - 1 else m_i
            end = start + m_p
            X_i = X[start : end]
            X_lst.append(X_i)
            start = end

    elif type(p) == list:
        if sum(p) != m: raise ValueError(f"Suma de p ({sum(p)}) no es igual a m ({m})")
        
        for m_p in p:
            end = start + m_p
            X_i = X[start : end]
            X_lst.append(X_i)
            start = end

    elif type(p) == dict:
        len_m_p = [len(m_p) for m_p in p.values()]
        if sum(len_m_p) != m: raise ValueError(f"Suma de valores p ({sum(len_m_p)}) no es igual a m ({m})")
        m_p_cols = []
        for m_p in p.values():
            X_i = X[m_p]
            X_lst.append(X_i)
            m_p_cols.extend(m_p)

    else:
        raise TypeError(f"Fórmato erroneo (int|list|dict): {type(p)}")

    if type(p) == dict:
        assert np.array_equal(X[m_p_cols], np.concatenate(X_lst))
    else:
        assert np.array_equal(X, np.concatenate(X_lst))
    return X_lst


def run_grid_search(X_train, y_train, X_test, y_test, param_grid, mode='ovr'):
    """
    Run experiments over a grid-search combination of hyperparameters.
    
    param_grid = {
        # ROLANN PARAMS
        'f_act': ['logs'],                      # f.act.
        'lam': [0.001, 0.01]                    # regularization

        # BCD params
        'p': [[2, 2], [4, 4]],                  # node config
        'mode_bcd': ['parallel', 'sequential'], # modes
        'lr': [0.5, 0.75, 1.0],                 # learning rates
        'tol': [1e-5]                           # tolerance
        'max_iter': [20, 50]                    # max iterations
        ...
    }
    """

    # Combinaciones posibles de hiperparams - prod. cartesiano
    combinations = product_grid(param_grid)
    print(f"Iniciando grid-search con {len(combinations)} combinaciones...")
    
    old_params = None   # Optim. para no repetir modelos centralizados
    results = []
    for i, params in enumerate(combinations):
        print(f"--- Experimento Nº {i+1}/{len(combinations)}: {params} ---")
        
        # [ROLANN] solución centralizada
        f_act       = params['f_act']
        lam         = params['lam']
        if (f_act, lam) != old_params:
            t_rolann    = perf_counter()
            mdls_rolann = fit_rolann_models(X_train, y_train, mode=mode, f_act=f_act, lam=lam)
            print(f"ROLANN centralizado entrenado en: {perf_counter() - t_rolann:.2f} (s)")

        # [BCD ROLANN]
        params['w_rolann']  = [mdl.w_ for mdl in mdls_rolann]
        params['test_data'] = (X_test.T, y_test)
        mdls_bcd, _         = fit_bcd_models(X_train, y_train, mode=mode, **params)

        # Agregamos resultados
        results.append((mdls_rolann, mdls_bcd))
        old_params = f_act, lam
    
    # Lista de tuplas con los modelos entrenados ROLANN y BCD
    return results


def load_dataset(path):
    """Load dataset given in 'path' and returns X/y dataframes."""
    print(f"Cargando dataset - {path} ...", end=" ")
    df_dic = pd.read_excel(path, sheet_name=["X","y"])
    X, y = df_dic['X'], df_dic['y']
    print(f"Listo!\nMuestras/Variables: {X.shape} | Target: {y.shape}")

    return X, y


def check_condition(X, lam=1e-3, f_act="linear"):
    """Checks condition number of matrix A = H H^T + lam I"""

    _, f_inv, f_der = load_act_fn(f_act)
    f_escalar = f_der(f_inv(POS_VALUE))

    H = X.T * f_escalar
    A = H @ H.T + lam * np.eye(len(X.T))
    k_A = np.linalg.cond(A)
    
    return k_A


def check_condition_lr(X, D=0.25, lam=1e-3):
    n, m    = X.shape
    D_diag  = np.eye(n)
    lam_diag= np.eye(m)
    H_lr    = X.T @ (D * D_diag) @ X + (lam * lam_diag)
    k_Hlr   =  np.linalg.cond(H_lr) # igual a cte de Lipschitz L = 1/4 * max_eigval(X X.T) + lamI
    return k_Hlr


def check_spectral_radius(X, p, mode_bcd="sequential", lam=1e-3, f_act="linear", **kwargs):
    """Computes spectral radius of the Gauss-Seidel iteration matrix M_GS (BCD ROLANN).
    Values closer to 1 represent slow convergence. Values closer to 0 indicate fast convergen rápida."""

    # Split X according to the VFL split in p
    X_split = split_vfl_X(X.T, p)
    # Add bias term on first Xsplit element (equiv. to first BCD client)
    bias = np.ones(len(X)).reshape(-1,1)
    X_split[0] = np.hstack([bias , X_split[0].T]).T
    X_ordered_T = np.vstack(X_split)  # (m, n)
    m = X_ordered_T.shape[0]

    # Build matrix A = H @ H.T + lam * I
    _, f_inv, f_der = load_act_fn(f_act)
    f_escalar = f_der(f_inv(POS_VALUE))
    H = X_ordered_T * f_escalar
    A = H @ H.T + lam * np.eye(m)
    
    # Decompose A in D, L, U blocks
    D = np.zeros_like(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    # VFL block sizes (m_i)
    m_i = [X_i.shape[0] for X_i in X_split]
    
    start_idx = 0
    for size in m_i:
        end_idx = start_idx + size
        # Diag block
        D[start_idx:end_idx, start_idx:end_idx] = A[start_idx:end_idx, start_idx:end_idx]
        # Lower triag. block
        L[end_idx:, start_idx:end_idx] = A[end_idx:, start_idx:end_idx]
        # Upper triag. block
        U[start_idx:end_idx, end_idx:] = A[start_idx:end_idx, end_idx:]
        
        start_idx = end_idx

    # Compute iteration matrix:
    # we use `linalg.solve` instead of `linalg.inv` as it is more numerically robust
    lr = kwargs.get('lr', None)
    try:
        if mode_bcd == 'sequential':
            if lr == 1: # GS puro
                M_iter = np.linalg.solve(D + L, -U)
            else:       # SOR
                M_iter = np.linalg.solve(D + lr * L, (1.0 - lr) * D - lr * U)
        elif mode_bcd == 'parallel':
            M_J = np.linalg.solve(D, -(L + U))
            M_iter = ((1.0 - lr) * np.eye(m) + lr * M_J) if lr else M_J
        else:
            raise ValueError(f"Modo BCD desconocido: {mode_bcd}")
    except np.linalg.LinAlgError:
    # we defer to a pseudo-inverse in case of linalg error (singular matrix)
        if mode_bcd == 'sequential':
            if lr == 1:
                M_iter = -np.linalg.pinv(D + L) @  U
            else:
                M_iter = np.linalg.pinv(D + lr * L) @ ((1.0 - lr) * D - lr * U)
        else:
            M_J = -np.linalg.pinv(D) @ (L + U)
            M_iter = ((1.0 - lr) * np.eye(m) + lr * M_J) if lr else M_J

    # Spectral radius (max abs. eigenvalue)
    rho = np.max(np.abs(np.linalg.eigvals(M_iter)))

    return rho


def compute_max_lr_parallel(X, p, lam=1e-3, f_act="linear"):
    """Computes the theoretical maximum learning rate of convergen for the
    'parallel' BCD ROLANN mode (Damped Jacobi) for a given split 'p'"""

    # Split X according to the VFL split in p
    X_split = split_vfl_X(X.T, p)
    # Add bias term on first Xsplit element (equiv. to first BCD client)
    bias = np.ones(len(X)).reshape(-1,1)
    X_split[0] = np.hstack([bias , X_split[0].T]).T
    X_ordered_T = np.vstack(X_split)  # (m, n)
    m = X_ordered_T.shape[0]

    # Build matrix A = H @ H.T + lam * I
    _, f_inv, f_der = load_act_fn(f_act)
    f_escalar = f_der(f_inv(POS_VALUE))
    H = X_ordered_T * f_escalar
    A = H @ H.T + lam * np.eye(m)
    
    # Decompose diagonal block D
    D = np.zeros_like(A)
    
    # VFL block sizes (m_i)
    m_i = [X_i.shape[0] for X_i in X_split] 
    
    start_idx = 0
    for size in m_i:
        end_idx = start_idx + size
        D[start_idx:end_idx, start_idx:end_idx] = A[start_idx:end_idx, start_idx:end_idx]
        start_idx = end_idx

    # Compute max eigval of matrix (D^{-1} @ A)
    try:
        gammas = spla.eigvals(A, D) # A v = gamma D v
    except np.linalg.LinAlgError:
        gammas = np.linalg.eigvals(np.linalg.solve(D, A))
    
    # Max learning rate
    gamma_max = np.max(np.real(gammas))
    lr_max = 2.0 / gamma_max

    return lr_max


def score2label(scores, labels=None):
    """Transforms prediction scores into prediction labels for both binary and multiclass classification.
    If no labels are given for binary classification a 0=NEG 1=POS fallback will be used.
    Otherwise, labels should have the same length as the no. of columns in `scores` array of dim. (n, c).
    """

    # Formateamos a numpy por si viene como lista o dataframe
    scores = np.asarray(scores)
    
    # Multiclase
    if scores.ndim == 2 and scores.shape[1] > 1:
        if scores.shape[1] != len(labels):
            raise ValueError(f"Dimensión de scores {scores.shape} no coincide con número de clases ({len(labels)})")
        # Obtenemos el índice de la distancia mínima a POS_VALUE
        dist_pos = np.abs(scores - POS_VALUE)
        indices = np.argmin(dist_pos, axis=1)
        pred = labels[indices]
    
    # Binaria
    else:
        scores_flat = scores.ravel()
        # Umbral como promedio entre mapeado de clases
        threshold = 0.5 * (POS_VALUE + NEG_VALUE)
        # Generamos índices: 1 si supera el umbral (POS), 0 si no (NEG)
        indices = np.where(scores_flat > threshold, 1, 0)
        if labels is None or len(labels) != 2:
            return indices  # por si no pasan las etiquetas correctas devolvemos (1, 0)
        
        # Mapeamos a las etiquetas reales: labels[0] = NEG, labels[1] = POS
        # Asumimos convención: classes[0] -> Negativo, classes[1] -> Positivo
        pred = labels[indices]

    return pred


def fit_rolann_models(X_train, y_train, mode='ovr', **params):
    """Function that fits one ROLANN model (with given `params`) to the train data
    in case of a OvR classification task or multi-output regression."""

    # Manejo de clasificación tipo OvR y regresión múltiple
    def _yield_target(y, mode='ovr'):
        if mode == 'ovr':
            classes = np.unique(y)
            for cls in classes:
                yield (y == cls).astype(int)    # mapeado clase pos=1, neg=0
        elif mode == 'binary' or mode == 'regression':
            yield y.reshape([-1, 1])            # mapeado clase pos=1, neg=0, devolvemos vector columna
        else:                                   # multioutput
            n_outputs = y.shape[1]
            for i in range(n_outputs):
                yield y[:, i]                   # devolvemos columna 'i'
    
    # Fit - un modelo por salida
    models = []
    for y_target in _yield_target(y_train, mode):
        model = ROLANN(**params) 
        model.fit(X_train, y_target)
        models.append(model)
    
    return models


def predict_models(X, models):
    """Build prediction scores for collection of models."""

    # Matriz (n x c), siendo c el nº clases o salidas
    scores = np.hstack([mdl.predict(X) for mdl in models])
    return scores


def fit_bcd_models(X_train, y_train, mode='ovr', **params):
    """Function that fits BCD ROLANN models to the train data in both OvR classification
    tasks and multi-output regression. {`params`} contains both ROLANN hyperparameters as
    well as BCD-related train parameters."""

    # Params ROLANN
    f_act       = params['f_act']
    lam         = params['lam']

    # Params BCD
    p           = params['p']
    max_iter    = params['max_iter']
    tol         = params['tol']
    lr          = params['lr']
    mode_bcd    = params['mode_bcd']
    w_rolann    = params['w_rolann']
    Xtest, ytest= params['test_data']


    # Dividimos 'X' para los distintos clientes
    X_lst_T = split_vfl_X(X_train.T, p)
    # EXPLICACION: cuando 'p' es un diccionario, el split puede estar desordenado
    # en relación a las variables, por lo que hay que re-ordenar los pesos
    # de la sol. centralizada (ROLANN) y las filas de Xtest (atributos)
    if type(p) == dict: 
        # en verdad funciona también cuando 'p' es int|list pero evitamos ops. innecesarias
        # w_rolann se tiene que reordenar teniendo en cuenta el 'bias' en idx[0]
        Xtest = np.vstack(split_vfl_X(Xtest, p))
        for i, w_r in enumerate(w_rolann):
            w_r = arrange_weights_by_split(w_r, p)
            w_rolann[i] = w_r

    # Preparamos targets 'y' en caso de clasificación OvR
    if mode == 'ovr':
        classes = np.unique(y_train) 
        assert len(classes) == len(np.unique(ytest))    # mismas clases en train que en test
        y_target = np.hstack([(y_train == cls).astype(int) for cls in classes])
        y_test = np.hstack([(ytest == cls).astype(int) for cls in classes])
    else:
        # mode = 'binary' o 'regression' o 'multioutput'
        y_target = y_train if y_train.ndim > 1 else y_train.reshape([-1, 1])
        y_test = ytest if ytest.ndim > 1 else ytest.reshape([-1, 1])

    # Clientes y coordinador
    num_nodos = len(X_lst_T)
    num_outputs = y_target.shape[1]

    lst_clients = [BCDClient(id=i+1, X=X_lst_T[i], f_act=f_act, lam=lam)
                    for i in range(num_nodos)]
    #lst_coords = [BCDCoordinator(clients=lst_clients, f_act=f_act, d=y_target[:,n], test_data=(Xtest, y_test[:,n]), w_rolann=w_rolann[n])
    #              for n in range(num_outputs)]
    from bcd import BCDCoordinator_lrp
    lst_coords = [BCDCoordinator_lrp(clients=lst_clients, f_act=f_act, d=y_target[:,n], test_data=(Xtest, y_test[:,n]), w_rolann=w_rolann[n]) 
                  for n in range(num_outputs)]

    # Fit
    for i, coord in enumerate(lst_coords):
        print(f"Entrenando BCD ROLANN para clase-salida {i+1}/{num_outputs}...")
        coord.fit(max_iter=max_iter, tol=tol, lr=lr, mode=mode_bcd)
    
    return lst_coords, lst_clients


def check_correlation_weights(X, w_rolann, w_bcd, threshold=0.9, tol=1e-3):
    """For highly correlated features, checks if the corresponding pair of weights from both the
       centralized ROLANN solution and the BCD ROLANN solution sums the same value, implying there
       might be two sets of different but equivalent solutions (weights).
       X needs to be standardized."""
    
    # Aplanamos arrays y quitamos bias
    w_c = w_rolann.flatten()[1:]
    w_b = w_bcd.flatten()[1:]
    
    # Calculamos Matriz de Correlación
    corr_matrix = X.corr()

    # Análisis
    n_features = len(corr_matrix)
    found_issues = False
    results = {'Par (i, j)': [], 'Columnas': [], 'Corr': [], 'W_rolann (i, j)': [], 'W_BCD (i, j)': [], 'Suma equiv?': []}
    for i in range(n_features):
        for j in range(i + 1, n_features):
    
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > threshold:
                found_issues = True

                # Recuperamos pares de pesos correspondientes
                wc_i, wc_j = w_c[i], w_c[j]
                wb_i, wb_j = w_b[i], w_b[j]
                
                # Suma de pesos (A veces el modelo reparte la suma total)
                sum_c = wc_i + wc_j
                sum_b = wb_i + wb_j
                diff_sum = abs(sum_c - sum_b)
                equiv   = diff_sum < tol
                
                # Guardamos resultados
                results['Par (i, j)'].append(f"({i}, {j})")
                results['Columnas'].append(f"{corr_matrix.columns[i]} | {corr_matrix.columns[j]}")
                results['Corr'].append(rho)
                results['W_rolann (i, j)'].append(f"({wc_i:.3f}, {wc_j:.3f})")
                results['W_BCD (i, j)'].append(f"({wb_i:.3f}, {wb_j:.3f})")
                results['Suma equiv?'].append(equiv)
    if not found_issues:
        print("No se encontraron pares de variables con alta correlación.")

    return pd.DataFrame.from_dict(results)


def combine_multioutput_dfs(dfs):
    sum_columns = ['diff_w', 'diff_Z', 'diff_w_rolann', 'time', 'time_setup']
    stt_columns = ['max_iter', 'tol', 'lr', 'mode', 'iter']

    sum_df = sum(df[sum_columns] for df in dfs)
    stt_df = dfs[0][stt_columns]
    df_combined = pd.concat([sum_df, stt_df], axis=1)
    return df_combined


def communication_costs(k, p, n, size_bytes=8, c=1, out='bytes'):

    iters = np.arange(1, k+1) if isinstance(k, int) else k
    comm_costs = iters * p * n * size_bytes * c
    if out == 'megabytes':
        comm_costs = comm_costs * 1e-6
    elif out  == 'gigabytes':
        comm_costs = comm_costs* 1e-9
    return pd.Series(comm_costs, name=out)


def obtain_global_scores(dfs, test_data, train_data, f_act, skmetricfn, labels=None):

    Xtest, ytest    = test_data
    Xtrain, ytrain  = train_data

    df_wi = pd.concat([df.w_i for df in dfs], axis=1)
    df_wi.columns = [f"w_{i+1}" for i in range(len(dfs))]

    f, _,  _ = load_act_fn(f_act)
    Xp_test  = np.vstack([np.ones((1, Xtest.shape[1])), Xtest])
    Xp_train = np.vstack([np.ones((1, Xtrain.shape[1])), Xtrain])

    df_scores = [df[['score_test', 'score_train']].rename(columns={'score_test': f'score_test_{i+1}',
                                                                   'score_train': f'score_train_{i+1}'}) 
                 for i, df in enumerate(dfs)]
    df_scores = pd.concat(df_scores, axis=1)
    for iter, wi in df_wi.iterrows():
        score_test  = np.vstack([f(w.T @ Xp_test) for w in wi])
        score_train = np.vstack([f(w.T @ Xp_train) for w in wi])
        if labels is not None:
            ypred_test  = score2label(score_test.T, labels)
            ypred_train = score2label(score_train.T, labels)
        else:
            ypred_test  = score_test.T
            ypred_train = score_train.T

        acc_test = skmetricfn(ytest, ypred_test)
        acc_train = skmetricfn(ytrain, ypred_train)
        df_scores.loc[iter, 'score_test'] = acc_test
        df_scores.loc[iter, 'score_train'] = acc_train
    
    df_scores['iter'] = df_scores.index + 1
    return df_scores


def fit_lr_models(X_train, y_train, mode='ovr', **params):
    """Function that fits a LogisticRegression|Ridge model (with given `params`) to the
    train data for both binary, multi-label classification tasks and single output regression.
    (mode='auto'|'ovr'|'regression' - for binary|multilabel classification| regression)"""

    # Fit
    mdl_params = params.copy()
    if mode == 'regression':
        mdl_params['alpha'] = mdl_params.pop('lam') # alpha := l2 reg param in Ridge
        model = Ridge(fit_intercept=True, **mdl_params)
    else:
        mdl_params['C'] = 1 / mdl_params.pop('lam') # C := 1/lam inverse of l2 reg param in LR
        model = LogisticRegression(penalty='l2', fit_intercept=True, multi_class=mode, **mdl_params)
    model.fit(X_train, y_train)
    
    # Save optim. weights (ridge|lr)
    if type(model.intercept_) == float or len(model.intercept_) == 1: 
        # single output regression (ridge) | binary clasif (lr)
        w = np.hstack([model.intercept_, model.coef_.ravel()]).reshape(-1, 1)
    else:
        # multi-output (ridge) | multilabel clasif (lr)
        w = np.hstack([model.intercept_.reshape(-1, 1), model.coef_]).T      

    model.w_ = w
    
    return [model]  # Por uniformizar tipo devuelto con fit_bcd_models... será lista con 1 elemento


def fit_lrvfl_models(X_train, y_train, mode='ovr', **params):
    """Function that fits LR VFL models to the train data in both OvR and binary
    classification tasks. {`params`} contains both LR hyperparameters as well as
    VFL-related train parameters."""

    # Params LR
    lam         = params['lam']
    tol         = params['tol']
    max_iter    = params['max_iter']

    # Params VFL
    p           = params['p']
    lr          = params['lr']
    method      = params['method']
    taylor      = params['taylor']
    w_lr        = params['w_lr']
    Xtest, ytest= params['test_data']


    # Dividimos 'X' para los distintos clientes
    X_lst_T = split_vfl_X(X_train.T, p)

    # Cuando 'p' es un diccionario, el split puede estar desordenado
    # en relación a las variables, por lo que hay que re-ordenar los pesos
    # de la sol. centralizada (LR) y las filas de Xtest (atributos)
    if type(p) == dict: 
        Xtest = np.vstack(split_vfl_X(Xtest, p))
        for i, w_r in enumerate(w_lr):
            w_r = arrange_weights_by_split(w_r, p)
            w_lr[i] = w_r

    # Preparamos targets 'y' en caso de clasificación OvR
    if mode == 'ovr':
        classes = np.unique(y_train) 
        assert len(classes) == len(np.unique(ytest))    # mismas clases en train que en test
        y_target = np.hstack([(y_train == cls).astype(int) for cls in classes])
        y_test = np.hstack([(ytest == cls).astype(int) for cls in classes])
    else:
        # mode = 'binary'|'regression'
        y_target = y_train if y_train.ndim > 1 else y_train.reshape([-1, 1])
        y_test = ytest if ytest.ndim > 1 else ytest.reshape([-1, 1])

    # Clientes y coordinador
    num_nodos = len(X_lst_T)
    num_outputs = y_target.shape[1]

    # Instanciamos clientes y coordinador 
    Client, Coord = (RidgeClient, RidgeCoordinator) if mode == 'regression' else (LRClient, LRCoordinator)
    lst_clients = [Client(id=i+1, X=X_lst_T[i], lam=lam)
                    for i in range(num_nodos)]
    lst_coords = [Coord(clients=lst_clients, y=y_target[:,n], test_data=(Xtest, y_test[:,n]), w_lr=w_lr[n])
                    for n in range(num_outputs)]

    # Fit
    for i, coord in enumerate(lst_coords):
        print(f"Entrenando {'Ridge' if mode=='regression' else 'LR'} VFL para clase-salida {i+1}/{num_outputs}...")
        coord.fit(max_iter=max_iter, tol=tol, lr=lr, method=method, taylor=taylor)
    
    return lst_coords, lst_clients


def run_grid_search_lr(X_train, y_train, X_test, y_test, param_grid, mode='ovr'):
    """
    Run experiments over a grid-search combination of hyperparameters in a LR/Ridge VFL scenario.
    
    param_grid = {
        # LR/Ridge PARAMS
        'lam': [1e-3],                          # l2 regularization
        'tol': [1e-5, 0]                        # solver tolerance
        'max_iter': [200]                       # solver max iters

        # VFL params
        'p': [[2, 2], [4, 4]],                  # node config
        'lr': [0.5, 0.75, 1.0],                 # learning rates
        'method': ['gd', 'bfgs'],               # modes
        'taylor': [True, False]                 # taylor aprox of loss fn
        ...
    }
    """
    
    # Combinaciones posibles de hiperparams - prod. cartesiano
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Iniciando grid-search con {len(combinations)} combinaciones...")
    
    old_params = None   # Optim para no reentrenar LR|Ridge centralizado con mismos hiperparams
    results = []
    for i, params in enumerate(combinations):
        print(f"--- Experimento Nº {i+1}/{len(combinations)}: {params} ---")
        
        # [LR|Ridge] solución centralizada
        lam         = params['lam']
        tol         = params['tol']
        max_iter    = params['max_iter']
        if (lam, tol, max_iter) != old_params:
            t_lr        = perf_counter()
            mdl_lr      = fit_lr_models(X_train, y_train, mode=mode, lam=lam, tol=tol, max_iter=max_iter)
            print(f"{'LR' if mode != 'regression' else 'Ridge'} \
            centralizado entrenado en: {perf_counter() - t_lr:.2f} (s)")

        # [VFL|Ridge LR]
        params['w_lr']      = mdl_lr[0].w_
        params['test_data'] = (X_test.T, y_test)
        mdls_lrvfl, _       = fit_lrvfl_models(X_train, y_train, mode=mode, **params)

        # Agregamos resultados
        results.append((mdl_lr, mdls_lrvfl))
        old_params = lam, tol, max_iter
    
    # Lista de tuplas con los modelos entrenados LR y VFL LR
    return results


def optimality_gap(loss, min_loss=None, percentage=False):
    lb = min_loss if min_loss is not None else loss.min()   # TODO: coerce? and min_loss < loss.min()
    opt_gap = loss - lb
    if percentage:  opt_gap = opt_gap / lb * 100
    return opt_gap


def product_grid(bcd_grid):
    """Función auxiliar para obtener el producto cartesiano de los hiperparametros
       de una búsqueda en cuadricula (grid-search) para BCD ROLANN que solo produce
       combinaciones con modo 'sequential' que tengan 'lr' a 1."""
    
    # Combinaciones posibles de hiperparams - prod. cartesiano
    par_combinations, seq_combinations = [], []

    # En el modo paralelo tenemos en cuenta los distintos 'lr' dados
    if 'parallel' in bcd_grid['mode_bcd']:
        par_grid = copy.deepcopy(bcd_grid)
        par_grid['mode_bcd'] = ['parallel']
        keys, values = zip(*par_grid.items())
        par_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # En el modo secuencial machacamos el 'lr' fijado a 1.0
    if 'sequential' in bcd_grid['mode_bcd']:
        seq_grid = copy.deepcopy(bcd_grid)
        seq_grid['mode_bcd'] = ['sequential']
        seq_grid['lr'] = [1.]
        keys, values = zip(*seq_grid.items())
        seq_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return par_combinations + seq_combinations


def total_time(df, rolann=True):
    # Valor tiempo preprocesado + suma acumulada de tiempos de iteración

    groups = (df['iter'] == 1).cumsum()
    time   = pd.to_numeric(df.time, errors='coerce')
    total = time.groupby(groups).cumsum()
    if rolann:
        setup = pd.to_numeric(df.time_setup, errors='coerce')
        total += setup.groupby(groups).transform('first')
    
    return total


def merge_df_gridsearch(exps, name=None):
    # Separamos modelos resultado del GridSearch en ROLANN y BCD ROLANN
    lst_rolann, lst_bcd = list(zip(*exps))

    # Obtenemos resultados de cada ejecución
    lst_rolann = list(itertools.chain(*lst_rolann))
    lst_bcd = list(itertools.chain(*lst_bcd))
    df_bcd = [coord.get_results(name=name) for coord in lst_bcd]

    # Concatenamos los resultados en un único dataframe global
    df = pd.concat(df_bcd)
    return df


def compute_extra_results(df, w_optim=None, rolann=True, **kwargs):
    
    # 0. Solución centralizada y su valor para la f. objetivo
    f_act = kwargs.get('f_act', 'logs')
    f_obj = compute_loss_J_w if rolann else (compute_loss_log if f_act == 'logs' else compute_loss_linear)
    loss_w_optim = f_obj(w_optim, **kwargs) if w_optim is not None else None
    # 1. F. objetivo ROLANN en cada iteracion - loss
    df['loss'] = df.w_i.apply(func=f_obj, **kwargs)
    # 2. Suboptimalidad respecto a la solución centralizada - optgap
    df['opt_gap'] = optimality_gap(df.loss, min_loss=loss_w_optim)
    # 3. Coste de comunicación - comm_costs
    df['comm_cost'] = communication_costs(df.iter, df.p, n=len(kwargs['X']), size_bytes=8, out='megabytes')
    # 4. Tiempo total de ejecución hasta cada iteración - time_total
    df['total_time'] = total_time(df, rolann=rolann)

    return df


def balance_train_data(X_train, y_train, random_state=1):
    # Aplica under-sampling aleatorio para igualar las clases en el conjunto de entrenamiento
    np.random.seed(random_state)
    
    # Proporcion de clases
    proportion = y_train.value_counts(normalize=True, ascending=True)
    minority_class = proportion.index[0][0]
    majority_class = proportion.index[1][0]
    print(f"Desequilibrio de clases dataset:\n{proportion}")
    print(f"Clase minoritaria: {minority_class}, clase mayoritaria: {majority_class}")

    # Encontramos los índices de ambas clases (ilocs)
    idx_minority = np.where(y_train == minority_class)[0]
    idx_majority = np.where(y_train == majority_class)[0]
    
    # Contamos los casos de la clase minoritaria
    n_minority = len(idx_minority)
    
    # Seleccionamos aleatoriamente la misma cantidad de la clase mayoritaria
    idx_majority_downsampled = np.random.choice(idx_majority, size=n_minority, replace=False)
    
    # Juntamos los índices y los mezclamos para que no queden ordenados
    idx_balanced = np.concatenate([idx_minority, idx_majority_downsampled])
    np.random.shuffle(idx_balanced)
    
    # Extraemos el nuevo conjunto de balanceado (50/50)
    X_train_balanced = X_train.iloc[idx_balanced]
    y_train_balanced = y_train.iloc[idx_balanced]
    
    return X_train_balanced, y_train_balanced
