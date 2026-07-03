#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
plotters.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Funciones para la visualización de los resultados
"""
# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def use_log(arr, min_orders=2):
    values = np.asarray(arr, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values <= 0):
        return False
    orders = np.log10(values.max()) - np.log10(values.min())
    return orders >= min_orders


def plot_diff(df, title="BCD ROLANN DIFFS"):
    fig, ax = plt.subplots()
    fig.suptitle(title)

    tol = df.tol.values[0]
    mode = df['mode'].values[0]
    lr = df.lr.values[0]
    max_iter = df.max_iter.values[0]

    iters = df.index + 1
    # Eje 1: semilogaritmico para diff_w, diff_Z y tol
    l1, = ax.semilogy(iters, df.diff_w, label=r'$\Vert \Delta \mathbf{w} \Vert$', marker='o')
    l2, = ax.semilogy(iters, df.diff_Z, label=r'$\Vert \Delta \mathbf{z} \Vert$', marker='v')
    l3  = ax.axhline(tol, label=r'$\epsilon$', linestyle='--', color='grey', linewidth='1.5')

    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Diferencia valores')
    ax.set_title(f"(mode = {mode}, lr = {lr}, max_iter = {max_iter})")
    ax.grid()

    # Eje 2: semilogaritmico o lineal dependiendo del rango
    if use_log(df.diff_w_rolann):
        l4, = ax.semilogy(iters, df.diff_w_rolann, label=r'$\Vert \Delta \mathbf{w}_{ROLANN} \Vert$', marker='d')
    else:
        ax2 = ax.twinx()
        ax2.set_ylabel('Diferencia valores', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        l4, = ax2.plot(iters, df.diff_w_rolann, label=r'$\Vert \Delta \mathbf{w}_{ROLANN} \Vert$', marker='d', color='tab:red')

    # Leyenda
    handles = [l1, l2, l3, l4]
    ax.legend(handles, [h.get_label() for h in handles])
    
    plt.show(block=False)


def plot_score(df, title="BCD ROLANN SCORES"):
    fig, ax = plt.subplots()
    fig.suptitle(title)

    mode = df['mode'].values[0]
    lr = df.lr.values[0]
    max_iter = df.max_iter.values[0]

    iters = df.index + 1
    ax.plot(iters, df.score_train, label='score train', marker='o')
    ax.plot(iters, df.score_test, label='score test', marker='d')

    ax.set_xlabel('Iteraciones')
    ax.set_ylabel('Puntuación')
    ax.set_title(f"(mode= {mode}, lr = {lr}, max_iter = {max_iter})")
    ax.legend()
    ax.grid()
    plt.show(block=False)


def plot_mode_diffs(dfs, title="BCD ROLANN DIFFS"):   
    # Preparamos formato -> TODO debería de hacerse fuera de esta función
    if isinstance(dfs, list):
        dfs_lst = []
        for df in dfs:
            if "iter" not in df.columns:
                df = df.reset_index(names="iter")
            dfs_lst.append(df)
        # Hay que concatenarlas
        dfs = pd.concat(dfs_lst, ignore_index=False)
    elif isinstance(dfs, pd.DataFrame):
        if "iter" not in dfs.columns:
            dfs = dfs.reset_index(names="iter")
    else:
        raise TypeError(f"Arg. `dfs` debe ser una lista o un dataframe con el formato correcto!: {type(dfs)}")

    
    df_long = dfs.melt(
        id_vars=['mode', 'iter'],                           # identificador tipo groupby
        value_vars=['diff_w', 'diff_Z', 'diff_w_rolann'],   # columnas con valores
        var_name='Valor',                                   # Nombre para la leyenda de colores
        value_name='Diferencia valores',                    # Nombre para el eje Y
        ignore_index=False
    )
    # Renombramos columnas y valores
    df_long = df_long.rename(columns={'mode':'Modo', 'iter': 'Iteración'})
    df_long['Modo'] = df_long['Modo'].replace({'parallel': 'Paralelo (Jacobi)', 'sequential': 'Secuencial (Gauss-Seidel)'})
    df_long['Valor'] = df_long['Valor'].replace({'diff_w': r'$\Vert \Delta \mathbf{w} \Vert$',
                                                 'diff_Z': r'$\Vert \Delta \mathbf{z} \Vert$',
                                                 'diff_w_rolann': r'$\Vert \Delta \mathbf{w}_{ROLANN} \Vert$'})
    
    tol = dfs.tol.values[0]
    lr = dfs.lr.values[0]
    max_iter = dfs.max_iter.values[0]

    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.axhline(tol, label=r'Tolerancia ($\epsilon$)', linestyle='--', color='grey', linewidth='1.5')
    sns.lineplot(
        data=df_long,
        x='Iteración',
        y='Diferencia valores',
        hue='Valor',            # colores para distinguir value_vars
        style='Modo',           # estilos de linea para distinguir modos
        markers=True,           
        dashes=True,
        ax=ax
    )
    
    ax.set_yscale('log')
    ax.set_xlabel("Iteraciones")
    ax.set_title(f"(lr = {lr}, max_iter = {max_iter})")
    ax.grid()
    
    plt.show(block=False)


def plot_mode_scores(dfs, title="BCD ROLANN MODE SCORES"):
    # Preparamos formato
    if isinstance(dfs, list):
        dfs_lst = []
        for df in dfs:
            if "iter" not in df.columns:
                df = df.reset_index(names="iter")
            dfs_lst.append(df)
        # Hay que concatenarlas
        dfs = pd.concat(dfs_lst, ignore_index=False)
    elif isinstance(dfs, pd.DataFrame):
        if "iter" not in dfs.columns:
            dfs = dfs.reset_index(names="iter")
    else:
        raise TypeError(f"Arg. `dfs` debe ser una lista o un dataframe con el formato correcto!: {type(dfs)}")
    
    df_long = dfs.melt(
        id_vars=['mode', 'iter'],                   # identificador tipo groupby
        value_vars=['score_train', 'score_test'],   # columnas con valores
        var_name='Métrica',                         # Nombre para la leyenda de colores
        value_name='Puntuación',                    # Nombre para el eje Y
        ignore_index=False
    )
    # Renombramos columnas y valores
    df_long = df_long.rename(columns={'mode':'Modo', 'iter': 'Iteración'})
    df_long['Modo'] = df_long['Modo'].replace({'parallel': 'Paralelo (Jacobi)', 'sequential': 'Secuencial (Gauss-Seidel)'})
    df_long['Métrica'] = df_long['Métrica'].replace({'score_train': 'score train', 'score_test': 'score test'})
    
    lr = dfs.lr.values[0]
    max_iter = dfs.max_iter.values[0]

    fig, ax = plt.subplots()
    fig.suptitle(title)
    sns.lineplot(
        data=df_long,
        x='Iteración',
        y='Puntuación',
        hue='Métrica',          # colores para distinguir value_vars
        style='Modo',           # estilos de linea para distinguir modos
        markers=True,           
        dashes=True,
        ax=ax
    )

    ax.set_xlabel("Iteraciones")
    ax.set_title(f"(lr = {lr}, max_iter = {max_iter})")
    ax.grid()
    
    plt.show(block=False)


def plot_weights_comparison(w1, w2, label_w1='ROLANN', label_w2='BCD', title="COMPARACION PESOS"):
    # Aplanamos por si vienen como vectores columna (m, 1)
    w_c = w1.flatten()
    w_b = w2.flatten()
    
    # Eje X: índices de las variables
    x = np.arange(len(w_c))
    
    # Configuración del gráfico
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(title)

    # Axes 1: pesos
    ax1.stem(x, w_c, linefmt='r-', markerfmt='ro', label=rf'$\mathbf{{w}}_{{{label_w1}}}$', basefmt='k')
    ax1.stem(x, w_b, linefmt='b-.', markerfmt='bx', label=rf'$\mathbf{{w}}_{{{label_w2}}}$', basefmt='k')
    ax1.set_ylabel(r'Valor pesos ($w_i$)')
    ax1.set_title(f"Solución {label_w1} vs. {label_w2}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Axes 2: residuos
    diff = w_c - w_b
    rmse_weights = np.sqrt(np.mean(diff**2))
    ax2.stem(x, diff, linefmt='k-', markerfmt='k.', basefmt='k')
    ax2.set_ylabel('Diferencia')
    ax2.set_xlabel('Índice de la variable correspondiente')
    ax2.set_title(f'Residuo entre pesos (RMSE: {rmse_weights:.4e})')
    ax2.grid(alpha=0.3)
    
    # Modificamos xticks para indicar donde está el bias
    xticks_arr, xticks_labels = plt.xticks()
    idx = np.argwhere(xticks_arr == 0).item()
    xticks_labels[idx].set_text('Bias')
    xticks_labels[0].set_text('')
    xticks_labels[-1].set_text('')
    plt.xticks(xticks_arr, xticks_labels)
    
    plt.show(block=False)


def plot_correlation_map(X, title="Matriz de correlación entre variables"):
    # X es la matriz de datos en formato DataFrame y dimensiones (n, m)
    # Calculamos matriz de correlación
    corr_matrix = X.corr()
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title(title)
    plt.xlabel("Variable")
    plt.ylabel("Variable")
    plt.show()


def plot_score_global(df, labels, ylabel="Accuracy"):

    x = df['iter']
    _, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    
    # Scores por salida/clase
    test_cols  = df.filter(like=f'score_test_').columns
    train_cols = df.filter(like=f'score_train_').columns
    for i, (test_col, train_col) in enumerate(zip(test_cols, train_cols)):
        axes[0].plot(x, df[test_col],  alpha=0.5, linewidth=1, label=f"{labels[i]}")
        axes[1].plot(x, df[train_col], alpha=0.5, linewidth=1, label=f"{labels[i]}")
    # Score global
    axes[0].plot(x, df['score_test'],  color="black", linewidth=3, label="Global")
    axes[1].plot(x, df['score_train'], color="black", linewidth=3, label="Global")

    # Titulos y ejes
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel)
    axes[0].set_title("Score test")
    axes[1].set_title("Score train")
    plt.legend(loc='center', bbox_to_anchor=(1.25, 0.5), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_communication_costs(df, ylabel='Accuracy', out='Bytes', show_iter=True):
    
    _, ax = plt.subplots(figsize=(6,4))

    # Plot normal
    x = df.comm
    ax.plot(x, df.score_test,  color="C0", lw=2.5, label="Test")
    ax.plot(x, df.score_train, color="C1", lw=2.5, label="Train")

    ax.set_xlabel(f"Communication Costs ({out})")
    ax.set_ylabel(ylabel)
    ax.legend()

    if show_iter:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_indices = np.linspace(0, len(df) - 1, 5, dtype=int)

        target_comms = df.iloc[tick_indices]["comm"]
        target_iters = df.iloc[tick_indices]["iter"]

        # Asignamos ticks al eje superior
        ax2.set_xticks(target_comms)
        ax2.set_xticklabels(target_iters)
        ax2.set_xlabel("Iteración")

    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_time_scores(df, ylabel='Accuracy'):
    _, ax = plt.subplots(figsize=(6,4))

    # Plot normal
    t_setup = df.time_setup.values[0]
    x = t_setup + df.time.cumsum()
    ax.plot(x, df.score_test,  color="C0", lw=2.5, label="Test")
    ax.plot(x, df.score_train, color="C1", lw=2.5, label="Train")
    ax.axvline(t_setup, label='Clients setup', color='grey', linestyle='--')
    ax.set_xlabel(f"Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.grid()
    plt.title("Acc. vs Time")
    plt.show()


def plot_sns(df, plot_log=True, tol=None, title="", **kwargs):
    """
    kwargs ejemplo:  
        {
        'x': 'iter',        # Eje X
        'y': 'diff_Z',       # Eje Y
        'hue': 'lr',        # Distinto color para cada Learning Rate
        'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
        'size': 'p',
        'linewidth': 1.5,
        ...
        }
    """
    # Figura
    ax = kwargs.get('ax', None)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    xcol, ycol = kwargs['x'], kwargs['y']

    # Seaborn
    sns.lineplot(data=df, **kwargs|{'ax':ax})

    # línea de tolerancia si se solicita
    if tol: 
        ax.axhline(tol, color='grey', linestyle='--', label=f'tol = {tol}')

    # Escala logarítmica
    if plot_log:
        ax.set_yscale('log')

    # Estética y etiquetas
    ax.set_title(f'({ycol}) {title}')
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, which="both", ls="--", alpha=0.5) # Grid visible incluso en escala log
    
    # Leyenda fuera del gráfico
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if ax is None:
        plt.tight_layout() 
        plt.show()


def plot_im_split(img, lst_idx, title=None):
    
    # Máscara
    n = len(lst_idx)                # nº de nodos
    mask = np.full_like(img, -1)    # mascara para colorear img según los nodos
    for i, idx in enumerate(lst_idx):
        mask.flat[idx] = i
    
    # Gráfica
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(img, cmap="grey")    #img original
    axes[1].imshow(img, cmap="grey")

    # Colores según nº nodos y leyenda
    cmap    = plt.colormaps["tab20"]
    overlay = axes[1].imshow(mask, cmap=cmap, alpha=0.5)
    colors  = cmap(np.linspace(0, 1, n))
    patches = [mpatches.Patch(color=colors[i], label=rf"$p_{{{i+1}}}$") for i in range(n)]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()


def plot_im_weights(w_optim, w_vfl, p_dict, imgsize=(28, 28), title="OPTIM vs VFL"):
    
    from helpers import disarrange_weights_by_split
    w_vfl_sorted = disarrange_weights_by_split(w_vfl, p_dict)

    bias_optim = w_optim[0]
    bias_vfl   = w_vfl[0]

    # Gráfica
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(w_optim[1:].reshape(imgsize), cmap="viridis", aspect="equal")
    axes[1].imshow(w_vfl_sorted[1:].reshape(imgsize), cmap="viridis", aspect="equal")

    axes[0].set_title(f"Bias: {bias_optim}")
    axes[1].set_title(f"Bias: {bias_vfl}")
    fig.suptitle(title)
    fig.tight_layout()
