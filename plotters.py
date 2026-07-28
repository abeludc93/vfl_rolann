#!/usr/bin/env python3
"""
plotters.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Funciones para la visualización de los resultados de un coordinador BCD_ROLANN.
"""
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
# Application-modules
from helpers import disarrange_weights_by_split

# BASIC PARAMS EXAMPLE FOR `plot_sns()` `kwargs`
SNS_PARAMS = {
    'x': 'iter',        # Eje X
    'y': 'diff_Z',      # Eje Y
    'hue': 'lr',        # Distinto color para cada Learning Rate
    'style': 'mode',    # Distinto trazo (sólido, punteado) para SEQ vs PAR
    'size': 'p',        # Distinto tamaño
    'linewidth': 1.5,   # etc
    'palette': 'bright',
}

def plot_sns(df, plot_log=True, tol=None, title="", **kwargs):
    """
    From a wide tabular dataframe `df` with BCD_ROLANN coordinator results, graphs
    a seaborn lineplot with given settings.
    :param      df (:pd.DataFrame)
                plot_log, logarithmic y-ax  (:bool)
                tol, horizontal dashed line to show off tolerance stop criteria (:float)
                title (string)
                kwargs for extra settings, see SNS_PARAMS (:dict)
    Multiple subplots can be achived by specifying {'ax': ax} in kwargs.
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
    legend = kwargs.get('legend', True)
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if ax is None:
        plt.tight_layout() 
        plt.show()

def plot_weights_comparison(w1, w2, label_w1='ROLANN', label_w2='BCD', title="COMPARACION PESOS"):
    """
    Graphs a stem plot showing the differences between `w1` and `w2` weight vectors for comparison purposes.
    :param      w1, w2 column vectors (:np.array)
                label_w1, label_w2, title (:string)
    """

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


#====================
#===[AUX PLOTTERS]===
#====================

def plot_correlation_map(X, title="Matriz de correlación entre variables"):
    # Plots a heatmap with of the correlation matrix for a given dataset
    # training data (X) of dimensions (n, m).

    # Calculamos matriz de correlación
    corr_matrix = X.corr()
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title(title)
    plt.xlabel("Variable")
    plt.ylabel("Variable")
    plt.show()

def plot_score_global(df, labels, ylabel="Accuracy", title=None):
    # Graphs two subplots showing both OvR and global train/test scores.

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
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_im_split(img, lst_idx, title=None):
    # Given a 2D data sample and a list of index masks, plots a colored representation
    # showing how the VFL split partitions the image data.
    # Useful in datasets such as MNIST, etc. to represent a given VFL scenario.

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
    # Plots a 2D representation of the weights provided in `w_optim` and `w_vfl`,
    # after reshaping the column vectors according to `imgsize`dimensions.
    # Useful in datasets such as MNIST to illustrate the 2D form of a BCD_ROLANN solution.

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
