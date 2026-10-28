#!/usr/bin/env python3
"""
rolann.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementación básica del entrenamiento, cómputo de pesos óptimos 
(solución) y predicción del algoritmo ROLANN original.
No implementamos el aprendizaje incremental (no se guardan M, U o S).
Realiza regresión o clasificación en función de la 'f_act' dada.
"""
# Third-party libraries
import numpy as np
# Application modules
from activation_functions import load_act_fn, map_labels


def compute_loss_J_w(w, X, y, lam, f_act):
    # Cálculo de la función objetivo J(w) de ROLANN
    
    _, f_inv, f_der = load_act_fn(f_act)    # Cargamos f. act.
    f_map, _ = map_labels(f_act)               # Cargamos mapper y -> d
    Xp = np.vstack([np.ones(len(X)), X.T])  # Xp traspuesto con bias incluido
    d = f_map(y.reshape(-1,1))              # Mapeamos para obtener target
    w = w.reshape(-1,1)                     # Aseguramos vector columna

    # Obtenemos \bar{d}, F y Z
    d_bar = f_inv(d).flatten()
    f_dbar = f_der(d_bar).flatten()
    z = w.T @ Xp
    e_bar = d_bar - z
    
    # Multiplicación por F (Haadamard para evitar construcción de matriz diag.)
    F_e = f_dbar * e_bar
        
    # 1er sumando J(w): pérdida
    loss = 0.5 * np.sum(F_e ** 2)
    
    # 2o sumando J(w): regularización
    reg_loss = 0.5 * lam * np.sum(w ** 2)

    return loss + reg_loss


class ROLANN:
    def __init__(self, f_act='logs', lam=.001):
        # Constructor
        self.f_act              = f_act                 # Función de activación
        self.lam                = lam                   # Regularización

    def fit(self, X, y):
        # Cargamos f. act.
        _, f_inv, f_der = load_act_fn(self.f_act)
        
        # Preparamos 'X' -> Xp traspuesto con bias
        Xp = np.vstack([np.ones(len(X)), X.T])

        # Preparamos 'y' -> d, dbar, f_dbar
        f_map, _ = map_labels(self.f_act)
        d = f_map(y)
        d_bar = f_inv(d).flatten()
        f_dbar = f_der(d_bar).flatten()
        
        # SVD y matriz de factorización
        H = Xp * f_dbar
        U, s, _ = np.linalg.svd(H, full_matrices=False)     # SVD sobre matriz de factorizacion
        S = np.diag(s)
        
        # Solución
        M = Xp @ (f_dbar * f_dbar * d_bar)                  # Calculo de M como XFFd_bar
        SS_lamI = S @ S + self.lam * np.eye(len(S))         # Construimos (S^2 + \lambda I)
        w = U @ (np.diag(1/SS_lamI.diagonal()) @ (U.T @ M)) # Pesos optimos

        self.w_ = w.reshape([-1, 1])
        return self.w_

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones(len(X)), X.T])  # Introducimos bias en X
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])          # convertimos en vector columna

        return y_pred
