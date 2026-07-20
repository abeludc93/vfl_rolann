#!/usr/bin/env python3
"""
rolann.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementación básica del entrenamiento, computo de pesos óptimos 
(solución) y predicción del algoritmo ROLANN original.
No implementamos aprendizaje incremental (no se guardan M, U o S).
Esta pensado únicamente como benchmark para los resultados del BCD.
Realiza regresión o clasificación automáticamente en función de 'f_act'.
"""
# Third-party libraries
import numpy as np
# Application modules
from activation_functions import load_act_fn, map_labels


def compute_loss_J_w(w, X, y, lam, f_act):
    """
    Calculo de la función objetivo J(w) de ROLANN
    """
    _, f_inv, f_der = load_act_fn(f_act)    # Cargamos f. act.
    f_map = map_labels(f_act)               # Cargamos mapper y -> d
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

    def _fit(self, X, y):
        # Cargamos f. act.
        _, f_inv, f_der = load_act_fn(self.f_act)
        
        # Preparamos 'X' -> Xp traspuesto con bias
        Xp = np.vstack([np.ones(len(X)), X.T])

        # Preparamos 'y' -> d, dbar, f_dbar
        if self.f_act == 'linear':
            # suponemos regresión
            d = y
        else:
            # suponemos clasificación
            f_map = map_labels(self.f_act)
            d = f_map(y)
        d_bar = f_inv(d)
        f_dbar = f_der(d_bar).flatten()
        
        # SVD y términos sistema
        F = np.diag(f_dbar)
        H = Xp @ F                                      
        U, s, _ = np.linalg.svd(H, full_matrices=False)     # SVD sobre matriz de factorizacion
        S = np.diag(s)
        
        # Solución
        M = Xp @ (F @ (F.T @ d_bar))                        # Calculo de M como XFFd_bar
        SS_lamI = S @ S + self.lam * np.eye(len(S))         # Construimos (S^2 + \lambda I)
        w = U @ (np.diag(1/SS_lamI.diagonal()) @ (U.T @ M)) # Pesos optimos

        self.w_ = w.reshape([-1, 1])                        # Aseguramos vector columna pesos
        return self.w_
    
    def fit(self, X, y):
        # Cargamos f. act.
        _, f_inv, f_der = load_act_fn(self.f_act)
        
        # Preparamos 'X' -> Xp traspuesto con bias
        Xp = np.vstack([np.ones(len(X)), X.T])

        # Preparamos 'y' -> d, dbar, f_dbar
        if self.f_act == 'linear':
            # suponemos regresión
            d = y
        else:
            # suponemos clasificación
            f_map = map_labels(self.f_act)
            d = f_map(y)
        d_bar = f_inv(d).flatten()
        f_dbar = f_der(d_bar).flatten()
        
        # SVD y términos sistema
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
