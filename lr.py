#!/usr/bin/env python3
"""
lr.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementación básica de un modelo tipo LogisticRegression|Ridge con reg.
L2 para el escenario federado VFL en tareas de clasificación o regresión.
"""
# Third-party libraries
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.metrics import accuracy_score, root_mean_squared_error
import numpy as np


#===============================================
#===[Funciones de activación y aprox. Taylor]===
#===============================================
def linear(z):
    return z

def logsig(z):
    # Sigmoide
    z_clip = np.clip(z, -250, 250)
    return 1 / (1 + (np.exp(-z_clip)))

def logsig_taylor(z):
    # Aprox. Taylor a sigmoide de grado 1: f(z) = 1/2 + 1/4 z alrededor de z_0 = 0
    return 0.5 + 0.25 * z

def log_likelihood(ytrue, z):
    # Negative log-likelihood: cross-entropy
    epsilon = 1e-15
    ypred   = np.clip(logsig(z), epsilon, 1 - epsilon)
    loss    = -np.mean(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))
    return loss
    
def log_likelihood_taylor(ytrue, z):
    # Aproximación de Taylor de grado 2 a negative log-likelihood: cross-entropy
    loss = np.mean(np.log(2) + (0.5 - ytrue) * z + 0.125 * (z ** 2))
    return loss

#=====================================
#===[Funciones de pérdida/objetivo]===
#=====================================
def compute_loss_log(w, X, y, lam, **kwargs):
    # Adapts `val_loss_log` to the same signature as `ROLANN.compute_loss_J_w` for easier comparisons.

    # Proxy para calcular valor de la f.objetivo a semejanza de compute_loss_J_w de ROLANN.py (mismos args.)
    Xp = np.vstack([np.ones(len(X)), X.T])
    w = w.reshape(-1, 1)
    z = w.T @ Xp
    val_loss = val_loss_log(y, z, w, lam, **kwargs)

    return val_loss

def val_loss_log(y, z, w, lam, taylor=False, reg_bias=False):
    # Computes the value of the objective function used in LogisticRegression models.
    y_f, z_f = y.flatten(), z.flatten()
    loss    = log_likelihood_taylor(y_f, z_f) if taylor else log_likelihood(y_f, z_f)
    loss_l2 = np.sum(w**2) if reg_bias else np.sum(w[1:]**2)
    
    return loss + 0.5 * lam * loss_l2

def compute_loss_linear(w, X, y, lam, **kwargs):
    # Adapts `val_loss_linear` to the same signature as `ROLANN.compute_loss_J_w` for easier comparisons.

    # Proxy para calcular valor de la f.objetivo a semejanza de compute_loss_J_w de ROLANN.py (mismos args.)
    Xp = np.vstack([np.ones(len(X)), X.T])
    w = w.reshape(-1, 1)
    z = w.T @ Xp
    val_loss = val_loss_linear(y, z, w, lam, **kwargs)

    return val_loss

def val_loss_linear(y, z, w, lam, reg_bias=False, **kwargs):
    # Computes the value of the objective function used in Ridge models.
    y_f, z_f = y.flatten(), z.flatten()
    loss     = np.mean((y_f - z_f)**2)
    loss_l2  = np.sum(w**2) if reg_bias else np.sum(w[1:]**2)
    
    return 0.5 * (loss + lam * loss_l2)


#===================================================
#===[VFL_LR: implementación similar a BCD_ROLANN]===
#===================================================

class LRClient:
    def __init__(self, id, X, lam=0.001, debug=True):
        self.id     = id        
        self.X      = X         # Matriz de datos local m x n (variables, muestras)
        self.lam    = lam       # Regularización L2
        self.f_act  = 'logs'    # Función de activación (solo 'logs')
        self.debug  = debug     # Debug (historial)

    def precompute(self):
        # There is no need for a strict `precompute` in VFL_LR, but we adapt this
        # implementation to mirror BCD_ROLANN and use it to setup different variables

        if hasattr(self, 'precomputed_') and self.precomputed_: return
        # Bias sobre matriz de datos
        if self.id == 1:
            self.Xp  = np.vstack([np.ones((1, self.X.shape[1])), self.X])
        else:
            self.Xp = self.X
        # Variables y muestras de matriz de datos
        self.m, self.n = self.Xp.shape
        # Pesos inicializados a cero
        self.w_ = np.zeros([self.m, 1])
        # Ajustamos regularización a VFL: lam_vfl = lam / N (muestras)
        self.lam /= self.n
        self.precomputed_ = True
        if self.debug:  self.history= {'norm_grd': [], 'norm_res': [], 'norm_inp': []}

    def get_net_input(self):
        # Devuelve z_i = X.T @ w (Dimensiones: n x 1)
        return self.Xp.T @ self.w_

    def update_weights(self, e, lr=0.75, method='gd', reg_bias=False):
        # Calcula el gradiente local y actualiza los pesos
        # e: error despues de f.act. (y_pred - y_true) recibido del coordinador con dim. (n, 1)
        # lr: learning_rate
        
        # Gradiente de la función de pérdida: Cross-Entropy con regularización L2
        grad = (self.Xp @ e) / self.n       # promedio entre no. muestras
        reg_term = self.lam * self.w_
        if self.id == 1 and not reg_bias:
            reg_term[0] = 0.0   # no regularizo bias
        grad += reg_term 
        
        # Descenso de gradiente (GD: Gradient Descent)
        if method == 'gd':
            self.w_ = self.w_ - lr * grad
        # Quasi-Newton (BFGS: Broyden–Fletcher–Goldfarb–Shanno)
        else:
            if hasattr(self, 'H_'):
                # Diferencia de gradientes e incrementos
                y = grad - self.g_old_
                s = self.w_ - self.w_old_
                # Actualización de la matriz Hessiana Inversa
                ys = np.dot(y.T, s)[0, 0]
                if ys > 1e-10:
                    rho = 1.0 / ys
                    I = np.eye(self.m)
                    term1 = I - rho * (s @ y.T)
                    term2 = I - rho * (y @ s.T)
                    self.H_ = term1 @ self.H_ @ term2 + rho * (s @ s.T)
            else:
                self.H_ = np.eye(self.m)

            # Guardar estado para la siguiente iteración
            self.w_old_ = self.w_.copy()
            self.g_old_ = grad.copy()

            # Calcular dirección de descenso y actualizar peso según 'lr'
            grad = -self.H @ grad
            self.w_ = self.w_ + lr * grad
       
        # Debug
        if self.debug:
            self.history['norm_grd'].append(np.linalg.vector_norm(grad))
            self.history['norm_res'].append(np.linalg.vector_norm(e))
            self.history['norm_inp'].append(np.linalg.vector_norm(self.get_net_input()))

        return self.w_
    
    def predict(self, X):
        # Bias
        if self.id == 1:
            Xp = np.vstack([np.ones((1, X.shape[1])), X])
        else:
            Xp = X
        # Devolvemos y_pred = f(z) = f(X.T w) -> Dimensiones X (m, n)
        f_act  = logsig if self.f_act == 'logs' else linear
        y_pred = f_act(Xp.T @ self.w_)
        return y_pred
    
    def reset(self, hard=False):
        # Reset weights and auxiliary variables for a new `fit()` process
        if hasattr(self, 'w'):
            self.w_ = np.zeros_like(self.w_)
        if hasattr(self, 'H_'):
            del self.H_; del self.w_old_; del self.g_old_
        if hard and hasattr(self, 'precomputed_'):
            self.precomputed_ = False
            if self.debug:  self.history = {'norm_grd': [], 'norm_res': [], 'norm_inp': []}

    def get_results(self, name=None):
        # Returns client history as a DataFrame
        if not self.debug:  return None
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        return df


class LRCoordinator:
    def __init__(self, clients, y, test_data=None, w_lr=None):
        # Constructor
        self.clients    = clients               # list of clients
        self.y          = y.reshape([-1, 1])    # real/desired outputs (no multi-output, 1D data)
        self.test_data  = test_data             # test dataset
        self.w_lr       = w_lr                  # solution from centralized LogisticRegression
        self.f_act      = 'logs'                # act. function ('logs' only)

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, method='gd', taylor=False, init='zero', reg_bias=False):
        # This method resembles the BCD_ROLANN coordinator `fit` base logic for the VFL_LR models.

        # [INIT HISTORY]
        print(f"Iniciando VFL {type(self)} (modo={method}, taylor={taylor}, reg_bias={reg_bias}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'lr': lr, 'mode': method, 'taylor': taylor, 'p': len(self.clients), 'init': init}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_lr': [], 'loss':[], 'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        # [RESET]
        self.reset(hard=False)  # Reiniciamos pesos de clientes de 'fits' anteriores
        
        # [PREPARE TEST DATA]
        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # F. activación y otras func. auxiliares
        if self.f_act == 'logs':
            f_act       = logsig_taylor if taylor else logsig
            map_pred    = lambda y: np.where(y >= 0.5, 1, 0)
            metric_fn   = accuracy_score
        else:
            f_act       = linear
            map_pred    = lambda y: y
            metric_fn   = root_mean_squared_error

        # [PRECOMPUTE]: inexistente, solo asignamos matrices de datos locales X_i e inicializamos pesos w_i
        for client in self.clients: client.precompute()
        
        # [INIT PARAMS]
        m = sum([len(c.w_) for c in self.clients])
        w = np.zeros([m, 1])
        Z = np.zeros_like(self.y, dtype=float) if init == 'zero' else np.random.rand(self.y.shape)
        lam = self.clients[0].lam   # Para la pérdida correspondiente a la reg. L2 - asumimos igual en todos los clientes

        # [LOOP]
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            
            # Variables para guardar info iter anterior
            w_iter = w
            Z_iter = Z
            
            # [ITERATION]
            # 1. Predicción y error despues de la f. act.
            y_pred = f_act(Z_iter)
            
            # 2. Derivada de la función de perdida (Binary cross-entropy) = y_pred - y
            e = y_pred - self.y 
            
            # 3. Enviar el error a los clientes para que actualicen sus pesos
            for c in self.clients: # TODO: paralelize via `joblib`
                c.update_weights(e, lr, method, reg_bias)

            # 4. Input neto nuevo
            Z = sum(c.get_net_input() for c in self.clients)

            t_iter_end = perf_counter()

            # [UPDATE HISTORY]
            # Información de iteración para estudio - monitorización convergencia y criterio de parada
            w       = np.concatenate([c.w_ for c in self.clients])  # pesos alcanzados en esta iter
            diff_w  = np.linalg.vector_norm(w - w_iter)             # respecto a iter anterior
            diff_Z  = np.linalg.vector_norm(Z - Z_iter)             # respecto a residuos anteriores
            if self.f_act == 'logs':                                # valor de la función de perdida
                loss = val_loss_log(self.y, Z, w, lam, taylor, reg_bias)
            else:
                loss = val_loss_linear(self.y, Z, w, lam, reg_bias)
            
            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())
            self.history['loss'].append(loss)

            if self.w_lr is not None:
                self.w_lr = self.w_lr.reshape(-1, 1)                # aseguramos vector columna
                diff_w_lr = np.linalg.vector_norm(w - self.w_lr)    # respecto a w_lr
                self.history['diff_w_lr'].append(diff_w_lr)

            # Información de iteración para estudio - monitorización métricas alcanzadas
            mapped_pred = map_pred(y_pred)
            score_train = metric_fn(self.y, mapped_pred)
            score_test = np.nan
            if self.test_data:
                y_pred_test = f_act(X_test.T @ w)
                mapped_pred_test = map_pred(y_pred_test)
                score_test = metric_fn(y_test, mapped_pred_test)

            self.history['score_train'].append(score_train)
            self.history['score_test'].append(score_test)
            
            # [LOG PROGESS]
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1} | Loss: {loss:.4f} | diff_Z: {diff_Z:.4f}")
            
            # [CHECK STOP]: criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break
        
        # Guardamos pesos finales
        self.w_ = w
                
    def get_results(self, name=None):
        # Return coordinator history as a DataFrame
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def get_client_results(self, name=None):
        # Returns all client history as a single DataFrame
        if len(self.clients) > 0 and all([c.debug for c in self.clients]):
            return pd.concat([c.get_results(f"{name if name else ''}{c.id}") 
                              for c in self.clients])
        else:
            return None

    def reset(self, hard=False):
        # Resets clients
        [client.reset(hard) for client in self.clients];

    def predict(self, X, taylor=False):
        # Basic predict
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias
        z = self.w_.T @ Xp
        f_act = linear if self.f_act == 'linear' else (logsig_taylor if taylor else logsig)
        y_pred = f_act(z).reshape([-1, 1])

        return y_pred


#==================================================
#===[RIDGE: solo es necesario cambiar la f.act.]===
#==================================================

class RidgeClient(LRClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f_act = 'linear'

class RidgeCoordinator(LRCoordinator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f_act = 'linear'
