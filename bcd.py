#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
bcd.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementación de cliente y coordinador tipo BCD para VFL con ROLANN.
Ojo! Importante!
Hay dos pares de clases (BCDClient, BCDCoordinator) y (BCDClient_alt, BCDCoordinator_alt).
Son equivalentes y producen los mismos resultados.

El 1er par implementa BCD según las ecuaciones descritas en el documento 'VFL_FedHEONN.pdf'.

El 2º hace lo mismo pero reemplazando operaciones matriciales @ y multiplicación de matrices
diagonales por operaciones tipo Hadamard (con numpy broadcasting) con vectores. Además, también
implementa una paralelización real de los clientes (librería joblib) en el bucle BCD 'parallel',
por lo que es la opción preferida para cronometrar tiempos de ejecución realistas.

"""
from math import sqrt
from time import perf_counter
# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, accuracy_score
# Application modules
from activation_functions import load_act_fn, map_labels


# Auxiliary function
def simple_score(f_act, y_true, y_pred):
    if f_act == 'linear':
        # suponemos regresión
        score = root_mean_squared_error(y_true, y_pred)
    else:
        # suponemos clasificación
        _, negpos_tuple = map_labels(f_act)
        UMBRAL = 0.5 * sum(negpos_tuple)
        y_pred = np.where(y_pred <= UMBRAL, 0, 1)
        score = accuracy_score(y_true, y_pred)
    return score

class BCDClient:
    def __init__(self, id, X, f_act='linear', lam=0.001, debug=True):
        # Constructor
        self.id     = id        # ID cliente. (id == 1 -> responsable del 'bias')
        self.X      = X         # Matriz de datos local m x n (variables, muestras)
        self.f_act  = f_act     # Función de activación
        self.lam    = lam       # Regularización l2
        self.debug  = debug     # Debug (historial pesos y gradientes)
    
    def precompute(self):
        if hasattr(self, 'precomputed_') and self.precomputed_: return

        # Bias
        if self.id == 1:
            self.Xp  = np.vstack([np.ones((1, self.X.shape[1])), self.X])
        else:
            self.Xp = self.X

        # Inicializamos pesos
        m, n = self.Xp.shape
        self.w = np.zeros([m, 1])

        # Cargamos f.act.
        _, f_inv, f_der = load_act_fn(self.f_act)
        _, negpos_tuple = map_labels(self.f_act)
        scalar_f_dbar   = 1 if self.f_act == 'linear' else f_der(f_inv(negpos_tuple[0]))
        
        # Matriz factorizacion: H = X @ F
        H = self.Xp * scalar_f_dbar

        # SVD (V ya viene traspuesta)
        U, S, V = np.linalg.svd(H, full_matrices=False)
        Sigma = (S / (S**2 + self.lam))

        # Calculo inversa regularizada:  U @ Sigma @ V @ F
        self.USVF = (U * Sigma) @ (V * scalar_f_dbar)

        print(f"[{self.__class__.__name__} {self.id}] SVD-precompute done!")
        self.precomputed_ = True
        if self.debug:  self.history = {'norm_grd': [], 'norm_res': [], 'norm_inp': []}
        return self.USVF

    def get_net_input(self):
        # Devolvemos z = X^T @ w
        return self.Xp.T @ self.w
    
    def update_weights(self, e_i, lr=0.75):
        # Calculamos actualización del peso multiplicando por el residual enviado por el coord.
        w_new = self.USVF @ e_i
        
        # Aplicamos tasa de aprendizaje (lr, learning rate)
        self.w = (1 - lr) * self.w + lr * w_new

        # Debug
        if self.debug:
            self.history['norm_grd'].append(np.linalg.vector_norm(w_new))
            self.history['norm_res'].append(np.linalg.vector_norm(e_i))
            self.history['norm_inp'].append(np.linalg.vector_norm(self.get_net_input()))

        return self.w
    
    def predict(self, X):
        # Devolvemos y_pred = f(z) = f(X.T w) -> Dimensiones X (m, n)
        if self.id == 1:
            Xp = np.vstack([np.ones((1, self.X.shape[1])), self.X])  # bias
        else:
            Xp = X
        f, _, _ = load_act_fn(self.f_act)
        y_pred = f(Xp.T @ self.w)

        return y_pred
    
    def reset(self, hard=False):
        if hasattr(self, 'w'):
            self.w = np.zeros_like(self.w)
        if hard and hasattr(self, 'precomputed_'):
            self.precomputed_ = False
            if self.debug:  self.history = {'norm_grd': [], 'norm_res': [], 'norm_inp': []}

    def get_results(self, name=None):
        if not self.debug:  return None
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        return df

class BCDCoordinator:
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None):
        # Constructor
        self.clients    = clients       # Lista con clientes
        self.d          = d             # Vector de salidas reales
        self.f_act      = f_act         # Función de activación
        self.test_data  = test_data     # Tupla con (X_test, d_test) para evaluación y gráficas
        self.w_rolann   = w_rolann      # Solución óptima ROLANN para estudio y gráficas

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero'):

        print(f"Iniciando {self.__class__.__name__} ({mode}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'lr': lr, 'mode': mode, 'init': init, 'p': len(self.clients)}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        self.reset(hard=False)  # Reiniciamos pesos de clientes de 'fits' anteriores

        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # [Precomputo]
        t_setup_ini = perf_counter()
        for client in self.clients: client.precompute()
        t_setup_end = perf_counter()
        self.hparams['time_setup'] = t_setup_end - t_setup_ini

        # [BCD]        
    
        # Calculo d_bar
        map_fn, _ = map_labels(self.f_act)      # mapeo de clases en caso de clasif.
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        f, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)  

        # Inicialización de Z y w:
        # todos los w_i empiezan en 0 -> podemos empezar con un Z aleatorio para enmascarar d_bar^-1 en la 1era iter
        Z = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1])

        #1) Inicializamos dict con inputs de los clientes e input global Z (iter=0)
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}

        # Bucle         
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            # Variables para guarda info iter anterior
            Z_iter = Z
            w_iter = w

            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]

                # 2) Calculamos residuo y actualizamos pesos simultaneamente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_i = z_dict[client.id]
                    e_i = d_bar - (Z - z_i)
                    # Actualizamos pesos
                    client.update_weights(e_i, lr)
                
                # Calculamos input neto global
                z_dict = {c.id: c.get_net_input() for c in self.clients}
                Z = sum(z_dict.values())

            else:
                # [Modo secuencial. Estilo Gauss-Seidel]
                
                # 2) Calculamos residuo, actualizamos pesos y reemplazamos nuevo input antes de pasar al siguiente cliente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_old = z_dict[client.id]
                    e_i = d_bar - (Z - z_old)
                    # Actualizamos pesos (ojo, lr = 1)
                    client.update_weights(e_i, lr=1)
                    # Reemplazamos Z con nuevo input tras la actualización de los pesos
                    z_new = client.get_net_input()
                    Z = Z - z_old + z_new       # nuevo input neto global
                    z_dict[client.id] = z_new

            t_iter_end = perf_counter()

            # Información de iteración para estudio
            w = np.concatenate([c.w for c in self.clients])         # solución iter
            diff_w = np.linalg.vector_norm(w - w_iter)              # respecto a iter anterior
            diff_Z = np.linalg.vector_norm(Z - Z_iter)              # respecto a residuos anteriores
            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())

            if self.w_rolann is not None:
                diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
                self.history['diff_w_rolann'].append(diff_w_rolann)
            
            if self.test_data is not None:
                ypred_test = f(X_test.T @ w)
                score_test = simple_score(self.f_act, y_test, ypred_test)
                self.history['score_test'].append(score_test)
            
            ypred_train = f(Z)
            ytrain = self.d.reshape([-1, 1])
            score_train = simple_score(self.f_act, ytrain, ypred_train)
            self.history['score_train'].append(score_train)

            # Log progreso
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1}: Diff Z = {diff_Z:.4f}")
            # Criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break

        # Guardamos pesos finales
        self.w_ = w

    def get_results(self, name=None):
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def get_client_results(self, name=None):
        if len(self.clients) > 0 and all([c.debug for c in self.clients]):
            return pd.concat([c.get_results(f"{name if name else ''}{c.id}") 
                              for c in self.clients])
        else:
            return None

    
    def reset(self, hard=False):
        [client.reset(hard) for client in self.clients];

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en Xtest
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred

#====================================================================================================================
#===[Versiones alternativas del coordinador con operaciones eficientes - para cronometrado y paralelización]===
#====================================================================================================================
from joblib import Parallel, delayed


class BCDCoordinator_par:
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None):
        # Constructor
        self.clients    = clients       # Lista con clientes
        self.d          = d             # Vector de salidas reales
        self.f_act      = f_act         # Función de activación
        self.test_data  = test_data     # Tupla con (X_test, d_test) para evaluación y gráficas
        self.w_rolann   = w_rolann      # Solución óptima ROLANN para estudio y gráficas

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero'):

        print(f"Iniciando BCD ({mode}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'lr': lr, 'mode': mode}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        self.reset(hard=False)    # Reiniciamos pesos de clientes de 'fits' anteriores

        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # Función local wrapper para paralelización
        def update_client_parallel(client, d_bar, Z_global, z_dict, lr):
            z_i = z_dict[client.id]
            e_i = d_bar - (Z_global - z_i)
            client.update_weights(e_i, lr)
            z_i_new = client.get_net_input()
            return client.id, z_i_new
        
        t_setup_ini = perf_counter()
        # [Precomputo]
        Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.precompute)() for client in self.clients
        )
        t_setup_end = perf_counter()
        self.hparams['time_setup'] = t_setup_end - t_setup_ini

        # [BCD]        
    
        # Calculo d_bar
        map_fn, _ = map_labels(self.f_act)      # mapeo de clases en caso de clasif.
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        f, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)  

        # Inicialización de Z y w:
        # todos los w_i empiezan en 0 -> Z aleatorio para enmascarar d_bar^-1 en la 1era iter
        Z = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1])

        #1) Inicializamos dict con inputs de los clientes e input global Z (iter=0)
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}

        # Bucle         
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            # Variables para guarda info iter anterior
            Z_iter = Z
            w_iter = w

            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]

                # 2) Calculamos residuo, actualizamos pesos y calculamos nuevo input simultaneamente
                results = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
                    delayed(update_client_parallel)(c, d_bar, Z, z_dict, lr) 
                    for c in self.clients
                )
                # Calculamos input neto global
                z_dict = dict(results)
                Z = sum(z_dict.values())

            else:
                # [Modo secuencial. Estilo Gauss-Seidel]
                
                # 2) Calculamos residuo, actualizamos pesos y reemplazamos nuevo input antes de pasar al siguiente cliente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_old = z_dict[client.id]
                    e_i = d_bar - (Z - z_old)
                    # Actualizamos pesos (ojo, lr = 1)
                    client.update_weights(e_i, lr=1)
                    # Reemplazamos Z con nuevo input tras la actualización de los pesos
                    z_new = client.get_net_input()
                    Z = Z - z_old + z_new       # nuevo input neto global
                    z_dict[client.id] = z_new
        
            t_iter_end = perf_counter()

            # Información de iteración para estudio
            w = np.concatenate([c.w for c in self.clients]) # solución iter
            diff_w = np.linalg.vector_norm(w - w_iter)      # respecto a iter anterior
            diff_Z = np.linalg.vector_norm(Z - Z_iter)      # respecto a residuos anteriores
            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())

            if self.w_rolann is not None:
                diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
                self.history['diff_w_rolann'].append(diff_w_rolann)
            
            if self.test_data is not None:
                ypred_test = f(X_test.T @ w)
                score_test = simple_score(self.f_act, y_test, ypred_test)
                self.history['score_test'].append(score_test)
            
            ypred_train = f(Z)
            ytrain = self.d.reshape([-1, 1])
            score_train = simple_score(self.f_act, ytrain, ypred_train)
            self.history['score_train'].append(score_train)

            
            # Log progreso
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1}: Diff Z = {diff_Z:.6f}")
            # Criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break
        
        # Guardamos pesos finales
        self.w_ = w

    def get_results(self):
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def reset(self, hard=False):
        [client.reset(hard) for client in self.clients];

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en X
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred

#==================================================================================================================
#===[Versiones alternativas cliente y coordinador con operaciones encriptadas homomorficamente y paralelización]===
#==================================================================================================================
from joblib import Parallel, delayed
import tenseal as ts


class BCDClient_fhe(BCDClient):
    def __init__(self, id, X, f_act='linear', lam=0.001, debug=True, ctx=None):
        # Constructor
        super().__init__(id, X, f_act, lam, debug)
        self.ctx    = ctx       # TenSEAL FHE context
        
    def get_net_input(self):
        assert self.ctx is not None and self.ctx.has_secret_key()
        z = super().get_net_input().flatten()           # vector col. input neto -> aplanamos
        z_fhe = ts.ckks_vector(self.ctx, z.tolist())    # trabaja con listas como vectores
        return z_fhe

    def update_weights(self, e_i_fhe, lr=0.75):
        assert self.ctx is not None and self.ctx.has_secret_key()
        e_i_lst = e_i_fhe.decrypt(self.ctx.secret_key())
        e_i = np.asarray(e_i_lst).reshape(-1, 1)        # lista a vector columna
        return super().update_weights(e_i, lr)


class BCDCoordinator_fhe:
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None, ctx=None):
        # Constructor
        self.clients    = clients       # Lista con clientes
        self.d          = d             # Vector de salidas reales
        self.f_act      = f_act         # Función de activación
        self.test_data  = test_data     # Tupla con (X_test, d_test) para evaluación y gráficas
        self.w_rolann   = w_rolann      # Solución óptima ROLANN para estudio y gráficas
        self.ctx        = ctx           # Contexto TenSEAL FHE público

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero'):

        print(f"Iniciando BCD ({mode}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'lr': lr, 'mode': mode}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        self.reset(hard=False)    # Reiniciamos pesos de clientes de 'fits' anteriores

        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # Función local wrapper para paralelización
        def update_client_parallel(client, d_bar, Z_global, z_dict, lr):
            z_i = z_dict[client.id]
            e_i = d_bar - (Z_global - z_i)
            client.update_weights(e_i, lr)
            z_i_new = client.get_net_input()
            return client.id, z_i_new
        
        t_setup_ini = perf_counter()
        # [Precomputo]
        Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.precompute)() for client in self.clients
        )
        t_setup_end = perf_counter()
        self.hparams['time_setup'] = t_setup_end - t_setup_ini

        # [BCD]        
    
        # Calculo d_bar
        map_fn, _ = map_labels(self.f_act)        # mapeo en caso de clasificación
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        f, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)
        d_bar = d_bar.flatten().tolist()        # para fhe necesitamos listas planas  

        # Inicialización de Z y w: en este caso 'init' ya no es necesario porque Z ya irá encriptado
        Z_iter = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1])

        #1) Inicializamos dict con inputs de los clientes e input global Z (iter=0)
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}
        Z = sum(z_dict.values())
        assert isinstance(Z, ts.CKKSVector)     # Z, e_i y los diferentes z_i son vectores CKKS

        # Bucle         
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            # Variables para guarda info iter anterior
            w_iter = w
            Z_iter = Z_ if iter > 0 else Z_iter

            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]

                # 2) Calculamos residuo, actualizamos pesos y calculamos nuevo input simultaneamente
                results = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
                    delayed(update_client_parallel)(c, d_bar, Z, z_dict, lr) 
                    for c in self.clients
                )
                
                # Calculamos input neto global
                z_dict = dict(results)
                Z = sum(z_dict.values())

            else:
                # [Modo secuencial. Estilo Gauss-Seidel]
                
                # 2) Calculamos residuo, actualizamos pesos y reemplazamos nuevo input antes de pasar al siguiente cliente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_old = z_dict[client.id]
                    e_i = d_bar - (Z - z_old)
                    # Actualizamos pesos (ojo, lr = 1)
                    client.update_weights(e_i, lr=1)
                    # Reemplazamos Z con nuevo input tras la actualización de los pesos
                    z_new = client.get_net_input()
                    Z = Z - z_old + z_new       # nuevo input neto global
                    z_dict[client.id] = z_new
                    

            t_iter_end = perf_counter()

            # Información de iteración para estudio
            w   = np.concatenate([c.w for c in self.clients])       # solución iter
            Z_  = sum([c._get_net_input() for c in self.clients])   # input neto iter

            diff_w = np.linalg.vector_norm(w - w_iter)              # respecto a iter anterior
            diff_Z = np.linalg.vector_norm(Z_ - Z_iter)             # respecto a residuos anteriores

            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())

            if self.w_rolann is not None:
                diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
                self.history['diff_w_rolann'].append(diff_w_rolann)
            
            if self.test_data is not None:
                ypred_test = f(X_test.T @ w)
                score_test = simple_score(self.f_act, y_test, ypred_test)
                self.history['score_test'].append(score_test)
            
            ypred_train = f(Z_) # Ojo respecto a Z_ no 'Z' que es un CKKS vector
            ytrain = self.d.reshape([-1, 1])
            score_train = simple_score(self.f_act, ytrain, ypred_train)
            self.history['score_train'].append(score_train)

            
            # Log progreso
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1}: Diff Z = {diff_Z:.6f}")
            # Criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break
        
        # Guardamos pesos finales
        self.w_ = w

    def get_results(self):
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def reset(self, hard=False):
        [client.reset(hard) for client in self.clients];

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en X
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred

#========================================================================================================
#===[Versiones alternativas cliente y coordinador con operaciones de agregación seguras con SecureAgg]===
#========================================================================================================
from cryptography.hazmat.primitives.asymmetric import x25519

class BCDClient_sec(BCDClient):
    def __init__(self, id, X, f_act='linear', lam=0.001, debug=True, ss=None):
        # Constructor
        super().__init__(id, X, f_act, lam, debug)
        self.ss     = ss        # Shared secrets dictionary
        
    def get_net_input(self):
        nclients = len(self.ss)
        z = super().get_net_input()

        mask = np.zeros_like(z, dtype=float)
        for j in range(nclients):
            if j + 1 == self.id:continue
            ss_j = self.ss[j]
            rng = np.random.default_rng(int.from_bytes(ss_j[:8], "little"))
            m = rng.normal(size=z.shape)
            if self.id < j + 1: mask += m
            else:               mask -= m

        return z + mask

    def generate_dh_keys(self):
        sk = x25519.X25519PrivateKey.generate()
        pk = sk.public_key()
        return sk, pk


class BCDCoordinator_sec:
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None):
        # Constructor
        self.clients    = clients       # Lista con clientes
        self.d          = d             # Vector de salidas reales
        self.f_act      = f_act         # Función de activación
        self.test_data  = test_data     # Tupla con (X_test, d_test) para evaluación y gráficas
        self.w_rolann   = w_rolann      # Solución óptima ROLANN para estudio y gráficas

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero'):

        print(f"Iniciando BCD ({mode}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'lr': lr, 'mode': mode}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        self.reset(hard=False)    # Reiniciamos pesos de clientes de 'fits' anteriores

        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # Función local wrapper para paralelización
        def update_client_parallel(client, d_bar, Z_global, z_dict, lr):
            z_i = z_dict[client.id]
            e_i = d_bar - (Z_global - z_i)
            client.update_weights(e_i, lr)
            z_i_new = client.get_net_input()
            return client.id, z_i_new
        
        t_setup_ini = perf_counter()
        # [Precomputo]
        Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.precompute)() for client in self.clients
        )
        keys = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.generate_dh_keys)() for client in self.clients
        )
        skeys, pkeys = zip(*keys)
        ss = compute_pairwise_dh(skeys, pkeys)
        for i, client in enumerate(self.clients):
            client.ss = ss[i]
        print(f"Setup - DH pairwise keys done!")
        t_setup_end = perf_counter()
        self.hparams['time_setup'] = t_setup_end - t_setup_ini

        # [BCD]        
    
        # Calculo d_bar
        map_fn, _ = map_labels(self.f_act)      # mapeo en caso de clasificación
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        f, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)

        # Inicialización de Z y w
        Z = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1])

        #1) Inicializamos dict con inputs de los clientes e input global Z (iter=0)
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}

        # Bucle         
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            # Variables para guarda info iter anterior
            w_iter = w
            Z_iter = Z

            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]

                # 2) Calculamos residuo, actualizamos pesos y calculamos nuevo input simultaneamente
                results = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
                    delayed(update_client_parallel)(c, d_bar, Z, z_dict, lr) 
                    for c in self.clients
                )
                
                # Calculamos input neto global
                z_dict = dict(results)
                Z = sum(z_dict.values())

            else:
                # [Modo secuencial. Estilo Gauss-Seidel]
                
                # 2) Calculamos residuo, actualizamos pesos y reemplazamos nuevo input antes de pasar al siguiente cliente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_old = z_dict[client.id]
                    e_i = d_bar - (Z - z_old)
                    # Actualizamos pesos (ojo, lr = 1)
                    client.update_weights(e_i, lr=1)
                    # Reemplazamos Z con nuevo input tras la actualización de los pesos
                    z_new = client.get_net_input()
                    Z = Z - z_old + z_new       # nuevo input neto global
                    z_dict[client.id] = z_new
                    

            t_iter_end = perf_counter()

            # Información de iteración para estudio
            w   = np.concatenate([c.w for c in self.clients])       # solución iter
            Z  = sum([c._get_net_input() for c in self.clients])    # input neto iter

            diff_w = np.linalg.vector_norm(w - w_iter)              # respecto a iter anterior
            diff_Z = np.linalg.vector_norm(Z - Z_iter)             # respecto a residuos anteriores

            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())

            if self.w_rolann is not None:
                diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
                self.history['diff_w_rolann'].append(diff_w_rolann)
            
            if self.test_data is not None:
                ypred_test = f(X_test.T @ w)
                score_test = simple_score(self.f_act, y_test, ypred_test)
                self.history['score_test'].append(score_test)
            
            ypred_train = f(Z)
            ytrain = self.d.reshape([-1, 1])
            score_train = simple_score(self.f_act, ytrain, ypred_train)
            self.history['score_train'].append(score_train)

            
            # Log progreso
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1}: Diff Z = {diff_Z:.6f}")
            # Criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break
        
        # Guardamos pesos finales
        self.w_ = w

    def get_results(self):
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def reset(self, hard=False):
        [client.reset(hard) for client in self.clients];

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en X
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred


def compute_pairwise_dh(priv_keys, pub_keys):
    n = len(priv_keys)
    ss = [[None]*n for _ in range(n)]

    for i in range(n):
        for j in range(i+1, n):
            shared = priv_keys[i].exchange(pub_keys[j])
            ss[i][j] = ss[j][i] = shared
    return ss


#================================================================================================================
#===[Versión alternativa de coordinador con heurística simple de relajamiento para el modo paralelo]===
#================================================================================================================

class BCDCoordinator_lrp:
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None):
        # Constructor
        self.clients    = clients       # Lista con clientes
        self.d          = d             # Vector de salidas reales
        self.f_act      = f_act         # Función de activación
        self.test_data  = test_data     # Tupla con (X_test, d_test) para evaluación y gráficas
        self.w_rolann   = w_rolann      # Solución óptima ROLANN para estudio y gráficas

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero', lr_heuristic=True, lr_transient_iters=10):

        print(f"Iniciando BCD ({mode}) con lr={lr}, tol={tol}, max_iter={max_iter}...")
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'mode': mode, 'init': init, 'p': len(self.clients)}
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'score_test': [], 'score_train': [], 'time': [], 'w_i': [], 'lr': []}
        self.reset(hard=False)  # Reiniciamos pesos de clientes de 'fits' anteriores

        # Dejo preparado bias en X_test
        if self.test_data:
            X_test, y_test  = self.test_data
            X_test = np.vstack([np.ones((1, X_test.shape[1])), X_test])

        # [Precomputo]
        t_setup_ini = perf_counter()
        for client in self.clients: client.precompute()
        t_setup_end = perf_counter()
        self.hparams['time_setup'] = t_setup_end - t_setup_ini

        # [BCD]        
    
        # Calculo d_bar
        map_fn, _ = map_labels(self.f_act)      # mapeo en caso de clasificacion
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        f, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)  

        # Inicialización de Z y w:
        # todos los w_i empiezan en 0 -> podemos empezar con un Z aleatorio para enmascarar d_bar^-1 en la 1era iter
        Z = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1])

        #1) Inicializamos dict con inputs de los clientes e input global Z (iter=0)
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}

        # Bucle
        i = 0
        sor_lr = False
        for iter in range(max_iter):
            t_iter_ini = perf_counter()
            # Variables para guarda info iter anterior
            Z_iter = Z
            w_iter = w

            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]                    

                # 2) Calculamos residuo y actualizamos pesos simultaneamente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_i = z_dict[client.id]
                    e_i = d_bar - (Z - z_i)
                    # Actualizamos pesos
                    client.update_weights(e_i, lr)
                
                # Calculamos input neto global
                z_dict = {c.id: c.get_net_input() for c in self.clients}
                Z = sum(z_dict.values())
                
            else:
                # [Modo secuencial. Estilo Gauss-Seidel]
                
                # 2) Calculamos residuo, actualizamos pesos y reemplazamos nuevo input antes de pasar al siguiente cliente
                for client in self.clients:
                    # Residuo excluyendo al propio nodo
                    z_old = z_dict[client.id]
                    e_i = d_bar - (Z - z_old)
                    # Actualizamos pesos (ojo, lr == 1)
                    #client.update_weights(e_i, lr=1)
                    client.update_weights(e_i, lr)
                    # Reemplazamos Z con nuevo input tras la actualización de los pesos
                    z_new = client.get_net_input()
                    Z = Z - z_old + z_new       # nuevo input neto global
                    z_dict[client.id] = z_new
 
            t_iter_end = perf_counter()

            # Información de iteración para estudio
            w = np.concatenate([c.w for c in self.clients])         # solución iter
            diff_w = np.linalg.vector_norm(w - w_iter)              # respecto a iter anterior
            diff_Z = np.linalg.vector_norm(Z - Z_iter)              # respecto a residuos anteriores
            self.history['diff_w'].append(diff_w)
            self.history['diff_Z'].append(diff_Z)
            self.history['time'].append(t_iter_end - t_iter_ini)
            self.history['w_i'].append(w.copy())
            self.history['lr'].append(lr)

            if self.w_rolann is not None:
                diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
                self.history['diff_w_rolann'].append(diff_w_rolann)
            
            if self.test_data is not None:
                ypred_test = f(X_test.T @ w)
                score_test = simple_score(self.f_act, y_test, ypred_test)
                self.history['score_test'].append(score_test)
            
            ypred_train = f(Z)
            ytrain = self.d.reshape([-1, 1])
            score_train = simple_score(self.f_act, ytrain, ypred_train)
            self.history['score_train'].append(score_train)

            # Heuristica convergencia BCD ROLANN
            tol_lr= 0.1
            min_diff_Z = 1e-8 # evitamos dividir ruido
            floor = lambda n, precision=2: np.true_divide(np.floor(n * 10**precision), 10**precision)
            if lr_heuristic:
                if  1 < lr_transient_iters < i and diff_Z_old > min_diff_Z:
                    rho_emp = diff_Z / diff_Z_old
                    if mode == 'parallel' and rho_emp > (1.0 + tol_lr):
                        lr_max = 2 * lr / (rho_emp + 1)
                        lr = floor(0.8 * lr_max)
                        print(f"[{iter+1}] rho_emp ({rho_emp:.3f}) mayor a 1: reducimos lr ({lr_max:.3f})")
                        i = 0   # reiniciamos contador de fase transitoria
                    if mode == 'sequential' and (1.0 - tol_lr) < rho_emp < 1.0 and not sor_lr:
                        lr_max = 2 / (1 + sqrt(1 - rho_emp))
                        lr = floor(lr_max)
                        print(f"[{iter+1}] rho_emp ({rho_emp:.3f}) lento, cercano a 1: aumentamos lr ({lr_max:.3f})")
                        sor_lr = True
                        i = 0   # reiniciamos contador de fase transitoria
                i += 1
                diff_Z_old = diff_Z

            # Log progreso
            if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
                print(f"Iter {iter+1}: Diff Z = {diff_Z:.4f}")
            # Criterio de parada
            if diff_Z < tol:
                print("Convergencia alcanzada.")
                break

        # Guardamos pesos finales
        self.w_ = w

    def get_results(self, name=None):
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def get_client_results(self, name=None):
        if len(self.clients) > 0 and all([c.debug for c in self.clients]):
            return pd.concat([c.get_results(f"{name if name else ''}{c.id}") 
                              for c in self.clients])
        else:
            return None

    
    def reset(self, hard=False):
        [client.reset(hard) for client in self.clients];

    def predict(self, X):
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en Xtest
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred
