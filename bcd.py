#!/usr/bin/env python3
"""
bcd.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implementación de cliente y coordinador para VFL con ROLANN.
Ojo! Importante!
Hay cuatro pares de clases:
    1. (BCDClient,      BCDCoordinator)
    2. (BCDClient_par,  BCDCoordinator_par)
    3. (BCDClient_fhe,  BCDCoordinator_fhe)
    4. (BCDClient_sec,  BCDCoordinator_sec)

Son equivalentes y producen los mismos resultados.
(1) implementa BCD_ROLANN según la propuesta original descrita en el documento VFL_ROLANN.pdf.
(2) añade una paralelización real de los clientes (`joblib`) en el modo 'parallel'.
(3) añade protocolo FHE en las comunicaciones entre clientes y coordinador.
(4) añade protocolo SecureAggregation en las comunicaciones entre clientes y coordinador.

Como los bloques fundamentales del algoritmo son compartidos,
(1) forma una base de la que heredan el resto de variaciones.

[HERENCIAS Cliente/Coordinador]
(1) -> (2)  -> (3)
            -> (4)
"""
# Standard libraries
from joblib import Parallel, delayed
from time import perf_counter
# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, accuracy_score
import tenseal as ts
from cryptography.hazmat.primitives.asymmetric import x25519
# Application modules
from activation_functions import load_act_fn, map_labels


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
        # Reinicia los pesos, borra historial y activa nuevo `precompute`
        if hasattr(self, 'w'):
            self.w = np.zeros_like(self.w)
        if hard and hasattr(self, 'precomputed_'):
            self.precomputed_ = False
            if self.debug:  self.history = {'norm_grd': [], 'norm_res': [], 'norm_inp': []}

    def get_results(self, name=None):
        # Devuelve historial con la norma de gradiente, residuos e inputs netos
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

    def fit(self, max_iter=20, tol=1e-3, lr=0.75, mode='parallel', init='zero', heur=False, **kwargs):
        print(f"Iniciando {self.__class__.__name__} ({mode})...")
        
        # [RESET/SETUP]
        self._reset(hard=False)     # Reiniciamos pesos de clientes de 'fits' anteriores
        self._prepare_test_data()   # Si hay conj. de test metemos bias y guardamos

        # [PRECOMPUTO]
        t_setup = perf_counter()
        self._precompute_clients()  # Calculo P_i = USVF_i en clientes

        # [INIT]
        # Creamos historial y volcamos datos estáticos
        self._init_history(max_iter, tol, mode, init, heur, (perf_counter()-t_setup))
        # Inicializamos variables BCD_ROLANN: $\bar{d}, \mathbf{Z}, \mathbf{w}$
        d_bar, Z, z_dict = self._init_params(init)
        w = self._init_weights(init)

        # [LOOP]
        for iter in range(max_iter):
            # Variables auxiliares: tiempo e vars de iter anterior
            t_iter, Z_iter, w_iter = perf_counter(), Z, w
            
            if mode == 'parallel':
                # [Modo paralelo. Estilo Jacobi]
                Z, z_dict = self._parallel_iteration(d_bar, Z, z_dict, lr)
            else:
                # [Modo paralelo. Estilo Jacobi]
                Z, z_dict = self._sequential_iteration(d_bar, Z, z_dict, lr)

            # Información de la iteración (diff_w, diff_Z, ...) y métricas (scores) para historial
            w, diff_Z = self._update_history(w_iter, Z_iter, Z, lr, (perf_counter()-t_iter))

            # Criterio de parada y log de progreso
            if self._check_stop(iter, max_iter, diff_Z, tol):
                break
                
            # Heuristica convergencia (factor amortiguación 'lr')
            if heur and mode == 'parallel':
                lr = self._check_heuristic(lr, iter, diff_Z, **kwargs)
            
        # Guardamos pesos finales
        self.w_ = w

    def get_results(self, name=None):
        # Devuelve el historial del proceso de entrenamiento (dataframe)
        df = pd.DataFrame.from_dict(self.history, orient='index').transpose()
        df['iter'] = df.index + 1
        if name:     df['name'] = name
        for k, v in self.hparams.items():
            df[k] = v
        return df
    
    def get_client_results(self, name=None):
        # Devuelve el historial de todos los clientes (en un solo dataframe)
        if len(self.clients) > 0 and all([c.debug for c in self.clients]):
            return pd.concat([c.get_results(f"{name if name else ''}{c.id}") 
                              for c in self.clients])
        else:
            return None

    def predict(self, X):
        # Predecir con los pesos finales (w_)
        f, _ , _ = load_act_fn(self.f_act)
        Xp = np.vstack([np.ones([1, X.shape[1]]), X])  # Introducimos bias en Xtest
        z = self.w_.T @ Xp
        y_pred = f(z).reshape([-1, 1])

        return y_pred
    
    def simple_score(self, true, pred):
        # Métrica simple para historial con RMSE/accuracy
        if self.f_act == 'linear':
            # suponemos regresión
            score = root_mean_squared_error(true, pred)
        else:
            # suponemos clasificación
            _, negpos_tuple = map_labels(self.f_act)
            UMBRAL = 0.5 * sum(negpos_tuple)
            y_pred = np.where(pred <= UMBRAL, 0, 1)
            score = accuracy_score(true, y_pred)

        return score

    # [Funciones protegidas para fit(): sobreescribir en clases hijas para modificar su comportamiento]
    def _reset(self, hard=False):
        # Borramos vars auxiliares coord. y reseteamos clientes
        if hasattr(self, '_iter_lr'):       delattr(self, '_iter_lr')
        if hasattr(self, '_diff_Z_old'):    delattr(self, '_diff_Z_old')
        if hasattr(self, '_Xtest'):         delattr(self, '_Xtest')
        if hasattr(self, '_ytest'):         delattr(self, '_ytest')
        for client in self.clients:         client.reset(hard)

    def _precompute_clients(self):
        # Impl. por defecto: lanzamos `precompute` de los clientes
        for client in self.clients:         client.precompute()
    
    def _get_Z_for_metrics(self, Z):
        # Impl. por defecto: Z es un array de datos en claro
        return Z
    
    def _init_history(self, max_iter, tol, mode, init, heur, t_setup):
        # Historial para metricas y monitorizacion de convergencia 
        self.history = {'diff_w': [], 'diff_Z': [], 'diff_w_rolann': [], 'lr': [],
                        'score_test': [], 'score_train': [], 'time': [], 'w_i': []}
        # Info estática sobre la ejecución (siempre fijas)
        num_p, fact = len(self.clients), self.f_act
        self.hparams = {'max_iter': max_iter, 'tol': tol, 'mode': mode, 'init': init,
                        'p': num_p, 'f_act': fact, 'heur': heur, 'time_setup': t_setup}

    def _init_params(self, init):
        # Variables bucle BCD_ROLANN:
        # necesitamos `d_bar`, input neto de los clientes `z_dict` y global `Z` 

        # Carga de f.act. para el cálculo de d_bar a partir de las salidas reales
        map_fn, _ = map_labels(self.f_act)      # mapeo de clases en caso de clasif.
        d = map_fn(self.d).reshape([-1, 1])     # reshape para vector columna
        _, f_inv, _ = load_act_fn(self.f_act)
        d_bar = f_inv(d)

        # Z global: un init aleatorio enmascara el $d_bar$ en la 1era iter
        Z = np.zeros_like(d) if init == 'zero' else np.random.rand(d.shape)

        # Dict Z:   {cliente: input neto} 
        z_dict = {c.id: c.get_net_input() for c in self.clients} # {id: z_i}

        return d_bar, Z, z_dict
    
    def _init_weights(self, init):
        # Inicializa con ceros o aleatoriamente el vector col. de pesos finales
        m = sum([len(c.w) for c in self.clients])
        w = np.zeros([m, 1]) if init == 'zero' else np.random.rand(m, 1)
        return w

    def _prepare_test_data(self):
        # Introducimos bias de conjunto de datos 
        if self.test_data is not None:
            X_test, y_test  = self.test_data
            self._Xtest = np.vstack([np.ones((1, X_test.shape[1])), X_test])
            self._ytest = y_test
        
    def _parallel_iteration(self, d_bar, Z, z_dict, lr):
        # [MODO JACOBI]: lr 'variable'

        # RESUMEN: Calculamos residuo y actualizamos pesos en todos simultaneamente
        for client in self.clients:
            # Residuo excluyendo al propio nodo
            z_i = z_dict[client.id]
            e_i = d_bar - (Z - z_i)
            # Actualizamos pesos
            client.update_weights(e_i, lr)
        
        # Actualizamos input neto global
        z_dict = {c.id: c.get_net_input() for c in self.clients}
        Z = sum(z_dict.values())
    
        return Z, z_dict
    
    def _sequential_iteration(self, d_bar, Z, z_dict, lr=1):
        # [MODO GAUSS-SEIDEL]: lr 'fijo'

        # RESUMEN: Calculamos residuo, actualizamos pesos y modificamos 
        # input global antes de pasar al siguiente cliente
        for client in self.clients:
            # Residuo excluyendo al propio nodo
            z_old = z_dict[client.id]
            e_i = d_bar - (Z - z_old)
            # Actualizamos pesos (ojo, lr=1)
            client.update_weights(e_i, lr)
            # Reemplazamos Z con nuevo input tras la actualización de los pesos
            z_new = client.get_net_input()
            Z = Z - z_old + z_new       # nuevo input neto global
            z_dict[client.id] = z_new
    
        return Z, z_dict

    def _update_history(self, w_iter, Z_iter, Z, lr, t_iter):
        # Guarda la información relevante de la iteracion:

        w = np.concatenate([c.w for c in self.clients])         # pesos iter
        diff_w = np.linalg.vector_norm(w - w_iter)              # respecto a pesos anteriores
        Z_metric = self._get_Z_for_metrics(Z)                   # residuo iter
        diff_Z = np.linalg.vector_norm(Z_metric - Z_iter)       # respecto a residuos anteriores

        # (1) Info convergencia
        self.history['diff_w'].append(diff_w)
        self.history['diff_Z'].append(diff_Z)
        self.history['w_i'].append(w.copy())
        self.history['lr'].append(lr)
        if self.w_rolann is not None:
            diff_w_rolann = np.linalg.vector_norm(w - self.w_rolann)    # respecto a w_rolann
            self.history['diff_w_rolann'].append(diff_w_rolann)
        
        # (2) Métricas
        f, _ , _ = load_act_fn(self.f_act)
        if self.test_data is not None:  # Test
            ypred_test = f(self._Xtest.T @ w)
            score_test = self.simple_score(self._ytest, ypred_test)
            self.history['score_test'].append(score_test)
        ypred_train = f(Z)              # Train
        ytrain = self.d.reshape([-1, 1])
        score_train = self.simple_score(ytrain, ypred_train)
        self.history['score_train'].append(score_train)

        # (3) Tiempo transcurrido
        self.history['time'].append(t_iter)

        return w, diff_Z

    def _check_stop(self, iter, max_iter, diff_Z, tol):
        # Progreso del bucle: cada cuarta parte del proceso se lanza un log
        if ((iter+1) % round(max_iter / 4)) == 0 or (iter + 1) == max_iter or diff_Z < tol:
            print(f"Iter {iter+1}: Diff Z = {diff_Z:.4f}")
        
        # Criterio de parada: si diff_Z cumple una tolerancia dada (pág.21 `VFL_ROLANN.pdf`)
        if (stop := diff_Z < tol):
            print("Convergencia alcanzada.")
        
        return stop
    
    def _check_heuristic(self, lr, iter, diff_Z, **kwargs):
        # Implementación de heurística para garantizar convergencia modo 'parallel'
        # (Sección 5.3 en VFL_ROLANN.pdf) -> descartamos heurística en modo 'sequential'

        # Contadores auxiliares para fase transitoria y ruido numerico
        if not hasattr(self, '_iter_lr'):       self._iter_lr = 0
        if not hasattr(self, '_diff_Z_old'):    self._diff_Z_old = 0

        # Función auxiliar para redondear hacia abajo con un nº determinado de decimales
        floor = lambda n, precision=2: np.true_divide(np.floor(n * 10**precision), 10**precision)

        # Heuristica convergencia BCD ROLANN paralelo
        tol_lr      = kwargs.get('tol_lr', .05)         # evita activar la heuristica por ruido en la aprox de $\rho$
        min_diff_Z  = kwargs.get('min_diff_Z', 1e-8)    # idem por ruido numérico debido a divisiones pequeñas
        transient   = kwargs.get('transient', 15)       # duracion de la fase transitoria
        beta_lr     = kwargs.get('beta_lr', .8)         # factor "conservador" a aplicar sobre estimación `lr_max`

        # Si acabó la fase transitoria y si diff_Z aún no es demasiado pequeño 
        if  (1 < transient < self._iter_lr) and (self._diff_Z_old > min_diff_Z):
            # Estimación radio espectral
            rho_emp = diff_Z / self._diff_Z_old
            # Si detectamos divergencia
            if rho_emp > (1.0 + tol_lr):
                # Estimación factor relajación óptimo
                lr_max = 2 * lr / (rho_emp + 1)
                lr = floor(beta_lr * lr_max)
                print(f"[{iter+1}] rho_emp ({rho_emp:.3f}) > 1: reducimos lr ({lr_max:.3f})")
                self._iter_lr = 0   # reiniciamos contador de fase transitoria
        
        # Actualizamos contadores
        self._iter_lr += 1
        self._diff_Z_old = diff_Z

        # o devuelve el mismo si no se activó la heurística o nueva estimación en caso contrario
        return lr   

#=============================================================
#===[Versiones del coordinador con paralelización efectiva]===
#=============================================================
class BCDClient_par(BCDClient): pass        # es exactamente igual al original


class BCDCoordinator_par(BCDCoordinator):   # solo sobreescribe aquellos "bloques" que hace diferente
    def _precompute_clients(self):
        # Paralelización del precomputo de los clientes
        Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.precompute)() for client in self.clients
        )
    
    def _parallel_iteration(self, d_bar, Z, z_dict, lr):
        # Paralelización real en la iteracion del modo paralelo
        results = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(_update_client_parallel)(c, d_bar, Z, z_dict, lr) 
            for c in self.clients
        )
        z_dict = dict(results)
        Z = sum(z_dict.values())

        return Z, z_dict

# Función auxiliar definida "fuera" (equivalente a iteración paralela)
# para evitar problemas de serialización con `joblib`
def _update_client_parallel(client, d_bar, Z_global, z_dict, lr):
    z_i = z_dict[client.id]
    e_i = d_bar - (Z_global - z_i)
    client.update_weights(e_i, lr)
    return client.id, client.get_net_input()

#======================================================================
#===[Versiones de cliente/coordinador bajo encriptación homomórfica]===
#======================================================================
class BCDClient_fhe(BCDClient):
    def __init__(self, id, X, f_act='linear', lam=0.001, debug=True, ctx=None):
        # Constructor
        super().__init__(id, X, f_act, lam, debug)
        self.ctx    = ctx       # TenSEAL FHE context
        
    def get_net_input(self):
        # Encriptamos y devolvemos el input neto del cliente
        assert self.ctx is not None and self.ctx.has_secret_key()
        z = super().get_net_input().flatten()           # vector col. input neto -> aplanamos
        z_fhe = ts.ckks_vector(self.ctx, z.tolist())    # trabaja con listas como vectores
        return z_fhe

    def update_weights(self, e_i_fhe, lr=0.75):
        # Desencriptamos residuos del coordinador y actualizamos pesos
        assert self.ctx is not None and self.ctx.has_secret_key()
        e_i_lst = e_i_fhe.decrypt(self.ctx.secret_key())
        e_i = np.asarray(e_i_lst).reshape(-1, 1)        # lista a vector columna
        return super().update_weights(e_i, lr)


class BCDCoordinator_fhe(BCDCoordinator_par):   # solo sobreescrible los "bloques" que hace diferente
    def __init__(self, clients, d, f_act, test_data=None, w_rolann=None, ctx=None):
        # Constructor
        super().__init__(clients, d, f_act, test_data, w_rolann)
        self.ctx = ctx          # TenSEAL FHE context

    def _init_params(self, init):
        # Lógica base
        d_bar, _, z_dict = super()._init_params(init)

        # Adaptamos a FHE: necesitamos listas planas
        d_bar = d_bar.flatten().tolist()

        # Adaptamos a FHE: Z necesita ser un CKKSVector
        Z = sum(z_dict.values())
        assert isinstance(Z, ts.CKKSVector) # Z, e_i, z_i son vectores CKKS
        
        return d_bar, Z, z_dict

    def _get_Z_for_metrics(self, Z):
        # En el bucle de BCD_ROLANN `Z` es un CKKSVector pero para el historial y
        # debug obtenemos este dato en claro llamando directamente a la superclase
        return sum([super(BCDClient_fhe, c).get_net_input() for c in self.clients])
    
    def set_ctx(self, ctx):
        # Setter de contexto tenSeal
        self.ctx = ctx

#===============================================================
#===[Versiones de cliente/coordinador bajo SecureAggregation]===
#===============================================================
class BCDClient_sec(BCDClient):
    def __init__(self, id, X, f_act='linear', lam=0.001, debug=True, ss=None):
        # Constructor
        super().__init__(id, X, f_act, lam, debug)
        self.ss     = ss        # Shared secrets dictionary
        
    def get_net_input(self):
        # Obtenemos el input y lo camuflamos sumando y restando
        # máscaras aleatorias con las semillas guardadas en `ss`
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
        # Fn aux: genera un par (sk, pk) a compartir con el resto de clientes
        sk = x25519.X25519PrivateKey.generate()
        pk = sk.public_key()
        return sk, pk


class BCDCoordinator_sec(BCDCoordinator_par):
    # Fijarse que el constructor se omite ya que tiene los mismos params,
    # es decir, no necesita almanecear el dictionario de semillas `ss`

    def _precompute_clients(self):
        # Precomputo paralelo
        super()._precompute_clients()
        
        # Generamos diccionario de intercambio Diffie-Hellman
        keys = Parallel(n_jobs=-1, require='sharedmem', prefer='threads')(
            delayed(client.generate_dh_keys)() for client in self.clients
        )
        # Guardamos claves en clientes
        skeys, pkeys = zip(*keys)
        ss = compute_pairwise_dh(skeys, pkeys)
        for i, client in enumerate(self.clients):
            client.ss = ss[i]
        print(f"Setup - DH pairwise keys done!")

# Función auxiliar para SecureAgg: intercambio claves
def compute_pairwise_dh(priv_keys, pub_keys):
    n = len(priv_keys)
    ss = [[None]*n for _ in range(n)]

    for i in range(n):
        for j in range(i+1, n):
            shared = priv_keys[i].exchange(pub_keys[j])
            ss[i][j] = ss[j][i] = shared
    return ss
