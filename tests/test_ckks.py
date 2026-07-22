#!/usr/bin/env python3
"""
tests/test_ckks.py
~~~~~~~~~~~~
Script that both 1) checks for a correct installation of the `tenseal` library
and 2) simulates the operations under BCD_ROLANN in order to benchmark the comms
and computational costs w/ an increasing number of clients and vector dimension.
~~~~~~~~~~~~
(INFO)

Librería: `pip install tenseal` -> implementa FHE con esquema CKSS (sumas, restas, multiplicaciones)
Pruebas simulando implementación en bucle iterativo BCD ROLANN.
Pasos:

(Cliente)
    1. Recibe su residuo e_i [n x 1] del coordinador y lo desencripta.
    2. Actualiza sus pesos w_i y calcula su nuevo input z_i.
    3. Encripta su input z_i [n x 1] y lo envía de nuevo al coordinador.

(Coordinador)
    1. Recibe los inputs z_i de todos los clientes (p vectores [n x 1]).
    2. Los agrega para calcular el input global Z.
    3. Calcula el residuo e_i de cada cliente.
        3.1 Calcula E = f_dbar - Z
        3.2 Hace e_i = E + z_i para cada los clientes.

El cliente trabaja con datos en claro pero necesita encriptar/desencriptar al enviar/recibir datos.
El coordinador puede trabajar enteramente con datos cifrados (incluido el f_dbar).
Esto implica un clave privada compartida por todos los clientes y una publica conocida solo por el coordinador.

(TESTS)
Simularemos 3 procesos:
+ Cliente:      1) Encriptar
                2) Desencriptar
+ Coordinador:  3) Agregacion y calculo residuos
"""
#%% [IMPORTS]
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tenseal as ts
import pickle

#%% [FHE example: check-up of tenseal library]

# Generacion claves
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

# Encriptar y desencriptar
lst_numeros = [3.14, 1, -1.2e-13]
enc_lst_numeros = ts.ckks_vector(context, lst_numeros)
dec_lst_numeros = enc_lst_numeros.decrypt()

assert np.allclose(dec_lst_numeros, lst_numeros)

#%% [Benchmarks]

# Simulates BCD_ROLANN fit process w/ FHE
def benchmark_fhe(N, p, repeticiones=3):
    """ Simula los procesos de cifrado llevados a cabo por cliente y coordinador usando
        un esquema de FHE CKKS para UNA iteración del bucle BCD ROLANN paralelo."""
    global context, sk

    print(f"Benchmark FHE CKKS: dimension vector ({N}), clientes ({p})")

    # Resultados: cliente encripta/desencripta un vector N y coordinador agrega p*N vectores
    tiempos = {'encriptar': [], 'agregar': [], 'desencriptar': []}

    for _ in range(repeticiones):
        t_enc, t_coord, t_dec = 0, 0, 0 # Para promedios
       
        # Datos aleatorios para 'p' clientes
        fd_bar_scalar = 0.5
        inputs_z_i = [np.random.rand(N) for _ in range(p)]  # Vectores tamaño N

        # [Cliente]

        # 1) Encriptar z_i
        ini_enc = time.perf_counter()
        # Encriptamos cada elemento del vector de dim. N del primer cliente z_0
        z_0_enc = ts.ckks_vector(context, inputs_z_i[0])   
        fin_enc = time.perf_counter() - ini_enc
        t_enc += fin_enc

        assert np.allclose(z_0_enc.decrypt(sk), inputs_z_i[0])

        # [Coordinador]

        # Encriptamos los inputs z_i de los 'p' clientes (esto no lo cronometramos)
        z_clientes_enc = [ts.ckks_vector(context, vec) for vec in inputs_z_i]

        # 2) Agregacion y calculo residuos
        ini_coord = time.perf_counter()
        
        # Sumamos los inputs para todos los 'p' clientes para obtener Z
        Z = sum(z_clientes_enc)

        # Los residuos se calculan para cada cliente
        for i in range(p):  # en escenario real habría un diccionario
            e_i_enc = fd_bar_scalar - Z + z_clientes_enc[i]

        fin_coord = time.perf_counter() - ini_coord
        t_coord += fin_coord
        
        assert np.allclose(Z.decrypt(sk), sum(inputs_z_i))

        # [Cliente]    
        # 3) Desencriptar e_i_enc
        ini_dec = time.perf_counter()

        e_i = e_i_enc.decrypt(sk)
        fin_dec = time.perf_counter() - ini_dec
        t_dec += fin_dec

    # Promedios
    tiempos['encriptar'].append(t_enc/repeticiones)
    tiempos['agregar'].append(t_coord/repeticiones)
    tiempos['desencriptar'].append(t_dec/repeticiones)

    # Resultados
    df = pd.DataFrame(tiempos)
    df['N'] = N
    df['p'] = p

    return df

# Simulates BCD_ROLANN comm costs w/ FHE
def benchmark_commcosts(N_lst):
    global context
    df_dict = {'N':[], 'plain': [], 'encrypted': [], 'overhead': []}
    for N in N_lst:
        vec         = np.random.rand(N)             # np,array
        vec_enc     = ts.ckks_vector(context, vec)  # encrypted array
        plain_size  = len(pickle.dumps(vec))        # normal serialization
        enc_size    = len(vec_enc.serialize())      # encripted serialization
        overhead    = enc_size / plain_size

        df_dict['plain'].append(plain_size/1024)
        df_dict['encrypted'].append(enc_size/1024)
        df_dict['overhead'].append(overhead)
        df_dict['N'].append(N)

    return pd.DataFrame.from_dict(df_dict)

# Plotters
def plot_tiempos_vector(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.N, df.encriptar, 'o-', label=r'Cliente: encriptar $z_i$')
    plt.plot(df.N, df.desencriptar, '^-', label=r'Cliente: desencriptar $e_i$')
    plt.plot(df.N, df.agregar, 's-', label=r'Coord: agregación $\mathbf{Z}$ y cálculo $e_i$')
    plt.xlabel('Dimensión del vector (N)')
    plt.ylabel('Tiempo por iteración (s)')
    plt.title(f"Benchmark FHE CKSS (tenSeal) - coste computacional - {df.p.values[0]} clientes")
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.show()

def plot_tiempos_clientes(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.p, df.agregar, 's-', label=r'Coord: agregación $\mathbf{Z}$ y cálculo $e_i$')
    plt.xlabel('Número de clientes (p)')
    plt.ylabel('Tiempo por iteración (s)')
    plt.title(f"Benchmark FHE CKSS (tenSeal) - coste computacional - Vector dim. {df.N.values[0]}")
    plt.legend()
    plt.grid()
    plt.show()

def plot_commcosts(df):
    plt.figure(figsize=(10, 6))
    plt.loglog(df.N, df.plain, 'o-', label=r'Tamaño vector en claro $z_i$')
    plt.loglog(df.N, df.encrypted, '^-', label=r'Tamaño vector encriptado $z_i$')
    plt.loglog(df.N, df.overhead, 's-', label=r'Overhead debido a FHE')
    plt.xlabel('Dimensión del vector (N)')
    plt.ylabel('Tamaño serializado (KB)')
    plt.title(f"Benchmark FHE - coste comunicaciones según tamaño vector" \
                 f"\nTenSeal poly modulus degree: {poly}")
    plt.legend()
    plt.grid()
    plt.show()

#%% [Setting up TenSEAL context]
print("Generando par de claves FHE TenSEAL...")
poly = 8192
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

# %% [Checking visibility of private-public keys]
print("Is the context private?", ("Yes" if context.is_private() else "No"))
print("Is the context public?", ("Yes" if context.is_public() else "No"))
sk = context.secret_key()
ctx = context.serialize(save_public_key=True, save_secret_key=True)
print(f"Context size: {len(ctx)/1024:.2f} KB")

# Context will drop the secret-key at this point
context.make_context_public()
print("Secret-key dropped")
print("Is the context private?", ("Yes" if context.is_private() else "No"))
print("Is the context public?", ("Yes" if context.is_public() else "No"))

ctx = context.serialize(save_public_key=True, save_secret_key=False)
print(f"Context size: {len(ctx)/1024:.2f} KB")

#%% [Exp1. Increasing vector size]
N_lst = [100, 500, 1000, 2000]
p = 3
df_lst = []
for N in N_lst:
    df = benchmark_fhe(N, p, repeticiones=2)
    df_lst.append(df)
df_vectores = pd.concat(df_lst, ignore_index=True)
plot_tiempos_vector(df_vectores)

#%% [Exp 2. Increasing number of clients]
p_lst = [3, 10, 50, 100]
df_lst = []
N = 25
for p in p_lst:
    df = benchmark_fhe(N, p, repeticiones=1)
    df_lst.append(df)
df_clientes = pd.concat(df_lst, ignore_index=True)
plot_tiempos_clientes(df_clientes)


#%% [Exp 3. Overhead in comm.costs due to increasing vector size]
N_lst=[10, 100, 1000, 10000, 100000]
df_commcost = benchmark_commcosts(N_lst)
plot_commcosts(df_commcost)

# %% CONCLUSIONES
"""
1.  Coste computacional es mayor en la suma de los inputs netos de los clientes en el coordinador
    que en la encriptació/desencriptación que se hace de ellos y los residuos en los clientes.
2.  Coste computacional del protocolo FHE no depende tanto del tamaño de los vectores N si no del
    nº clientes p. El tiempo por iteración en el coordinador (agregación y cómputo de residuos)
    crece linealmente con el p.
3.  El peso añadido en la serialización de los vectores encriptados por el uso de FHE (overhead)
    decrece con la dimensión del vector N. Por ejemplo: no suponen más KB encriptar un np.array
    de 10 elementos que de 1000. Este overhead parece asentarse en 10x.
"""
#%% OTHER CONSIDERATIONS
"""
1.  A partially homomorphic encryption protocol was considered using a Paillier scheme that enables
    sums of encrypted scalars and multiplications by non-encrypted scalars, thinking it would be 
    computationally less expensive than a full-on FHE scheme such as CKKS.
2.  Nonetheless, the Paillier implementations found in libraries such as `phe` do NOT support
    efficient vectorized operations, relying instead on element-by-element procedures that waste
    away any meaningful prior advantages over FHE.
3.  In older tags of this project, an equivalent benchmark was conducted for a simulated PHE scenario
    that supports these considerations (search for `test_phe.py`)
"""
