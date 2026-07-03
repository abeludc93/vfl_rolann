#%% HE Aditivo - Paillier
"""
Pruebas simulando implementación en bucle iterativo BCD ROLANN.

Librería: `pip install phe` -> implementa HE Aditivo esquema Paillier (permite sumas, restas y mult. por escalar)

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

El cliente trabaja con datos en claro pero necesita encriptar/desencriptar.
El coordinador puede trabajar enteramente con datos cifrados (incluido el f_dbar).
Esto implica un clave privada compartida por todos los clientes y una publica conocida solo por el coordinador.

[TESTS]

Simularemos 3 procesos:

+ Cliente:  1) Encriptar) 2) Desencriptar
+ Coordinador: 3) Agregacion y calculo residuos

Para tamaños de vector 'n' crecientes (muestras del dataset):

    Esperamos que el overhead en los clientes sea cada vez mayor pero que sufra menos que el coordinador
    ya que es más complejo la desencriptación/encriptación que la suma/resta una vez cifrados.

Para un numero creciente de nodos 'p':

    Esperamos un leve incremento del tiempo de calculo del coordinador. Los clientes no se verán afectados
    ya que "trabajan" en paralelo.

[OTRAS CONSIDERACIONES]

La librería 'phe' trabaja con escalares. Aun que soporta operaciones con numpy sobre listas de numeros cifrados
tipo np.sum(lst_encrypted) NO permite encriptar un nd.array directamente.

La longitud por defecto del par de claves publico/privada son 2048 bits (con dos modulos de 1024 bits cada una).
El cifrado resultante de encriptar un escalar es de 1024 bits -> 128 bytes.

Obviamos el estudio del overhead o costes de comunicación en comparación con 'float' (8bytes) o 'nd.array' de floats.
"""
#%% Ejemplo de libreria PHE
#==============================
#==============================

# Generacion claves
from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)  # defecto

# Encriptar y desencriptar

lst_numeros = [3.14, 1, -1.2e-13]
enc_lst_numeros = [public_key.encrypt(x) for x in lst_numeros]
dec_lst_numeros = [private_key.decrypt(x) for x in enc_lst_numeros]

assert lst_numeros == dec_lst_numeros

# Operaciones cifradas
a, b, c = enc_lst_numeros

t1 = a - 3.14                               # cifrado menos escalar en claro
t2 = a + b + c                              # agregación de cifrados
t3 = b * 10                                 # cifrado por escalar
t4 = np.mean(enc_lst_numeros)               # operaciones con numpy sobre lista de cifrados
t5 = np.dot(enc_lst_numeros, [2, 2, 2])     # idem


res_lst_test = [private_key.decrypt(x) for x in (t1, t2, t3, t4, t5)]
t1_res, t2_res, t3_res, t4_res, t5_res = res_lst_test

assert t1_res == 0                          # debería de ser 0
assert t2_res == sum(lst_numeros)           # la suma de los numeros de la lista
assert t3_res == lst_numeros[1] * 10        # 1 * 10
assert t4_res == sum(lst_numeros) / 3       # el promedio
assert t5_res == sum([i*2 for i in lst_numeros])    # producto escalar


#%% [TESTS CIFRADO BCD ROLANN]
#==============================
#==============================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phe import paillier


def benchmark_phe(N, p, repeticiones=3):
    """ Simula los procesos de cifrado llevados a cabo por cliente y coordinador usando
        un esquema de HE aditivo para UNA iteración del bucle BCD ROLANN paralelo."""
    global private_key, public_key

    print(f"Benchmark HE Paillier: dimension vector ({N}), clientes ({p})")

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
        z_0_enc = [public_key.encrypt(x) for x in inputs_z_i[0]]   
        fin_enc = time.perf_counter() - ini_enc
        t_enc += fin_enc

        # [Coordinador]

        # Encriptamos los inputs z_i de los 'p' clientes (esto no lo cronometramos)
        z_clientes_enc = [z_0_enc]
        for i in range(1, p):
            z_clientes_enc.append([public_key.encrypt(x) for x in inputs_z_i[i]])

        # 2) Agregacion y calculo residuos
        ini_coord = time.perf_counter()
        
        # Hay que sumar elemento a elemento los vectores de dim.N y para todos los 'p' clientes
        Z = list(z_0_enc) # partimos del primero
        for i in range(1, p):    
            for k in range(N):
                Z[k] = Z[k] + z_clientes_enc[i][k]
        
        # Los residuos se calculan para cada cliente y también es necesario ir elemento a elemento
        e_i_enc = list(Z)   # Copia de Z
        for i in range(p):  # Sobreescribimos los residuos de cada cliente en el bucle... en escenario real habría un diccionario
            for k in range(N):
                e_i_enc[k] = fd_bar_scalar - Z[k] + z_clientes_enc[i][k]

        fin_coord = time.perf_counter() - ini_coord
        t_coord += fin_coord
        
        # [Cliente]    
        # 3) Desencriptar e_i_enc
        ini_dec = time.perf_counter()

        e_i = [private_key.decrypt(x) for x in e_i_enc]
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


def plot_tiempos_vector(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.N, df.encriptar, 'o-', label=r'Cliente: encriptar $z_i$')
    plt.plot(df.N, df.desencriptar, '^-', label=r'Cliente: desencriptar $e_i$')
    plt.plot(df.N, df.agregar, 's-', label=r'Coord: agregación $\mathbf{Z}$ y cálculo $e_i$')
    plt.xlabel('Dimensión del vector (N)')
    plt.ylabel('Tiempo por iteración (s)')
    plt.title(f"Benchmark HE Aditivo Paillier (phe) - coste computacional - {df.p.values[0]} clientes")
    plt.legend()
    plt.grid()
    plt.show()


def plot_tiempos_clientes(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.p, df.agregar, 's-', label=r'Coord: agregación $\mathbf{Z}$ y cálculo $e_i$)')
    plt.xlabel('Número de clientes (p)')
    plt.ylabel('Tiempo por iteración (s)')
    plt.title(f"Benchmark HE Aditivo Paillier (phe) - coste computacional - Vector dim. {df.N.values[0]}")
    plt.legend()
    plt.grid()
    plt.show()


#%% [BENCHMARK]
print("Generando par de claves Paillier...")
public_key, private_key = paillier.generate_paillier_keypair()

# 1. Pruebas con tamaño vectores
N_lst = [2, 10, 50, 100, 500, 1000]
p = 3
df_lst = []
for N in N_lst:
    df = benchmark_phe(N, p, repeticiones=2)
    df_lst.append(df)
df_vectores = pd.concat(df_lst, ignore_index=True)

#%% [VISUALIZAMOS RESULTADOS]
plot_tiempos_vector(df_vectores)

#%%
# 2. Pruebas con numero de clientes
p_lst = [3, 10, 50, 100]
df_lst = []
for p in p_lst:
    df = benchmark_phe(25, p, repeticiones=1)
    df_lst.append(df)
df_clientes = pd.concat(df_lst, ignore_index=True)

#%% [VISUALIZAMOS RESULTADOS]
plot_tiempos_clientes(df_clientes)

# %%
