# VFL ROLANN

Esta carpeta contiene los ficheros esenciales necesarios para lanzar las pruebas del algoritmo BCD ROLANN.

He seleccionado y adaptado los ficheros del repositorio ROLANN para que no sea necesario instalar el repositorio como módulo de Python ni nada parecido.

Los ficheros con los primeros tests (`test_bcd_*.py`) se deberían de ejecutar sin problemas. Se recomienda ejecutar en VSCode o cualquier otro editor los ficheros de pruebas `test_bcd_*.py` para ver los primeros resultados y entender la estructura del código.

El fichero principal es `bcd.py`, donde están tanto las clases de cliente como coordinador del BCD ROLANN. Hay varias variantes: la original (`BCDClient|BCDCoordinator`), la que usa paralelización (`BCDClient_par|BCDCoordinator_par`) y otras dos variantes basadas en esta última que usan distintas técnicas de computación segura, (`BCDClient_sec|BCDCoordinator_sec`) para SecureAgg y (`BCDClient_fhe|BCDCoordinator_fhe`) para FHE. Todas son equivalentes y producen los mismos resultados. La versión original es más fácil de *debuggear*.

Tanto el BCDROLANN como el ROLANN expuestos aquí no permiten multi-output de forma nativa, pero para tareas de regresión con varias salidas o tareas de clasificación multiclase se ofrecen *wrappers* que generan un modelo por salida y se calculan luego las métricas finales globales (ver `test_bcd_multi.py`).

---

## REQUISITOS:

+ Python >= 3.11 con: numpy, pandas, scipy, matplotlib, seaborn, sklearn, tenseal y criptography. Para un versionado exacto consultar `requirements.txt`.
+ Se recomienda el uso de un entorno virtual para no mezclar dependencias.
+ TODO: hacer un optional con VertiBench por que descarga demasiadas dependencias -> torch, cuda, nvidia-*

## DESCRIPCION FICHEROS

+ `activation_functions.py`: funciones de activación utilizadas y mapeo de clases.
+ `helpers.py`: funciones auxiliares para VFL (dividir matrices de datos, etc).
+ `plotters.py`: funciones para graficar resultados.
+ `rolann.py`: implementación básica del algoritmo original ROLANN.
+ `bcd.py`: implementación del algoritmo de cliente y coordinador del BCD ROLANN.
+ `lr.py`: implementación básica de un modelo LR|Ridge en VFL (para uso como benchmark).

Bajo `/tests`, las mencionadas pruebas `test_bcd_*.py`.

Bajo `/experiments`, las experimentos realizados una selección de datasets reales.

---

## TODO

1. En `lr.py` no existe ninguna versión de LR|Ridge_VFL con una paralelización real a través de `joblib` equivalente a `BCDClient_par|BCDCoordinator_par`. En caso de hacer comparativas en los experimentos entre ellos, es necesario antes su implementación.