# VFL ROLANN

Esta carpeta contiene los ficheros esenciales necesarios para lanzar las pruebas del algoritmo BCD ROLANN.

He seleccionado y adaptado los ficheros del repositorio ROLANN para que no sea necesario instalar el repositorio como módulo de Python ni nada parecido.

Los ficheros con los primeros tests (`test_bcd_*.py`) se deberían de ejecutar sin problemas.

El fichero principal es `bcd.py`, donde están tanto las clases de cliente como coordinador del BCD ROLANN. Hay dos variantes (`BCDClient|BCDCoordinator` y `BCDClient_alt|BCDCoordinator_alt`).
Ambos son equivalentes y producen los mismos resultados pero la versión alternativa implementa los cálculos con operaciones más eficientes (*numpy broacasting*) e implementa una paralelización realista del bucle BCD (mediante librería *joblib* de Python).

Se recomienda ejecutar en VSCode o cualquier otro editor los ficheros de pruebas `test_bcd_*.py` para ver los primeros resultados y entender el desglose del código.

Tanto el BCD como el ROLANN expuestos aquí no permiten multi-output, por lo que para tareas de regresión con varias salidas o tareas de clasificación multiclase (ya que se convierte en 'k' modelos tipo OneVsAll binarios) hay que hacerlo de manera manual, generando un modelo por salida y calculando las métricas finales globales.

---

## REQUISITOS:

+ Python >= 3.10 con: numpy, pandas, matplotlib, seaborn y sklearn: `pip install numpy pandas matplotlib seaborn sklearn`

## DESCRIPCION FICHEROS

+ `activation_functions.py`: definidas las funciones de activación utilizadas.
+ `helpers.py`: funciones auxiliares para VFL (dividir matrices de datos, etc)
+ `plotters.py`: funciones auxiliares para grafica resultados.
+ `rolann.py`: implementación básica del algoritmo original ROLANN.
+ `bcd.py`: implementación del algoritmo de cliente y coordinador del BCD ROLANN.
+ `lr.py`: implementación sencilla de un modelo LogisticRegression|Ridge en VFL (uso como benchmark).


Por último, las mencionadas pruebas `test_bcd_*.py`.
