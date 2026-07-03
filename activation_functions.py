#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
activation_functions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Funciones de activación usadas en ROLANN/BCDROLANN.
Ojo! Importante!
En caso de tarea de clasificación, es necesario cambiar NEG_VALUE y POS_VALUE,
para mapear el rango de las etiquetas (0, 1) a (NEG_VALUE, POS_VALUE), según la
función de activación usada.

Ej: logsig (0.05, 0.95) | tanhip (-0.9, 0.9)

"""
# Third-party libraries
from numpy import log, exp, ones, divide, ndarray, clip
# Application modules
from decorators import clamp_unitary_range, clamp_abs_unitary_range

NEG_VALUE = 0.05 #TODO: arreglar de una vez!
POS_VALUE = 0.95

# Avoid using the @decorator's notation for multiprocessing reasons (can't be pickled)
# LOG-SIGMOID activation functions: f(x) range (0, 1) → need to clamp inverse
def logsig(x):
    x_clip = clip(x, -250, 250)                 # clip to (-250, 250) to avoid overflow
    return 1 / (1 + (exp(-x_clip)))
def _inv_logsig(x):
    return -log(divide(1, x) - 1)
def inv_logsig(x):
    return clamp_unitary_range(_inv_logsig)(x)  # clamp input to (0, 1)
def der_logsig(x):
    return 1 / ((1 + exp(-x)) ** 2) * exp(-x)   # = logsig(x) (1 - logsig(x))
def map_logsig(y):
    return y * (POS_VALUE - NEG_VALUE) + NEG_VALUE  # TODO: np.where(...)

# LINEAR activation functions: f(x) range (-inf, inf), that is R → no need to clamp inverse
def linear(x):
    return x
def inv_linear(x):
    return x
def der_linear(x):
    return ones(len(x)) if type(x) is list or type(x) is ndarray else 1
def map_linear(y):
    return y

# HYPERBOLIC TANGENT activation function: f(x) range (-1 ,1) → need to clamp inverse
def tanhip(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def _inv_tanhip(x):
    return 0.5 * log((1 + x) / (1 - x))
def inv_tanhip(x):
    return clamp_abs_unitary_range(_inv_tanhip)(x)  # clamp input to (-1, 1)
def der_tanhip(x):
    return 1 - tanhip(x)**2
def map_tanhip(y):
    return y * (2 * POS_VALUE) - NEG_VALUE

# AUXILIARY FUNCTION:
def load_act_fn(fn="linear"):
    """
    load_act_fn() Returns a tuple of three callable functions depending on the 'fn' string, representing
    the activation function desired for the neural network.

    :param fn: string for the activation function 'linear'|'logs'|'tanhip' (string)
    :return: tuple of three callable functions.
    """
    if fn == "linear":
        return linear, inv_linear, der_linear
    elif fn == "logs":
        return logsig, inv_logsig, der_logsig
    elif fn == 'tanhip':
        return tanhip, inv_tanhip, der_tanhip
    else:
        raise ValueError(f"Activation function unknown: {fn}")

def map_labels(fn="logs"):
    if fn == "linear":
        return map_linear
    elif fn == "logs":
        return map_logsig
    elif fn == "tanhip":
        return map_tanhip
    else:
        raise ValueError(f"Activation function unknown: {fn}")
