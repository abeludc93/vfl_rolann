#!/usr/bin/env python3
"""
activation_functions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Funciones de activación usadas en ROLANN/BCDROLANN.

También se definen funciones auxiliares para el mapeado
de clases y obtención de funciones derivadas e inversas.

EPSILON: establece el margen del mapeado binario.
    (0, 1) -> (NEG_VALUE, POS_VALUE)
    logsig (0 + EPSILON,  1 - EPSILON)
    tanhip (-1 + EPSILON, 1 - EPSILON)
Ej: EPSILON = 0.05 -> logsig (0.05, 0.95) | tanhip (-0.95, 0.95)
"""
# Standard libraries
from sys import float_info as sys_flt
# Third-party libraries
from numpy import log, exp, ones, divide, ndarray, clip, where

EPSILON = .05               # label mapping margins
FLT_EPS = sys_flt.epsilon   # system float min value

# LOG-SIGMOID activation functions:
#   f(x) range (0, 1) → need to clamp inverse
def logsig(x):
    x_clip = clip(x, -250, 250)                 # clip to (-250, 250) to avoid overflow
    return 1 / (1 + (exp(-x_clip)))
def _inv_logsig(x):
    return -log(divide(1, x) - 1)
def inv_logsig(x):
    x_clip = clip(x, 0+FLT_EPS, 1-FLT_EPS)      # clamp input to (0, 1)
    return _inv_logsig(x_clip)
def der_logsig(x):
    return 1 / ((1 + exp(-x)) ** 2) * exp(-x)   # = logsig(x) (1 - logsig(x))

# LINEAR activation functions:
#   f(x) range (-inf, inf), that is, all R → no need to clamp inverse
def linear(x):
    return x
def inv_linear(x):
    return x
def der_linear(x):
    return ones(len(x)) if type(x) is list or type(x) is ndarray else 1

# HYPERBOLIC TANGENT activation function:
#   f(x) range (-1, 1) → need to clamp inverse
def tanhip(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def _inv_tanhip(x):
    return 0.5 * log((1 + x) / (1 - x))
def inv_tanhip(x):
    x_clip = clip(x, -1+FLT_EPS, 1-FLT_EPS)     # clamp input to (-1, 1)
    return _inv_tanhip(x_clip)
def der_tanhip(x):
    return 1 - tanhip(x)**2

# AUXILIARY FUNCTIONS:
def load_act_fn(fn="linear"):
    """
    Returns a tuple of three callable functions depending on the 'fn' param,
    representing the activation function desired for the neural network.

    :param fn: 'linear', 'logs' or 'tanhip' (string)
    :return: tuple of three callable functions
    """
    if fn == "linear":
        return linear, inv_linear, der_linear
    elif fn == "logs":
        return logsig, inv_logsig, der_logsig
    elif fn == 'tanhip':
        return tanhip, inv_tanhip, der_tanhip
    else:
        raise ValueError(f"Activation function unknown: {fn}")

def map_labels(fn="logs", eps=EPSILON):
    """
    Returns a tuple specific to the given act. function used ('fn').
    First element is a callable that maps the labels/targets accordingly.
    Second elements is a tuple with the pair (NEG/POS) mapvalues.

    :param fn: 'linear', 'logs' or 'tanhip' (string)
    :return: callable and a two-element tuple ()
    """
    neg_logsig, pos_logsig =  0 + eps, 1 - eps
    neg_tanhip, pos_tanhip = -1 + eps, 1 - eps
    neg_linear, pos_linear =  1      , 1

    map_logsig = lambda y: where(y == 1, pos_logsig, neg_logsig)
    map_tanhip = lambda y: where(y == 1, pos_tanhip, neg_tanhip)
    map_linear = lambda y: y

    if fn == "linear":
        return map_linear, (neg_linear, pos_linear)
    elif fn == "logs":
        return map_logsig, (neg_logsig, pos_logsig)
    elif fn == "tanhip":
        return map_tanhip, (neg_tanhip, pos_tanhip)
    else:
        raise ValueError(f"Activation function unknown: {fn}")
