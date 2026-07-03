#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
decorators.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decoradores para las funciones de activación que limitan su dominio y rango.
"""
# Standard libraries
from sys import float_info as sys_flt
from numbers import Number
# Third-party libraries
import numpy as np


def _is_scalar(x):
    return np.isscalar(x) or isinstance(x, Number)

def clamp_unitary_range(func):
    """
    clamp_unitary_range() Decorator to clamp the value of the arguments to open range (0, 1).

    :param func: function to be decorated
    :return: decorated function
    """
    lowest = sys_flt.epsilon
    highest = 1 - lowest
    def _wrapper(x):
        if _is_scalar(x):
            if not 0 < x < 1:
                x = lowest if x <= 0 else highest
            return func(x)
        elif isinstance(x, list):
            lst = []
            for item in x:
                if not 0 < item < 1:
                    replace = lowest if item <= 0 else highest
                    lst.append(replace)
                else:
                    lst.append(item)
            return func(lst)
        else:
            # Something ndarray-like
            arr = np.asarray(x)
            x = np.where(arr <= 0, lowest, np.where(arr >= 1, highest, arr)).astype(float)
            return func(x)
    return _wrapper

def clamp_abs_unitary_range(func):
    """
    clamp_abs_unitary_range() Decorator to clamp the value of the arguments to open range (-1, 1).

    :param func: function to be decorated
    :return: decorated function
    """
    lowest = -1 + sys_flt.epsilon
    highest = 1 - sys_flt.epsilon
    def _wrapper(x):
        if _is_scalar(x):
            if not lowest < x < highest:
                x = lowest if x <= -1 else highest
            return func(x)
        elif isinstance(x, list):
            lst = []
            for item in x:
                if not lowest < item < highest:
                    replace = lowest if item <= -1 else highest
                    lst.append(replace)
                else:
                    lst.append(item)
            return func(lst)
        else:
            # Something ndarray-like
            arr = np.asarray(x)
            x = np.where(arr <= -1, lowest, np.where(arr >= 1, highest, arr)).astype(float)
            return func(x)
    return _wrapper
