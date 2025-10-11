"""Core functionality for z_tensor package."""

import numpy as np


def example_function(x):
    """
    An example function that doubles the input.
    
    Parameters
    ----------
    x : int or float or numpy.ndarray
        The input value(s) to be doubled
        
    Returns
    -------
    int or float or numpy.ndarray
        The input value(s) multiplied by 2
        
    Examples
    --------
    >>> example_function(5)
    10
    >>> example_function(np.array([1, 2, 3]))
    array([2, 4, 6])
    """
    return x * 2
