"""Tests for core functionality."""

import numpy as np
import pytest

from z_tensor.core import example_function


def test_example_function_with_int():
    """Test example_function with integer input."""
    assert example_function(5) == 10
    assert example_function(0) == 0
    assert example_function(-3) == -6


def test_example_function_with_float():
    """Test example_function with float input."""
    assert example_function(2.5) == 5.0
    assert example_function(-1.5) == -3.0


def test_example_function_with_array():
    """Test example_function with numpy array input."""
    input_array = np.array([1, 2, 3, 4])
    expected = np.array([2, 4, 6, 8])
    result = example_function(input_array)
    
    np.testing.assert_array_equal(result, expected)


def test_example_function_with_zero():
    """Test example_function with zero."""
    assert example_function(0) == 0
    assert example_function(0.0) == 0.0
