"""Unit tests for CROSMOR numerical algorithms."""

import numpy as np
import pytest

from crosmor.numerics.interpolation import rintpl
from crosmor.numerics.root_finding import zeroin
from crosmor.numerics.tridiagonal import tridiar
from crosmor.numerics.smoothing import smooth, smoop
from crosmor.numerics.ode_rk import rk_adaptive_step


def test_rintpl_interpolation():
    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([0.0, 100.0, 200.0, 300.0])

    # In-bounds interpolation: rintpl(x_val, f, xa)
    assert np.isclose(rintpl(5.0, y, x), 50.0)
    assert np.isclose(rintpl(15.0, y, x), 150.0)

    # Clamping / boundary extrapolation
    assert np.isclose(rintpl(-5.0, y, x), 0.0)
    assert np.isclose(rintpl(35.0, y, x), 300.0)


def test_zeroin_root_finding():
    # Test polynomial f(x) = x^2 - 4 on [0, 5] -> root at 2.0
    def f_poly(x):
        return x**2 - 4.0

    root, ok = zeroin(0.0, 5.0, f_poly, tol_func=lambda x: 1e-8)
    assert ok
    assert np.isclose(root, 2.0, atol=1e-5)

    # Test transcendental f(x) = cos(x) - x on [0, 1] -> root ~ 0.739085
    def f_trans(x):
        return np.cos(x) - x

    root_cos, ok_cos = zeroin(0.0, 1.0, f_trans, tol_func=lambda x: 1e-8)
    assert ok_cos
    assert np.isclose(root_cos, 0.739085, atol=1e-5)


def test_tridiar_thomas_algorithm():
    # Solve system:
    # [ 2 -1  0 ] [x1]   [ 1 ]
    # [-1  2 -1 ] [x2] = [ 0 ]
    # [ 0 -1  2 ] [x3]   [ 1 ]
    # Solution: x1 = 1, x2 = 1, x3 = 1
    # TRIDIAR.F indexing: for row j, a[j-1] is subdiagonal
    a = np.array([-1.0, -1.0, 0.0])
    b = np.array([2.0, 2.0, 2.0])
    c = np.array([-1.0, -1.0, 0.0])
    r = np.array([1.0, 0.0, 1.0])

    x = tridiar(a, b, c, r)
    assert np.allclose(x, [1.0, 1.0, 1.0])


def test_smoothing_filters():
    x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    lx = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    mean = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    # Moving average over distance coef * lx[i]
    sm = smooth(x, lx, mean, 2, 1.0)
    assert np.isclose(sm, 15.0)  # Average of mean[1] and mean[2]

    # Diffusive smoothing of fractions
    psi = np.array([[0.3, 0.4, 0.5, 0.6, 0.7]])
    out_psi = smoop(psi, 5, 1, 0.2, x)
    assert out_psi.shape == psi.shape
    assert np.allclose(np.sum(out_psi, axis=0), 1.0)


def test_ode_rk4():
    # Integrate dy/dx = -y from x=0 to 1 with y(0) = 1.0
    # Exact solution: y(1) = exp(-1) = 0.367879...
    def deriv(x, y):
        return np.array([-y[0]])

    x_curr = 0.0
    y_curr = np.array([1.0])
    h = 0.1
    while x_curr < 1.0:
        step_h = min(h, 1.0 - x_curr)
        y_curr, _ = rk_adaptive_step(x_curr, y_curr, x_curr + step_h, deriv)
        x_curr += step_h

    assert np.isclose(y_curr[0], np.exp(-1.0), atol=1e-3)
