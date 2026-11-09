"""Numerics package for CROSMOR."""

from crosmor.numerics.interpolation import rintpl, rintpl_vec
from crosmor.numerics.root_finding import zeroin, tolx_default, tolv_default
from crosmor.numerics.tridiagonal import tridiar
from crosmor.numerics.smoothing import smooth, smoop
from crosmor.numerics.ode_rk import rk_adaptive_step, comp_err, rk1st

__all__ = [
    "rintpl",
    "rintpl_vec",
    "zeroin",
    "tolx_default",
    "tolv_default",
    "tridiar",
    "smooth",
    "smoop",
    "rk_adaptive_step",
    "comp_err",
    "rk1st",
]
