"""Beach and dune kinematics module matching CROSMOR BEACH.F."""

import math
from typing import Tuple
import numpy as np


def compute_beach_profile_update(
    stlgr: float,
    dt: float,
    por: float,
    rhos: float,
    xggr: float,
    x_grid: np.ndarray,
    hdg: np.ndarray,
    nxx: int,
    nx: int,
    zbn: np.ndarray,
    sl_degrees: float,
    d50: float,
    hs0: float,
    ls0: float,
    tanbx: float,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute runup and bed level adjustments landward of the water line matching BEACH.F.

    Returns:
        (updated_zbn, runup, lgr, zm)
    """
    if tanbx <= 0.0 or ls0 <= 0.0 or hs0 <= 0.0:
        return zbn, 0.0, 0.1, 0.0

    tanbx_clamped = min(2.5, max(0.01, tanbx))
    ibar = tanbx_clamped / math.sqrt(hs0 / ls0)
    thulp = max(0.2, math.tanh(3.4 * ibar))
    runup = 0.4 * hs0 * thulp

    if d50 >= 0.002:
        runup = min(5.0, 1.0 * math.sqrt(hs0 * ls0))
    else:
        runup = min(5.0, runup)

    hend1 = hdg[nxx - 1]
    zrunup = runup + hdg[nxx - 1] + zbn[nxx - 1]

    # Find runup length lgr1
    lgr1 = 0.1
    for i in range(nxx - 1, nx):
        if zrunup <= zbn[i] or i == nx - 1:
            next_idx = min(i + 1, nx - 1)
            lgr1 = x_grid[next_idx] - xggr
            break

    # Dune erosion length lgr
    start_idx = max(0, nxx - 5)
    hhdg = float(np.mean(hdg[start_idx:nxx]))
    lgr = 6.0 * hhdg

    if lgr < lgr1:
        lgr = lgr1
    if lgr > 6.0 * hend1:
        lgr = 6.0 * hend1
    if lgr > 10.0:
        lgr = 10.0

    if d50 >= 0.002:
        lgr = runup / tanbx_clamped
        if lgr >= 15.0:
            lgr = 15.0

    zm = 0.0
    if lgr > 0.0 and abs(1.0 - por) > 1e-4:
        zm = stlgr / (0.5 * lgr * (1.0 - por) * rhos) * dt

    out_zbn = np.copy(zbn)

    # Apply triangular berm/dune adjustment
    if lgr > 0.0:
        for i in range(nxx - 1, nx):
            xx = x_grid[i]
            if xggr <= xx <= xggr + 0.8 * lgr:
                corbe = (xx - xggr) / (0.5 * lgr) * zm
                out_zbn[i] += corbe
            elif xggr + 0.8 * lgr < xx <= xggr + lgr:
                corbe = (xggr + lgr - xx) / (0.5 * lgr) * zm
                out_zbn[i] += corbe

    return out_zbn, runup, lgr, zm
