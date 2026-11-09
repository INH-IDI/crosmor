"""Near-bed orbital velocity asymmetry formulations matching CROSMOR."""

import math
from typing import Tuple


def orbital_asymmetry_isobe(
    h_val: float,
    hd: float,
    rls: float,
    trel: float,
    g: float = 9.81,
    hgrens: float = 0.25,
) -> Tuple[float, float, float, float]:
    """Isobe and Horikawa (2003) formulation (Switch1 = 0).

    Returns:
        (ubwf, ubwb, asym, udel1)
    """
    pi = math.pi
    kx = 2.0 * pi / rls
    kh = max(1e-4, kx * hd)
    sinh_kh = math.sinh(kh)
    adel = 0.0

    y13 = h_val * h_val
    udel = pi * math.sqrt(y13) / (trel * sinh_kh) if sinh_kh > 0 else 0.0
    adel = udel * trel / (2.0 * pi)

    # udel1 based on y13
    udel1 = (3.0 * pi * pi * y13) / (4.0 * trel * rls * (sinh_kh**4)) if sinh_kh > 0 else 0.0
    rhs13 = math.sqrt(y13) / hd

    rr = -0.4 * rhs13 + 1.0
    umax = rr * 2.0 * udel
    t1 = trel * math.sqrt(g / hd)
    u1 = umax / math.sqrt(g * hd)

    a11 = -0.0049 * (t1**2) - 0.069 * t1 + 0.2911
    ra = -5.25 - 6.1 * math.tanh(a11 * u1 - 1.76)
    if ra <= 0.5:
        ra = 0.5

    rmax = -2.5 * hd / rls + 0.85
    if rmax >= 0.75:
        rmax = 0.75
    if rmax <= 0.62:
        rmax = 0.62

    ubwf = umax * (0.5 + (rmax - 0.5) * math.tanh((ra - 0.5) / (rmax - 0.5)))

    hdhulp = min(1.5, 3.0 * hgrens)
    reduc1 = 1.0
    if hd <= hdhulp:
        reduc1 = (hd / hdhulp) ** 2.0
    ubwf = reduc1 * ubwf
    if ubwf <= 0.65 * umax and hd <= hdhulp:
        ubwf = 0.65 * umax

    ubwb = umax - ubwf
    if ubwb <= 0.0:
        ubwb = 0.001

    asym = ubwf / (ubwf + ubwb) if (ubwf + ubwb) > 0.0 else 0.5
    return ubwf, ubwb, asym, udel1


def orbital_asymmetry_ruessink2008(
    h_val: float,
    hd: float,
    rls: float,
    trel: float,
) -> Tuple[float, float, float, float]:
    """Ruessink et al. (2008) formulation (Switch1 = 1).

    Returns:
        (ubwf, ubwb, asym, udel1)
    """
    pi = math.pi
    skh = (2.0 * pi / rls) * hd
    skh = max(1e-4, skh)
    sinh_skh = math.sinh(skh)

    ursel = 0.75 * 0.5 * h_val * (2.0 * pi / rls) * (skh ** (-3.0))
    ursel = max(1e-6, ursel)

    sig = pi * (h_val / 1.41) / (1.41 * trel * sinh_skh)
    hhulp1 = 2.8256 * (-0.6065 - math.log10(ursel))
    bbb = 0.7939 * ((1.0 + math.exp(hhulp1)) ** (-1.0))
    beta = -90.0 + 90.0 * math.tanh(0.6373 / (ursel**0.5995))
    betar = (beta / 360.0) * (2.0 * pi)

    skew = bbb * math.cos(betar)

    cos_betar = math.cos(betar)
    if abs(cos_betar) < 1e-9:
        cos_betar = 1e-9

    arg_acos = -(1.5**0.5) * skew / cos_betar
    arg_acos = max(-1.0, min(1.0, arg_acos))
    phir1 = (1.0 / 3.0) * math.acos(arg_acos)
    phir2 = phir1 + (2.0 / 3.0) * pi
    phir3 = phir1 + (4.0 / 3.0) * pi

    factor = (2.0 / 3.0) ** 0.5 * 2.0 * sig
    am21 = factor * math.cos(phir1)
    am22 = factor * math.cos(phir2)
    am23 = factor * math.cos(phir3)

    # Smallest positive root
    pos_roots = [r for r in [am21, am22, am23] if r > 0.0]
    am2 = min(pos_roots) if pos_roots else 0.0

    radicand = 2.0 * (sig**2) - (am2**2)
    am1 = math.sqrt(max(0.0, radicand))

    # Search min/max in 100 steps over wave cycle
    thh = (2.0 * pi) / trel
    dt1 = trel / 100.0
    zz1 = 0.001
    zz2 = -0.001

    for step in range(101):
        tt = step * dt1
        ut = am1 * math.cos(thh * tt) + am2 * math.cos(2.0 * thh * tt - betar)
        if ut > zz1:
            zz1 = ut
        if ut < zz2:
            zz2 = ut

    ubwf = zz1
    ubwb = abs(zz2)
    asym = ubwf / (ubwf + ubwb) if (ubwf + ubwb) > 0.0 else 0.5
    udel1 = 0.0

    return ubwf, ubwb, asym, udel1


def orbital_asymmetry_ruessink2010(
    h_val: float,
    hd: float,
    rls: float,
    trel: float,
) -> Tuple[float, float, float, float]:
    """Ruessink et al. (2010) formulation (Switch1 = 2).

    Returns:
        (ubwf, ubwb, asym, udel1)
    """
    pi = math.pi
    skh = (2.0 * pi / rls) * hd
    skh = max(1e-4, skh)
    sinh_skh = math.sinh(skh)

    ursel = 0.75 * 0.5 * h_val * (2.0 * pi / rls) * (skh ** (-3.0))
    ursel = max(1e-6, ursel)

    bbb = 0.8507 / (1.0 + math.exp(-1.586 - 3.367 * math.log10(ursel)))
    rr1 = math.tanh(0.9305 * bbb)

    pp1 = (-90.0 / 360.0) * (2.0 * pi) * math.tanh(0.815 / (ursel**0.672))
    sqrt_term = math.sqrt(max(0.0, 1.0 - (rr1**2)))
    hh1 = rr1 * math.sin(pp1) / (1.0 + sqrt_term)
    fff = sqrt_term
    uww = pi * (h_val / 1.1) / (trel * sinh_skh)
    u1 = uww * fff

    thh = (2.0 * pi) / trel
    dt1 = trel / 100.0
    zz1 = 0.001
    zz2 = -0.001

    for step in range(101):
        tt = step * dt1
        denom = 1.0 - rr1 * math.cos(thh * tt + pp1)
        if abs(denom) < 1e-9:
            denom = 1e-9
        ut = u1 * (math.sin(thh * tt) + hh1) / denom
        if ut > zz1:
            zz1 = ut
        if ut < zz2:
            zz2 = ut

    ubwf = zz1
    ubwb = abs(zz2)
    asym = ubwf / (ubwf + ubwb) if (ubwf + ubwb) > 0.0 else 0.5
    udel1 = 0.0

    return ubwf, ubwb, asym, udel1


def compute_orbital_asymmetry(
    switch1: int,
    h_val: float,
    hd: float,
    rls: float,
    trel: float,
    g: float = 9.81,
    hgrens: float = 0.25,
) -> Tuple[float, float, float, float]:
    """Unified dispatcher for near-bed orbital asymmetry."""
    if switch1 == 0:
        return orbital_asymmetry_isobe(h_val, hd, rls, trel, g, hgrens)
    elif switch1 == 1:
        return orbital_asymmetry_ruessink2008(h_val, hd, rls, trel)
    else:
        return orbital_asymmetry_ruessink2010(h_val, hd, rls, trel)
