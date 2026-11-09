"""Unit tests for CROSMOR hydrodynamic and morphodynamic physical modules."""

import numpy as np
import pytest

from crosmor.models.config import WaveTimePoint
from crosmor.hydrodynamics.dispersion import solve_dispersion_wavelength
from crosmor.hydrodynamics.rayleigh import generate_rayleigh_classes
from crosmor.hydrodynamics.asymmetry import compute_orbital_asymmetry
from crosmor.morphodynamics.sediment_props import (
    compute_fluid_properties,
    compute_settling_velocity,
    compute_shields_critical,
)
from crosmor.morphodynamics.bedload import compute_bedload_cell


def test_dispersion_wavelength():
    # Deep water check: L0 = g * T^2 / (2*pi) ~ 156.13 m for T = 10 s
    c0_approx = 156.13 / 10.0
    rls_deep, c_deep, ateta, tpr, omeg = solve_dispersion_wavelength(
        tp=10.0,
        hd=1000.0,
        teta0_rad=0.0,
        c0=c0_approx,
    )
    assert np.isclose(rls_deep, 156.13, atol=1.0)

    # Shallow water check: c = sqrt(g * h) -> L ~ 31.32 m for h = 1 m
    rls_shallow, c_shallow, _, _, _ = solve_dispersion_wavelength(
        tp=10.0,
        hd=1.0,
        teta0_rad=0.0,
        c0=c0_approx,
    )
    assert np.isclose(rls_shallow, 31.32, atol=1.0)


def test_rayleigh_distribution():
    wave_series = [WaveTimePoint(time=0.0, hrms=2.0, t=8.0, teta=15.0, w10=0.0)]
    classes = generate_rayleigh_classes(time=0.0, nh=5, wave_series=wave_series)
    assert len(classes) == 5
    # Total probability must sum to 1.0
    total_p = sum(c.p for c in classes)
    assert np.isclose(total_p, 1.0, atol=1e-4)
    # Heights must be strictly increasing
    assert all(classes[i].h < classes[i + 1].h for i in range(len(classes) - 1))


def test_asymmetry():
    # Ruessink 2008 asymmetry test (switch1 = 1)
    ubwf, ubwb, asym, udel1 = compute_orbital_asymmetry(
        switch1=1,
        h_val=1.5,
        hd=3.0,
        rls=30.0,
        trel=8.0,
    )
    assert ubwf > ubwb  # Forward velocity peak greater under crest
    assert 0.0 < asym < 1.0


def test_fluid_and_sediment_props():
    rho_w, nu, delta = compute_fluid_properties(temp_c=15.0, salinity=0.0)
    assert np.isclose(rho_w, 1000.0, atol=5.0)
    assert nu > 0.0

    # Settling velocity for d50 = 200 microns
    ws = compute_settling_velocity(d_grain=0.0002, del_density=delta, rnu=nu)
    assert 0.015 < ws < 0.035

    # Critical Shields
    dster, taucr, theta_cr = compute_shields_critical(
        d_grain=0.0002,
        del_density=delta,
        rnu=nu,
        rhow=rho_w,
        pmud_pct=0.0,
    )
    assert 0.03 < theta_cr < 0.07


def test_bedload_cell():
    dsi = np.array([0.0002])
    psi = np.array([1.0])
    dsteri = np.array([5.0])
    egfi = np.array([1.0])
    fc1i_row = np.array([0.005])
    fw1i_row = np.array([0.01])

    sb_c, sb_l, sbi = compute_bedload_cell(
        vr=0.2,
        ur=-0.1,
        ub=0.05,
        tp1=8.0,
        abw=1.0,
        a_ref=0.01,
        ubwback=0.8,
        ubwfor=1.2,
        rc=0.02,
        rw=0.02,
        ra=0.02,
        delm=1.65,
        ateta_rad=0.2,
        phi_rad=0.0,
        bsly=0.0,
        tanif=0.6,
        rhow=1000.0,
        rhos=2650.0,
        d50=0.0002,
        taucr1=0.18,
        dster=5.0,
        facbed=1.0,
        kk1=1,
        kk2=1,
        dsi=dsi,
        psi=psi,
        dsteri=dsteri,
        egfi=egfi,
        fc1i_row=fc1i_row,
        fw1i_row=fw1i_row,
        fc1=0.005,
        fw1=0.01,
    )
    assert np.isfinite(sb_c)
    assert np.isfinite(sb_l)
    assert len(sbi) == 1
