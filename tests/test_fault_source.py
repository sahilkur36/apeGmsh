"""MT-4 (ADR 0062) — finite-fault → moment-tensor source converter.

Unit-tests the pure converter
:mod:`apeGmsh._kernel.resolvers._fault_source`: layered shear modulus,
``M0 = μ·A·D̄`` with the explicit unit scales, the frame z-sign, and the
array-alignment / units guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.resolvers._fault_source import (
    ffsp_subfaults_to_sources,
    shear_modulus_at_depth,
)


# --- shear_modulus_at_depth ----------------------------------------------

def test_shear_modulus_layered_lookup():
    crust = dict(
        thickness_m=np.array([1000.0, 2000.0]),
        vs_ms=np.array([1000.0, 2000.0]),
        rho_kgm3=np.array([2000.0, 2500.0]),
    )
    # depth 500 m → layer 0: μ = 2000·1000² = 2e9
    assert shear_modulus_at_depth(500.0, **crust) == pytest.approx(2.0e9)
    # depth 1500 m → layer 1: μ = 2500·2000² = 1e10
    assert shear_modulus_at_depth(1500.0, **crust) == pytest.approx(1.0e10)
    # depth below all interfaces → half-space (last layer)
    assert shear_modulus_at_depth(9999.0, **crust) == pytest.approx(1.0e10)


def test_shear_modulus_mismatched_arrays_fail_loud():
    with pytest.raises(ValueError, match="equal-length"):
        shear_modulus_at_depth(
            10.0, thickness_m=np.array([1.0]), vs_ms=np.array([1.0, 2.0]),
            rho_kgm3=np.array([1.0]),
        )


# --- ffsp_subfaults_to_sources -------------------------------------------

def _subfaults(**over):
    base = dict(
        x=np.array([0.5, 1.5]),            # km
        y=np.array([0.0, 0.0]),
        z=np.array([0.5, 0.5]),            # km depth
        slip=np.array([0.5, 1.0]),         # m
        strike=np.array([350.0, 10.0]),
        dip=np.array([40.0, 80.0]),
        rake=np.array([113.0, 0.0]),
        rupture_time=np.array([0.0, 0.3]),
        rise_time=np.array([1.0, 1.2]),
        peak_time=np.array([0.2, 0.3]),
    )
    base.update(over)
    return base


_CRUST = dict(
    thickness=np.array([1000.0]),          # m (half-space)
    vs=np.array([1000.0]),                 # m/s
    rho=np.array([2000.0]),                # kg/m³  → μ = 2e9 Pa
)


def test_ffsp_moment_and_units():
    specs = ffsp_subfaults_to_sources(
        _subfaults(), crust=_CRUST, area_m2=1.0e6,
        length_scale=1000.0, moment_scale=1.0e-3, frame="z-down",
    )
    assert len(specs) == 2
    # μ=2e9, A=1e6, D=0.5 → M0_SI = 1e15 N·m; ×1e-3 → 1e12 deck
    assert specs[0].M0 == pytest.approx(2.0e9 * 1.0e6 * 0.5 * 1.0e-3)
    assert specs[1].M0 == pytest.approx(2.0e9 * 1.0e6 * 1.0 * 1.0e-3)
    # mechanism + timing pass through
    assert specs[0].strike == 350.0 and specs[0].t0 == 0.0
    assert specs[1].t0 == 0.3


def test_ffsp_position_scaling_and_frame():
    z_down = ffsp_subfaults_to_sources(
        _subfaults(), crust=_CRUST, area_m2=1.0e6,
        length_scale=1000.0, moment_scale=1.0e-3, frame="z-down",
    )
    z_up = ffsp_subfaults_to_sources(
        _subfaults(), crust=_CRUST, area_m2=1.0e6,
        length_scale=1000.0, moment_scale=1.0e-3, frame="z-up",
    )
    # x = 0.5 km × 1000 = 500 m (both frames)
    assert z_down[0].position[0] == pytest.approx(500.0)
    # z = 0.5 km depth → +500 (z-down) vs -500 (z-up)
    assert z_down[0].position[2] == pytest.approx(500.0)
    assert z_up[0].position[2] == pytest.approx(-500.0)


def test_ffsp_misaligned_arrays_fail_loud():
    bad = _subfaults(slip=np.array([0.5]))   # length 1 vs 2
    with pytest.raises(ValueError, match="not aligned"):
        ffsp_subfaults_to_sources(
            bad, crust=_CRUST, area_m2=1.0e6,
            length_scale=1000.0, moment_scale=1.0e-3, frame="z-down",
        )


def test_ffsp_missing_key_fails_loud():
    sub = _subfaults()
    del sub["rupture_time"]
    with pytest.raises(ValueError, match="missing keys"):
        ffsp_subfaults_to_sources(
            sub, crust=_CRUST, area_m2=1.0e6,
            length_scale=1000.0, moment_scale=1.0e-3, frame="z-down",
        )


def test_ffsp_nonpositive_area_fails_loud():
    with pytest.raises(ValueError, match="area_m2 must be > 0"):
        ffsp_subfaults_to_sources(
            _subfaults(), crust=_CRUST, area_m2=0.0,
            length_scale=1000.0, moment_scale=1.0e-3, frame="z-down",
        )
