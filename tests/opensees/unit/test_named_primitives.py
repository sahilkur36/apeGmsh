"""Unit tests for bridge-side named primitives (name= alias).

``ops.timeSeries.Linear(..., name="ramp")`` registers the primitive
under a bridge-side alias; reference kwargs (``series=``) then accept
either the object handle or the name string. The name is a lookup
entry on the bridge — the primitive stays pure/tag-less, so nothing
flows into the lineage hash or the h5 schema.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.time_series.time_series import Linear


def _bridge() -> apeSees:
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration + dual-mode resolution
# ---------------------------------------------------------------------------

def test_name_registers_alias_and_returns_handle() -> None:
    ops = _bridge()
    ts = ops.timeSeries.Linear(name="ramp")
    assert isinstance(ts, Linear)
    # The same object is reachable by name.
    assert ops._resolve("ramp") is ts


def test_pattern_series_accepts_name_string() -> None:
    ops = _bridge()
    ts = ops.timeSeries.Linear(name="ramp")
    p = ops.pattern.Plain(series="ramp")
    # Resolved back to the exact same TimeSeries instance the name maps to.
    assert p.series is ts


def test_pattern_series_still_accepts_object_handle() -> None:
    ops = _bridge()
    ts = ops.timeSeries.Linear()          # anonymous — no name
    p = ops.pattern.Plain(series=ts)
    assert p.series is ts


def test_two_named_series_get_distinct_tags() -> None:
    ops = _bridge()
    a = ops.timeSeries.Linear(name="dead")
    b = ops.timeSeries.Linear(name="live")
    assert ops.tag_for(a) != ops.tag_for(b)


# ---------------------------------------------------------------------------
# Fail-loud behaviour
# ---------------------------------------------------------------------------

def test_duplicate_name_raises() -> None:
    ops = _bridge()
    ops.timeSeries.Linear(name="ramp")
    with pytest.raises(ValueError, match="already registered"):
        ops.timeSeries.Constant(name="ramp")


def test_unknown_name_raises_with_known_names() -> None:
    ops = _bridge()
    ops.timeSeries.Linear(name="ramp")
    with pytest.raises(KeyError, match="ramp"):
        ops.pattern.Plain(series="nope")


def test_name_kind_mismatch_raises() -> None:
    ops = _bridge()
    # Register a non-TimeSeries under a name, then mis-reference it as a series.
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0, nu=0.2, name="steel")
    assert mat is not None
    with pytest.raises(TypeError, match="TimeSeries is required"):
        ops.pattern.Plain(series="steel")


# ---------------------------------------------------------------------------
# Cross-family fan-out — every reference kwarg is dual-mode
# ---------------------------------------------------------------------------

def test_element_material_accepts_name() -> None:
    ops = _bridge()
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, name="conc")
    ele = ops.element.FourNodeTetrahedron(pg="Body", material="conc")
    assert ele.material is mat


def test_beam_integration_section_accepts_name() -> None:
    ops = _bridge()
    sec = ops.section.Elastic(E=1.0, A=1.0, Iz=1.0, name="sec")
    integ = ops.beamIntegration.Lobatto(section="sec", n_ip=5)
    assert integ.section is sec


def test_beam_column_transf_and_integration_accept_names() -> None:
    ops = _bridge()
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="t")
    ops.section.Elastic(E=1.0, A=1.0, Iz=1.0, name="sec")
    integ = ops.beamIntegration.Lobatto(section="sec", n_ip=3, name="integ")
    beam = ops.element.forceBeamColumn(
        pg="Cols", transf="t", integration="integ"
    )
    assert beam.transf is t
    assert beam.integration is integ


def test_material_wrapper_base_accepts_name() -> None:
    ops = _bridge()
    base = ops.uniaxialMaterial.ElasticMaterial(E=1.0, name="el")
    wrapped = ops.uniaxialMaterial.InitialStress(
        base_material="el", sigma_init=1.0
    )
    assert wrapped.base_material is base
