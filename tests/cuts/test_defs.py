"""Phase-1 unit tests for :class:`apeGmsh.cuts.SectionCutDef`.

These tests exercise the scaffold:

* Construction validation (normal != 0, non-empty filter, side enum).
* Auto-normalisation of the plane normal.
* Pickle round-trip without STKO_to_python in the picture.
* ``to_spec()`` round-trip when STKO_to_python is available
  (importorskip otherwise).

Phase-2+ tests live in their own files.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef


# --------------------------------------------------------------------- #
# Construction + validation
# --------------------------------------------------------------------- #
def test_construct_happy_path():
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 100.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1, 2, 3),
    )
    assert d.plane_point == (0.0, 0.0, 100.0)
    assert d.plane_normal == (0.0, 0.0, 1.0)
    assert d.element_ids == (1, 2, 3)
    assert d.side == "positive"
    assert d.label is None
    assert d.bounding_polygon is None


def test_normal_is_auto_normalised():
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 5.0),     # not unit
        element_ids=(1,),
    )
    assert d.plane_normal == (0.0, 0.0, 1.0)


def test_zero_normal_raises():
    with pytest.raises(ValueError, match="nonzero"):
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 0.0),
            element_ids=(1,),
        )


def test_empty_element_ids_raises():
    with pytest.raises(ValueError, match="non-empty"):
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(),
        )


def test_invalid_side_raises():
    with pytest.raises(ValueError, match="side"):
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(1,),
            side="up",  # type: ignore[arg-type]
        )


def test_non_finite_point_raises():
    with pytest.raises(ValueError, match="finite"):
        SectionCutDef(
            plane_point=(0.0, 0.0, float("nan")),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(1,),
        )


def test_element_ids_from_ndarray():
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=np.array([4, 5, 6], dtype=np.int64),
    )
    assert d.element_ids == (4, 5, 6)
    assert all(isinstance(e, int) for e in d.element_ids)


def test_bounding_polygon_too_few_vertices():
    with pytest.raises(ValueError, match="at least 3 vertices"):
        SectionCutDef(
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=(0.0, 0.0, 1.0),
            element_ids=(1,),
            bounding_polygon=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )


def test_bounding_polygon_stored_as_tuple_of_tuples():
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        bounding_polygon=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )
    assert d.bounding_polygon == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )


# --------------------------------------------------------------------- #
# Pickle (no STKO needed)
# --------------------------------------------------------------------- #
def test_pickle_roundtrip(tmp_path):
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 100.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1, 2, 3),
        side="negative",
        label="story-3",
        bounding_polygon=((0.0, 0.0, 100.0), (10.0, 0.0, 100.0), (10.0, 10.0, 100.0)),
    )
    p = d.save_pickle(tmp_path / "cut.pkl")
    restored = SectionCutDef.load_pickle(p)
    assert restored == d


def test_pickle_gzip_via_suffix(tmp_path):
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    p = d.save_pickle(tmp_path / "cut.pkl.gz")
    assert p.exists()
    restored = SectionCutDef.load_pickle(p)
    assert restored == d


# --------------------------------------------------------------------- #
# STKO interop (each test importorskips so non-STKO tests still run)
# --------------------------------------------------------------------- #
def test_to_spec_basic():
    stko_cuts = pytest.importorskip("STKO_to_python.cuts")
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 100.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1, 2, 3),
        label="story-3",
    )
    spec = d.to_spec()
    assert isinstance(spec, stko_cuts.SectionCutSpec)
    assert spec.plane.point == (0.0, 0.0, 100.0)
    assert spec.plane.normal == (0.0, 0.0, 1.0)
    assert spec.element_ids == (1, 2, 3)
    assert spec.side == "positive"
    assert spec.label == "story-3"
    assert spec.bounding_polygon is None


def test_to_spec_with_bounding_polygon():
    pytest.importorskip("STKO_to_python.cuts")
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 100.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        bounding_polygon=(
            (0.0, 0.0, 100.0),
            (10.0, 0.0, 100.0),
            (10.0, 10.0, 100.0),
            (0.0, 10.0, 100.0),
        ),
    )
    spec = d.to_spec()
    assert spec.bounding_polygon == (
        (0.0, 0.0, 100.0),
        (10.0, 0.0, 100.0),
        (10.0, 10.0, 100.0),
        (0.0, 10.0, 100.0),
    )


def test_to_spec_side_negative():
    pytest.importorskip("STKO_to_python.cuts")
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        side="negative",
    )
    spec = d.to_spec()
    assert spec.side == "negative"
    np.testing.assert_allclose(spec.signed_normal, (0.0, 0.0, -1.0))


def test_to_spec_hashable():
    pytest.importorskip("STKO_to_python.cuts")
    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1, 2),
    )
    spec_a = d.to_spec()
    spec_b = d.to_spec()
    assert spec_a == spec_b
    assert hash(spec_a) == hash(spec_b)


def test_to_spec_raises_clean_error_when_stko_missing(monkeypatch):
    """If STKO_to_python isn't installed, .to_spec() must raise
    ImportError with the install hint — not a cryptic
    ModuleNotFoundError.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("STKO_to_python"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    d = SectionCutDef(
        plane_point=(0.0, 0.0, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
    )
    with pytest.raises(ImportError, match="STKO_to_python"):
        d.to_spec()
