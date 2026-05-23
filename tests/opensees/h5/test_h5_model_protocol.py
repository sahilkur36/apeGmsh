"""ADR 0026 PR7-a — ``H5Model`` additive surfaces for the H5ModelReader Protocol.

Three new methods/properties on
:class:`apeGmsh.opensees.emitter.h5_reader.H5Model` so existing
already-open-reader code paths can satisfy the
:class:`apeGmsh.viewers.data._protocol.H5ModelReader` Protocol
(landed in PR7-c) without re-opening the file:

- ``path`` property — re-exposes the originally-passed file path
- ``has_opensees_orientation()`` — equivalent to the module-level
  probe in ``viewers/data/_h5_probe.py`` but on an already-open reader
- ``has_neutral_zone()`` — re-uses the ``neutral_schema_version``
  marker that :func:`open` already inspects at construction

The methods are intentionally cheap and side-effect-free; these
tests pin the contract before PR7-b (``FemToOpsTagMap.from_reader``)
becomes the first downstream consumer.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter


# =====================================================================
# Fixtures
# =====================================================================


def _write_oriented(path: Path) -> None:
    """Bridge-only oriented file: ``/opensees/transforms`` +
    ``/opensees/element_meta`` populated; no neutral zone."""
    em = H5Emitter(model_name="oriented", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    em.add_oriented_elements(
        type_token="forceBeamColumn",
        vecxz=(0.0, 1.0, 0.0),
        elements=[(101, (1, 2)), (102, (2, 3))],
        ndm=3,
    )
    with h5py.File(path, "w") as f:
        em._write_meta(f)
        em.write_opensees_into(f)


def _write_bridge_no_orientation(path: Path) -> None:
    """Bridge-only file with /meta but NO orientation groups."""
    em = H5Emitter(model_name="bare", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    with h5py.File(path, "w") as f:
        em._write_meta(f)
        em.write_opensees_into(f)


# =====================================================================
# path — property exposes the open()-time argument
# =====================================================================


def test_path_returns_pathlib_path(tmp_path: Path) -> None:
    """``H5Model.path`` is a :class:`pathlib.Path` regardless of how
    the file was passed to :func:`open` (``str`` is the public arg)."""
    p = tmp_path / "oriented.h5"
    _write_oriented(p)

    with h5_reader.open(str(p)) as model:
        assert isinstance(model.path, Path)
        assert model.path == p


def test_path_matches_open_argument(tmp_path: Path) -> None:
    """The path round-trips exactly what was passed — no resolution,
    no normalisation. Adapters that want a resolved path must do
    that themselves."""
    p = tmp_path / "subdir" / "oriented.h5"
    p.parent.mkdir()
    _write_oriented(p)

    with h5_reader.open(str(p)) as model:
        assert str(model.path) == str(p)


# =====================================================================
# has_opensees_orientation — both /opensees/ groups present
# =====================================================================


def test_has_opensees_orientation_true_on_oriented_file(tmp_path: Path) -> None:
    p = tmp_path / "oriented.h5"
    _write_oriented(p)

    with h5_reader.open(str(p)) as model:
        assert model.has_opensees_orientation() is True


def test_has_opensees_orientation_false_on_bare_bridge(tmp_path: Path) -> None:
    """A bridge file without ``add_oriented_elements`` calls writes
    neither ``/opensees/transforms`` nor ``/opensees/element_meta``."""
    p = tmp_path / "bare.h5"
    _write_bridge_no_orientation(p)

    with h5_reader.open(str(p)) as model:
        assert model.has_opensees_orientation() is False


def test_has_opensees_orientation_matches_standalone_probe(tmp_path: Path) -> None:
    """The method on the open reader returns the same answer as the
    standalone path-only probe at
    ``viewers/data/_h5_probe.has_opensees_orientation``. Cheaper —
    no second h5py.File open — but byte-equivalent in result."""
    from apeGmsh.viewers.data._h5_probe import has_opensees_orientation

    oriented = tmp_path / "oriented.h5"
    bare = tmp_path / "bare.h5"
    _write_oriented(oriented)
    _write_bridge_no_orientation(bare)

    with h5_reader.open(str(oriented)) as m:
        assert m.has_opensees_orientation() == has_opensees_orientation(oriented)
    with h5_reader.open(str(bare)) as m:
        assert m.has_opensees_orientation() == has_opensees_orientation(bare)


def test_has_opensees_orientation_is_side_effect_free(tmp_path: Path) -> None:
    """The probe never modifies the file. Two calls return the same
    answer; an unrelated read between them still works."""
    p = tmp_path / "oriented.h5"
    _write_oriented(p)

    with h5_reader.open(str(p)) as model:
        first = model.has_opensees_orientation()
        _ = model.meta()
        second = model.has_opensees_orientation()
        assert first is True and second is True


# =====================================================================
# has_neutral_zone — the broker-emit marker
# =====================================================================


def test_has_neutral_zone_false_on_bridge_only_file(tmp_path: Path) -> None:
    """Bridge-only files (no FEMData round-trip) do not carry the
    ``neutral_schema_version`` attribute on /meta — they are not
    "broker emit" files in the ADR 0023 / 0026 sense."""
    p = tmp_path / "bridge_only.h5"
    _write_oriented(p)

    with h5_reader.open(str(p)) as model:
        assert model.has_neutral_zone() is False


def test_has_neutral_zone_true_when_marker_attribute_present(tmp_path: Path) -> None:
    """Files written through ``FEMData.to_h5`` carry the
    ``neutral_schema_version`` attribute on /meta. The probe checks
    that attribute directly — re-using the same marker the
    schema-version validator at ``h5_reader.py:157`` consults. This
    test stamps the marker by hand to keep the probe surface focused
    (round-trip via real FEMData is covered by integration tests)."""
    em = H5Emitter(model_name="with_neutral_marker", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    p = tmp_path / "with_marker.h5"
    with h5py.File(p, "w") as f:
        em._write_meta(f)
        em.write_opensees_into(f)
        # Stamp the broker-emit marker (what FEMData.to_h5 would do).
        f["meta"].attrs["neutral_schema_version"] = "2.6.0"

    with h5_reader.open(str(p)) as model:
        assert model.has_neutral_zone() is True
