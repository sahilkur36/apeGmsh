"""PR0 / ADR 0027 viewer-side data plumbing — partition_by_eid +
boundary_node_ids.

PR0 wires the schema 2.10.0 ``partition_ids`` column on
``/opensees/element_meta/{token}/`` and the per-rank
``boundary_node_ids`` arrays under ``/opensees/partitions/`` into
:class:`apeGmsh.viewers.data.ViewerData`.  The viewer renders by FEM
element id (broker space), so the OpenSees-tag-indexed bridge column
gets joined with the parallel ``fem_eids`` column to produce a
``{fem_eid: rank}`` map exposed via
:meth:`ViewerElements.partition_for`.

The decoders live with the other H5-side decoders in
``viewers/data/_viewer_data.py``; these tests pin them against tiny
synthetic readers that satisfy only the surface each decoder calls.
End-to-end pinning against a partitioned ``model.h5`` lands in PR1
when ``ColorMode.PARTITION`` consumes the wired plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from apeGmsh.viewers.data._viewer_data import (
    _decode_boundary_node_ids,
    _decode_partition_by_eid,
)


# =====================================================================
# _decode_partition_by_eid
# =====================================================================


class _MetaOnlyReader:
    """Minimal reader exposing only the surface ``_decode_partition_by_eid``
    consumes: :meth:`element_meta` and :meth:`element_meta_arrays`.
    """

    def __init__(
        self,
        meta: dict[str, dict[str, Any]],
        arrays: dict[str, dict[str, Any]],
    ) -> None:
        self._meta = meta
        self._arrays = arrays

    def element_meta(self) -> dict[str, dict[str, Any]]:
        return self._meta

    def element_meta_arrays(self, token: str) -> dict[str, Any]:
        if token not in self._arrays:
            raise KeyError(token)
        return self._arrays[token]


def test_partition_by_eid_joins_fem_eids_with_partition_ids() -> None:
    """Two type tokens, both carrying ``fem_eids`` + ``partition_ids``
    — the join produces ``{fem_eid: rank}`` across them.
    """
    reader = _MetaOnlyReader(
        meta={"forceBeamColumn": {}, "elasticBeamColumn": {}},
        arrays={
            "forceBeamColumn": {
                "ids": np.array([100, 101], dtype=np.int64),
                "fem_eids": np.array([10, 11], dtype=np.int64),
                "partition_ids": np.array([0, 0], dtype=np.int64),
            },
            "elasticBeamColumn": {
                "ids": np.array([200, 201], dtype=np.int64),
                "fem_eids": np.array([20, 21], dtype=np.int64),
                "partition_ids": np.array([1, 1], dtype=np.int64),
            },
        },
    )
    out = _decode_partition_by_eid(reader)
    assert out == {10: 0, 11: 0, 20: 1, 21: 1}


def test_partition_by_eid_skips_minus_one_sentinels() -> None:
    """``fem_eids == -1`` marks no-FEM-origin records and ``partition_ids
    == -1`` marks emission outside a ``partition_open`` bracket.  Both
    sentinels must be excluded from the map.
    """
    reader = _MetaOnlyReader(
        meta={"forceBeamColumn": {}},
        arrays={
            "forceBeamColumn": {
                "ids": np.array([100, 101, 102, 103], dtype=np.int64),
                "fem_eids": np.array([10, -1, 12, 13], dtype=np.int64),
                "partition_ids": np.array([0, 1, -1, 1], dtype=np.int64),
            },
        },
    )
    out = _decode_partition_by_eid(reader)
    # Row 0: feid=10, rank=0 -> kept
    # Row 1: feid=-1            -> skipped (no FEM origin)
    # Row 2: feid=12, rank=-1   -> skipped (outside partition bracket)
    # Row 3: feid=13, rank=1    -> kept
    assert out == {10: 0, 13: 1}


def test_partition_by_eid_missing_column_yields_empty() -> None:
    """Pre-2.10.0 archives carry no ``partition_ids`` column; pre-Phase
    8.6 archives carry no ``fem_eids`` column.  Either absence yields
    an empty map without raising.
    """
    reader = _MetaOnlyReader(
        meta={"forceBeamColumn": {}},
        arrays={
            "forceBeamColumn": {
                "ids": np.array([100, 101], dtype=np.int64),
                # fem_eids absent
                "partition_ids": np.array([0, 0], dtype=np.int64),
            },
        },
    )
    assert _decode_partition_by_eid(reader) == {}


def test_partition_by_eid_empty_meta_yields_empty() -> None:
    """No element_meta groups at all (mesh-only / pre-bridge archive)
    yields an empty map silently."""
    reader = _MetaOnlyReader(meta={}, arrays={})
    assert _decode_partition_by_eid(reader) == {}


def test_partition_by_eid_no_element_meta_method_yields_empty() -> None:
    """Foreign-format adapters that don't implement ``element_meta``
    at all (the method raises) degrade to an empty map — exception
    catch is per the existing convention for bridge-zone reads."""

    class _NoBridgeReader:
        def element_meta(self) -> dict[str, dict[str, Any]]:
            raise RuntimeError("foreign-format reader has no bridge zone")

        def element_meta_arrays(self, token: str) -> dict[str, Any]:
            raise KeyError(token)

    assert _decode_partition_by_eid(_NoBridgeReader()) == {}


# =====================================================================
# _decode_boundary_node_ids
# =====================================================================


@dataclass(frozen=True)
class _FakePartitionRecord:
    """Minimal stand-in for ``PartitionEmittedRecord`` — the decoder
    only reads ``boundary_node_ids``."""
    rank: int
    boundary_node_ids: tuple[int, ...]


class _PartitionsReader:
    """Minimal reader exposing only :meth:`partitions`."""

    def __init__(self, recs: list[_FakePartitionRecord]) -> None:
        self._recs = recs

    def partitions(self) -> list[_FakePartitionRecord]:
        return self._recs


def test_boundary_node_ids_union_across_ranks() -> None:
    """Two ranks each carrying their own boundary set — the result is
    the union, symmetric across ranks per ADR 0027 INV-3."""
    reader = _PartitionsReader([
        _FakePartitionRecord(rank=0, boundary_node_ids=(2, 5)),
        _FakePartitionRecord(rank=1, boundary_node_ids=(2, 7)),
    ])
    out = _decode_boundary_node_ids(reader)
    assert out == frozenset({2, 5, 7})


def test_boundary_node_ids_empty_records_yield_empty() -> None:
    """Single-partition models produce ``partitions()`` records whose
    ``boundary_node_ids`` are empty tuples (per
    :class:`PartitionEmittedRecord` docstring).  Result: empty set."""
    reader = _PartitionsReader([
        _FakePartitionRecord(rank=0, boundary_node_ids=()),
    ])
    assert _decode_boundary_node_ids(reader) == frozenset()


def test_boundary_node_ids_no_partitions_method_yields_empty() -> None:
    """``partitions()`` is NOT in the ADR 0026 H5ModelReader Protocol —
    foreign-format adapters may omit it entirely.  The decoder uses
    ``getattr`` rather than try/except so an absent method degrades to
    the empty set without swallowing real runtime errors."""

    class _NoPartitionsReader:
        # No partitions() method at all
        pass

    assert _decode_boundary_node_ids(_NoPartitionsReader()) == frozenset()


def test_boundary_node_ids_empty_partitions_list_yields_empty() -> None:
    """Pre-2.10.0 archives and 2.10.x archives written outside any
    ``partition_open`` bracket both produce an empty partitions list
    from the apeGmsh reader."""
    reader = _PartitionsReader([])
    assert _decode_boundary_node_ids(reader) == frozenset()


# =====================================================================
# ViewerElements / ViewerNodes slot wiring
# =====================================================================


def test_viewer_elements_partition_slot_defaults_empty() -> None:
    """``ViewerElements`` constructed without ``partition_by_eid`` (the
    from_fem path) reports ``has_partitions == False`` and
    ``partition_for(any) is None``."""
    from apeGmsh.viewers.data._elements import (
        ElementLoadView,
        SurfaceConstraintView,
        ViewerElements,
    )
    from apeGmsh.viewers.data._nodes import _NamedNodeSelection

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    ve = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
    )
    assert ve.has_partitions is False
    assert ve.partition_for(42) is None


def test_viewer_elements_partition_slot_populated() -> None:
    """``ViewerElements`` accepts a ``partition_by_eid`` kwarg and
    surfaces it through ``partition_for`` / ``has_partitions``."""
    from apeGmsh.viewers.data._elements import (
        ElementLoadView,
        SurfaceConstraintView,
        ViewerElements,
    )
    from apeGmsh.viewers.data._nodes import _NamedNodeSelection

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    ve = ViewerElements(
        groups=[],
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=ElementLoadView([]),
        constraints=SurfaceConstraintView([]),
        partition_by_eid={10: 0, 11: 1},
    )
    assert ve.has_partitions is True
    assert ve.partition_for(10) == 0
    assert ve.partition_for(11) == 1
    assert ve.partition_for(99) is None


def test_viewer_nodes_boundary_slot_defaults_empty() -> None:
    """``ViewerNodes`` constructed without ``boundary_node_ids`` (the
    from_fem path) reports an empty frozenset and
    ``has_boundary_nodes == False``."""
    from apeGmsh.viewers.data._nodes import (
        MassView,
        NodalLoadView,
        NodeConstraintView,
        SPView,
        ViewerNodes,
        _NamedNodeSelection,
    )

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    vn = ViewerNodes(
        ids=np.array([], dtype=np.int64),
        coords=np.zeros((0, 3), dtype=np.float64),
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=NodalLoadView([]), sp=SPView([]),
        masses=MassView([]), constraints=NodeConstraintView([]),
    )
    assert vn.has_boundary_nodes is False
    assert vn.boundary_node_ids == frozenset()


def test_viewer_nodes_boundary_slot_populated() -> None:
    """``ViewerNodes`` accepts a ``boundary_node_ids`` kwarg and exposes
    it as an immutable frozenset."""
    from apeGmsh.viewers.data._nodes import (
        MassView,
        NodalLoadView,
        NodeConstraintView,
        SPView,
        ViewerNodes,
        _NamedNodeSelection,
    )

    empty_sel = _NamedNodeSelection({}, raise_on_missing=True, label="x")
    vn = ViewerNodes(
        ids=np.array([], dtype=np.int64),
        coords=np.zeros((0, 3), dtype=np.float64),
        physical=empty_sel, labels=empty_sel, selection=empty_sel,
        loads=NodalLoadView([]), sp=SPView([]),
        masses=MassView([]), constraints=NodeConstraintView([]),
        boundary_node_ids=frozenset({2, 5, 7}),
    )
    assert vn.has_boundary_nodes is True
    assert vn.boundary_node_ids == frozenset({2, 5, 7})
    # Immutable — assignment should not be possible because the
    # property returns a frozenset (not a mutable set).
    assert isinstance(vn.boundary_node_ids, frozenset)
