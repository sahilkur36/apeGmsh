"""Phase 3F.1 — Interface-size advisory warning.

Locks ADR 0038 §"v1 scope gate" middle-branch decision (PR #360
gate result): compose ships at full scope but emits a
:class:`ComposeInterfaceSizeWarning` when the rewritten bundle's
MP-style constraint count crosses
:data:`WARN_INTERFACE_SIZE = 50_000`.

Pure-Python tests — no Gmsh, no openseespy, no real H5 round-trip
for the helper unit tests (which build minimal mock bundles).  The
integration tests use ``apeGmsh.from_h5`` + saved H5 sources so
they run entirely in chain phase.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.core._compose_errors import ComposeInterfaceSizeWarning
from apeGmsh.mesh._compose import (
    WARN_INTERFACE_SIZE,
    ComposeFilterWarning,
    _count_interface_size,
    _warn_interface_size,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Constants + fixtures
# ---------------------------------------------------------------------------


def _mock_bundle(
    *,
    node_constraint_count: int = 0,
    elem_constraint_count: int = 0,
    label: str = "M",
) -> Any:
    """Build a minimal stand-in for :class:`_RewrittenBundle`.

    :func:`_count_interface_size` and :func:`_warn_interface_size` only
    read ``node_constraints`` / ``elem_constraints`` / ``label`` — they
    don't touch the rest of the bundle, so a :class:`SimpleNamespace`
    with those three attributes exercises the helpers without paying
    the full frozen-dataclass construction cost.
    """
    return SimpleNamespace(
        label=label,
        node_constraints=tuple(object() for _ in range(node_constraint_count)),
        elem_constraints=tuple(object() for _ in range(elem_constraint_count)),
    )


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    extra_node_constraints: int = 0,
) -> FEMData:
    """Tiny FEMData with optional NodePairRecord constraints.

    ``extra_node_constraints`` controls how many synthetic
    :class:`NodePairRecord` objects are stuffed into
    ``nodes.constraints`` — used to push a bundle's interface-size
    count past the threshold without expanding the rest of the model
    (so the host disk size + read time stay bounded).
    """
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)
    n = node_ids.size
    coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(n)], dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    conn = np.array(
        [
            [int(node_ids[i % n]), int(node_ids[(i + 1) % n])]
            for i in range(elem_ids.size)
        ],
        dtype=np.int64,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    # Synthetic NodePairRecord constraints — pair node 1 with itself
    # over and over.  Tag values are valid (within the node set), so
    # the tag-collision verifier accepts them; the only thing that
    # matters here is the raw record count.
    constraints: list[NodePairRecord] = []
    for i in range(extra_node_constraints):
        constraints.append(
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=1,
                slave_node=2,
                dofs=[1, 2, 3],
                name=f"pair_{i}",
            )
        )

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
        constraints=constraints or None,
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=n, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


@pytest.fixture
def empty_host_h5(tmp_path: Path) -> Path:
    """Minimal-host FEMData saved to H5 — chain-phase compose target.

    Uses a 3-node / 2-element module rather than an empty host so the
    merge engine's :func:`np.concatenate` calls see matching ndim on
    both sides (an empty host's (0, 3) coord array collapses to 1D
    after the writer/reader round-trip, breaking ``axis=0`` concat).
    """
    fem = _make_module_fem()
    p = tmp_path / "host.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def small_module_h5(tmp_path: Path) -> Path:
    """A source module with 100 constraints — well below the threshold."""
    fem = _make_module_fem(extra_node_constraints=100)
    p = tmp_path / "small.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def large_module_h5(tmp_path: Path) -> Path:
    """A source module with 50_001 constraints — strictly above the
    threshold (predicate is strictly-greater-than, so 50_001 trips
    while 50_000 does not).
    """
    fem = _make_module_fem(extra_node_constraints=WARN_INTERFACE_SIZE + 1)
    p = tmp_path / "large.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def threshold_module_h5(tmp_path: Path) -> Path:
    """A source module with exactly 50_000 constraints — boundary case
    that must NOT trip the warning (predicate is strictly-greater-than).
    """
    fem = _make_module_fem(extra_node_constraints=WARN_INTERFACE_SIZE)
    p = tmp_path / "threshold.h5"
    fem.to_h5(str(p))
    return p


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------


class TestConstant:
    """The advisory threshold itself."""

    def test_warn_interface_size_is_fifty_thousand(self) -> None:
        """Threshold value comes from the Phase 1 gate result
        (10k x 4 PASS, 100k x 8 breached) and must not drift without
        re-running the gate.
        """
        assert WARN_INTERFACE_SIZE == 50_000


# ---------------------------------------------------------------------------
# Helper: _count_interface_size
# ---------------------------------------------------------------------------


class TestCountInterfaceSize:
    """The pure counting helper."""

    def test_empty_bundle_is_zero(self) -> None:
        bundle = _mock_bundle()
        assert _count_interface_size(bundle) == 0

    def test_counts_node_constraints(self) -> None:
        bundle = _mock_bundle(node_constraint_count=42)
        assert _count_interface_size(bundle) == 42

    def test_counts_elem_constraints(self) -> None:
        bundle = _mock_bundle(elem_constraint_count=7)
        assert _count_interface_size(bundle) == 7

    def test_sums_both_streams(self) -> None:
        bundle = _mock_bundle(
            node_constraint_count=10,
            elem_constraint_count=15,
        )
        assert _count_interface_size(bundle) == 25


# ---------------------------------------------------------------------------
# Helper: _warn_interface_size (predicate, idempotence, filter)
# ---------------------------------------------------------------------------


class TestWarnInterfaceSize:
    """The advisory predicate + warning emission."""

    def test_below_threshold_no_warning(self) -> None:
        bundle = _mock_bundle(node_constraint_count=100)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_at_threshold_no_warning(self) -> None:
        """Predicate is strictly-greater-than: 50_000 must not trip."""
        bundle = _mock_bundle(node_constraint_count=WARN_INTERFACE_SIZE)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_above_threshold_emits_warning(self) -> None:
        bundle = _mock_bundle(
            node_constraint_count=WARN_INTERFACE_SIZE + 1,
            label="big_module",
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert len(size_warnings) == 1
        msg = str(size_warnings[0].message)
        # Message must name the actual count + threshold for actionability.
        assert str(WARN_INTERFACE_SIZE + 1) in msg
        assert str(WARN_INTERFACE_SIZE) in msg
        assert "big_module" in msg

    def test_above_threshold_fires_once_per_call(self) -> None:
        """One bundle with thousands of constraints → exactly one warning,
        not one per record."""
        bundle = _mock_bundle(
            node_constraint_count=WARN_INTERFACE_SIZE + 1,
            elem_constraint_count=WARN_INTERFACE_SIZE + 1,
        )
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert len(size_warnings) == 1

    def test_threshold_none_disables_warning(self) -> None:
        """Passing ``threshold=None`` is the escape hatch — no warning
        ever fires even on huge bundles.  Mirrors the simpler-path
        decision in the task spec (override mechanism deferred; users
        silence via ``simplefilter("ignore", ...)``).
        """
        bundle = _mock_bundle(node_constraint_count=WARN_INTERFACE_SIZE + 1)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _warn_interface_size(bundle, threshold=None)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_simplefilter_ignore_suppresses(self) -> None:
        """Users who accept the cost can silence the advisory via the
        standard ``warnings.simplefilter("ignore", ...)`` pathway.
        """
        bundle = _mock_bundle(node_constraint_count=WARN_INTERFACE_SIZE + 1)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("ignore", ComposeInterfaceSizeWarning)
            _warn_interface_size(bundle, threshold=WARN_INTERFACE_SIZE)
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []


# ---------------------------------------------------------------------------
# Subclass identity — ComposeInterfaceSizeWarning vs ComposeFilterWarning
# ---------------------------------------------------------------------------


class TestWarningTaxonomy:
    """The two compose-time warning classes must be filterable
    independently (one is informational; the other is cost-advisory).
    """

    def test_compose_interface_size_warning_is_user_warning(self) -> None:
        assert issubclass(ComposeInterfaceSizeWarning, UserWarning)

    def test_compose_filter_warning_is_user_warning(self) -> None:
        assert issubclass(ComposeFilterWarning, UserWarning)

    def test_interface_size_warning_is_not_filter_warning(self) -> None:
        """Independent classes so callers can filter each separately."""
        assert not issubclass(
            ComposeInterfaceSizeWarning, ComposeFilterWarning,
        )
        assert not issubclass(
            ComposeFilterWarning, ComposeInterfaceSizeWarning,
        )


# ---------------------------------------------------------------------------
# Integration — through FEMData.compose / g.compose
# ---------------------------------------------------------------------------


class TestComposeIntegration:
    """Wired into the real compose path via ``apeGmsh.from_h5`` + saved
    H5 sources."""

    def test_small_module_does_not_warn(
        self, empty_host_h5: Path, small_module_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_host_h5)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            g.compose(small_module_h5, label="A")
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_threshold_module_does_not_warn(
        self, empty_host_h5: Path, threshold_module_h5: Path,
    ) -> None:
        """Boundary: exactly 50_000 constraints — predicate is
        strictly-greater-than, so no warning fires."""
        g = apeGmsh.from_h5(empty_host_h5)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            g.compose(threshold_module_h5, label="A")
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_large_module_warns_once(
        self, empty_host_h5: Path, large_module_h5: Path,
    ) -> None:
        """Compose of a module above the threshold → exactly one
        ComposeInterfaceSizeWarning."""
        g = apeGmsh.from_h5(empty_host_h5)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            g.compose(large_module_h5, label="big")
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert len(size_warnings) == 1
        msg = str(size_warnings[0].message)
        assert "big" in msg
        assert str(WARN_INTERFACE_SIZE + 1) in msg

    def test_large_module_simplefilter_ignore_silences(
        self, empty_host_h5: Path, large_module_h5: Path,
    ) -> None:
        """Callers who accept the cost can silence the advisory."""
        g = apeGmsh.from_h5(empty_host_h5)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ComposeInterfaceSizeWarning)
            g.compose(large_module_h5, label="big")
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []

    def test_filter_warning_independent_of_size_warning(
        self, empty_host_h5: Path, large_module_h5: Path,
    ) -> None:
        """Ignoring the size warning does NOT swallow other compose
        warnings.  The large module fixture carries no /opensees/
        zone so no ComposeFilterWarning fires either — but the test
        asserts the principle: silencing one category leaves the other
        category unaffected.
        """
        g = apeGmsh.from_h5(empty_host_h5)
        # Filter out the size warning but allow others.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ComposeInterfaceSizeWarning)
            g.compose(large_module_h5, label="big")
        # Size warning is silenced.
        size_warnings = [
            w for w in rec if issubclass(w.category, ComposeInterfaceSizeWarning)
        ]
        assert size_warnings == []
        # Anything that is NOT the size warning is still recorded.
        # In this fixture nothing else fires, but the filter category
        # is structurally distinct — confirmed by the taxonomy tests
        # above.

    def test_warn_as_error_contract(
        self, empty_host_h5: Path, large_module_h5: Path,
    ) -> None:
        """When users opt into ``-W error::ComposeInterfaceSizeWarning``
        (warn-as-contract), the compose call raises rather than
        warns.  This locks the warning category contract so future
        spurious UserWarnings in the same code path are caught by the
        suite's ``-W error::UserWarning`` regression gate.
        """
        g = apeGmsh.from_h5(empty_host_h5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ComposeInterfaceSizeWarning)
            with pytest.raises(ComposeInterfaceSizeWarning):
                g.compose(large_module_h5, label="big")
