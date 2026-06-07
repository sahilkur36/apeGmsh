"""Partitioned mixed-ndf emit — cross-rank consistency via inference (ADR 0048).

Per-node ``ndf`` is INFERRED from the declared element classes and applied
authoritatively at emit. This file pins the **partitioned** mixed-ndf case:
a node declared on more than one rank (owned on its home rank, foreign/ghost
on a rank that references it through a cross-partition MP constraint) must
emit the SAME ``ndf`` token on every rank. Because inference is deterministic
over the global broker + element set, the owner and every ghost resolve
identically — no cross-rank communication, no per-node broker store.

Fixture
=======

A minimal 4-node, 2-partition stub. Two 2-node elements give distinct ndf:

* ``shell_pg`` → ``elasticBeamColumn`` on nodes 1, 2 → **ndf 6** (rank 0)
* ``solid_pg`` → ``Truss`` on nodes 3, 4 → **ndf 3** (rank 1)

A single cross-partition ``equalDOF(2, 4, [1,2,3])`` ties the two interfaces,
so node 2 is foreign on rank 1 and node 4 is foreign on rank 0 — exercising
the per-foreign-node emit path. The envelope is ``ndf=6``; with elide-on-equal
the ndf-6 nodes emit bare (envelope wins) and the ndf-3 nodes carry ``-ndf 3``.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import (
    ATTR_PHANTOM_NODE_TAGS,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# Silence the bridge's documented MP-auto-emit warnings (constraint
# handler / numberer / system) so the file is runnable under
# ``pytest -W error::UserWarning``.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


# ---------------------------------------------------------------------------
# Per-rank call bucketing helpers.
# ---------------------------------------------------------------------------


def _per_rank_calls(rec: RecordingEmitter) -> dict[int, list[tuple]]:
    """Bucket emit calls by the enclosing ``partition_open(K)`` rank."""
    out: dict[int, list[tuple]] = {}
    cur: int | None = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            cur = int(args[0])
            out.setdefault(cur, [])
        elif name == "partition_close":
            cur = None
        elif cur is not None:
            out[cur].append((name, args, kwargs))
    return out


def _node_emits_by_tag(calls: list[tuple]) -> dict[int, dict]:
    """Return ``{tag: kwargs}`` for every ``node(...)`` call in one rank's
    bucket (``{}`` kwargs means ``-ndf`` was elided — the envelope wins)."""
    out: dict[int, dict] = {}
    for name, args, kwargs in calls:
        if name != "node":
            continue
        tag = int(args[0])
        assert tag not in out, (
            f"node {tag} emitted twice within one rank's block: "
            f"first={out[tag]!r}, second={kwargs!r}"
        )
        out[tag] = dict(kwargs)
    return out


# ---------------------------------------------------------------------------
# Fixture — 4-node mixed-ndf interface with one cross-partition equalDOF.
# ---------------------------------------------------------------------------


def _make_mixed_ndf_partitioned_fem() -> FEMStub:
    """Build the 2-partition mixed-ndf fixture (rank 0 ndf-6, rank 1 ndf-3)."""
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 1.0),  # 1 — beam, rank 0
            (1.0, 0.0, 1.0),  # 2 — beam interface, rank 0
            (1.0, 0.0, 0.0),  # 3 — truss interface, rank 1
            (0.0, 0.0, 0.0),  # 4 — truss, rank 1
        ],
        node_pgs={"shell_pg": [1, 2], "solid_pg": [3, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "shell_pg": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
            "solid_pg": _ElementGroupView(ids=(2,), connectivity=((3, 4),)),
        },
    )
    stub = FEMStub(nodes=nodes, elements=elements)
    stub.set_partitions([
        (0, [1, 2], [1]),
        (1, [3, 4], [2]),
    ])
    return stub


def _declare_mixed_elements(ops: apeSees) -> None:
    """Declare a 3D beam (ndf 6) on ``shell_pg`` and a truss (ndf 3) on
    ``solid_pg`` so inference assigns nodes 1,2 → 6 and nodes 3,4 → 3."""
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="shell_pg", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.element.Truss(
        pg="solid_pg", A=0.01,
        material=ops.uniaxialMaterial.ElasticMaterial(E=200e9),
    )


def _interface_tie() -> NodePairRecord:
    return NodePairRecord(
        kind=ConstraintKind.EQUAL_DOF,
        master_node=2, slave_node=4, dofs=[1, 2, 3], name="interface_tie",
    )


# ---------------------------------------------------------------------------
# 1. Headline — inferred per-node ndf on owned AND foreign decls, elided
#    when equal to the envelope.
# ---------------------------------------------------------------------------


def test_emit_partitioned_mixed_ndf_inferred() -> None:
    fem = _make_mixed_ndf_partitioned_fem()
    fem.add_node_constraints([_interface_tie()])

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)       # envelope = 6 (the beam/ndf-6 value)
    _declare_mixed_elements(ops)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    per_rank = _per_rank_calls(rec)
    assert set(per_rank.keys()) == {0, 1}

    rank0 = _node_emits_by_tag(per_rank[0])
    rank1 = _node_emits_by_tag(per_rank[1])

    # Rank 0 owns the ndf-6 beam nodes (1, 2) + foreign truss node 4.
    assert set(rank0.keys()) == {1, 2, 4}
    assert rank0[1] == {} and rank0[2] == {}, (
        f"ndf-6 owned nodes equal the envelope (6) → -ndf elided; "
        f"got {rank0[1]!r}, {rank0[2]!r}"
    )
    assert rank0[4] == {"ndf": 3}, (
        f"foreign truss node 4 must carry its inferred ndf=3 (not the "
        f"envelope 6) so it matches its owner on rank 1; got {rank0[4]!r}"
    )

    # Rank 1 owns the ndf-3 truss nodes (3, 4) + foreign beam node 2.
    assert set(rank1.keys()) == {2, 3, 4}
    assert rank1[3] == {"ndf": 3} and rank1[4] == {"ndf": 3}, (
        f"ndf-3 owned nodes differ from the envelope (6) → explicit -ndf 3; "
        f"got {rank1[3]!r}, {rank1[4]!r}"
    )
    assert rank1[2] == {}, (
        f"foreign beam node 2 infers 6 == envelope → -ndf elided; "
        f"got {rank1[2]!r}"
    )


# ---------------------------------------------------------------------------
# 2. Cross-rank consistency — a node's emitted ndf token is identical on
#    every rank that declares it (owner + ghost).
# ---------------------------------------------------------------------------


def test_partitioned_per_node_ndf_consistent_across_ranks() -> None:
    fem = _make_mixed_ndf_partitioned_fem()
    fem.add_node_constraints([_interface_tie()])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _declare_mixed_elements(ops)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    per_rank = _per_rank_calls(rec)

    rank0 = _node_emits_by_tag(per_rank[0])
    rank1 = _node_emits_by_tag(per_rank[1])

    # Tag 4 — owned on rank 1, foreign on rank 0: both carry -ndf 3.
    assert rank0[4] == rank1[4] == {"ndf": 3}, (
        f"tag 4 must emit identical ndf on every declaring rank "
        f"(rank0 foreign={rank0[4]!r}, rank1 owned={rank1[4]!r})"
    )
    # Tag 2 — owned on rank 0, foreign on rank 1: both elided (6==envelope).
    assert rank0[2] == rank1[2] == {}, (
        f"tag 2 must emit identical ndf on every declaring rank "
        f"(rank0 owned={rank0[2]!r}, rank1 foreign={rank1[2]!r})"
    )


# ---------------------------------------------------------------------------
# 3. Foreign-node decl precedes the constraint line on each rank (INV-2).
# ---------------------------------------------------------------------------


def test_partitioned_foreign_node_decl_precedes_constraint() -> None:
    fem = _make_mixed_ndf_partitioned_fem()
    fem.add_node_constraints([_interface_tie()])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _declare_mixed_elements(ops)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    per_rank = _per_rank_calls(rec)

    def _first_index(calls: list[tuple], pred) -> int:
        for i, (name, args, kwargs) in enumerate(calls):
            if pred(name, args, kwargs):
                return i
        return -1

    for rank, foreign_tag in ((0, 4), (1, 2)):
        calls = per_rank[rank]
        decl_idx = _first_index(
            calls,
            lambda n, a, _k, t=foreign_tag: (
                n == "node" and len(a) >= 1 and int(a[0]) == t
            ),
        )
        ed_idx = _first_index(calls, lambda n, _a, _k: n == "equalDOF")
        assert decl_idx != -1, f"rank {rank}: foreign node {foreign_tag} not declared"
        assert ed_idx != -1, f"rank {rank}: equalDOF not emitted"
        assert decl_idx < ed_idx, (
            f"INV-2 regressed on rank {rank}: foreign node {foreign_tag} "
            f"decl at {decl_idx} must precede equalDOF at {ed_idx}"
        )


# ---------------------------------------------------------------------------
# 4. Phantom-tag predicate set contains only synthetic phantom tags.
# ---------------------------------------------------------------------------


def test_partitioned_phantom_predicate_set_empty_when_no_n2s() -> None:
    fem = _make_mixed_ndf_partitioned_fem()
    fem.add_node_constraints([_interface_tie()])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    _declare_mixed_elements(ops)
    rec = RecordingEmitter()
    ops.build().emit(rec)

    predicate = getattr(rec, ATTR_PHANTOM_NODE_TAGS, None)
    assert predicate is not None, (
        "emit_mp_constraints_partitioned must install the phantom-tag "
        "predicate on the emitter"
    )
    assert predicate == frozenset(), (
        f"no NodeToSurfaceRecord exists; the phantom-tag predicate must be "
        f"the empty frozenset. Got {predicate!r}"
    )


# ---------------------------------------------------------------------------
# 5. Uniform model — every node infers the envelope → no per-node -ndf token.
# ---------------------------------------------------------------------------


def test_partitioned_uniform_emits_no_per_node_ndf() -> None:
    """A partitioned model whose every node infers the envelope ndf emits
    no per-node ``-ndf`` token — owned AND foreign decls go bare (the
    ``model -ndf`` directive supplies it)."""
    fem = _make_mixed_ndf_partitioned_fem()
    fem.add_node_constraints([_interface_tie()])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)       # envelope 3
    # Truss on BOTH pgs → every node infers 3 == envelope.
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="shell_pg", A=0.01, material=mat)
    ops.element.Truss(pg="solid_pg", A=0.01, material=mat)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    per_rank = _per_rank_calls(rec)

    for rank, calls in per_rank.items():
        for tag, kwargs in _node_emits_by_tag(calls).items():
            assert "ndf" not in kwargs, (
                f"rank {rank} node {tag}: emitted -ndf {kwargs.get('ndf')!r} "
                f"although every node infers the envelope (3) → must be elided"
            )
