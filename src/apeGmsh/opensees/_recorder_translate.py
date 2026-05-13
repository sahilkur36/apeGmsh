"""Canonical-name → OpenSees recorder token translation (Phase 9).

The :class:`~apeGmsh.opensees.recorder.RecorderDeclaration` carries
components in canonical vocabulary (``"displacement_x"``,
``"reaction_force_y"``, ``"axial_force"``, …). Emitting them as
OpenSees ``recorder`` commands requires translation to the native
token shapes (``"disp"`` + ``-dof 1``, ``"reaction"`` + ``-dof 5``,
``"section force"`` per integration point, …).

This module owns the translation tables for **node-level** components
(Phase 9 commit 3a). Element-level translation (elements / gauss /
line_stations) lands in commit 3b and lifts the heavier translation
logic from :mod:`apeGmsh.results.spec._emit`.

The :class:`Node`, :class:`Element`, and :class:`MPCO` typed
primitives in :mod:`apeGmsh.opensees.recorder` continue to take raw
OpenSees tokens directly (no translation needed); only the unified
``RecorderDeclaration`` consumed by
:func:`~apeGmsh.opensees._internal.build.emit_recorder_spec` flows
through these tables.
"""
from __future__ import annotations

from typing import Optional


# =====================================================================
# Per-axis DOF index tables
# =====================================================================

_TRANS_AXIS_TO_DOF: dict[str, int] = {"x": 1, "y": 2, "z": 3}
_ROT_AXIS_TO_DOF: dict[str, int] = {"x": 4, "y": 5, "z": 6}


# =====================================================================
# Canonical-prefix → (ops_recorder_token, axis_kind) table
# =====================================================================
#
# axis_kind ∈ {"trans", "rot"} selects which DOF table the axis
# suffix is read against. ``displacement_z`` → ("disp", 3); a 2-D
# ``rotation_z`` → ("disp", 6) (z-rotation in ndf=3 lives at DOF 3,
# but OpenSees promotes to the 3-D convention by default; emit-time
# clipping handles ndm/ndf consistency via the canonical-name set
# the bridge already validated).

_NODAL_PREFIX_TABLE: dict[str, tuple[str, str]] = {
    "displacement":           ("disp",      "trans"),
    "rotation":               ("disp",      "rot"),
    "velocity":               ("vel",       "trans"),
    "angular_velocity":       ("vel",       "rot"),
    "acceleration":           ("accel",     "trans"),
    "angular_acceleration":   ("accel",     "rot"),
    "displacement_increment": ("incrDisp",  "trans"),
    "reaction_force":         ("reaction",  "trans"),
    "reaction_moment":        ("reaction",  "rot"),
    # OpenSees ``unbalance`` returns residual nodal forces.
    "force":                  ("unbalance", "trans"),
    "moment":                 ("unbalance", "rot"),
}


# =====================================================================
# Scalar canonical names (no axis suffix)
# =====================================================================
#
# ``-1`` is a sentinel meaning "OpenSees infers the DOF" — used for
# pressure (the formulation-dependent pressure DOF). Real emitters
# decide based on the bridge's ``ndf`` whether to emit ``-dof <pdof>``
# explicitly or rely on the OpenSees default.

_NODAL_SCALAR_TABLE: dict[str, tuple[str, int]] = {
    "pore_pressure": ("pressure", -1),
}


# =====================================================================
# Public translation API
# =====================================================================


def node_component_to_ops(canonical: str) -> Optional[tuple[str, int]]:
    """Map a canonical node component to ``(ops_recorder_token, dof)``.

    Returns ``None`` if ``canonical`` is not a recognized node-level
    component (caller should fall through to error).

    Examples
    --------
    >>> node_component_to_ops("displacement_x")
    ('disp', 1)
    >>> node_component_to_ops("rotation_z")
    ('disp', 6)
    >>> node_component_to_ops("reaction_force_y")
    ('reaction', 2)
    >>> node_component_to_ops("pore_pressure")
    ('pressure', -1)
    >>> node_component_to_ops("bogus") is None
    True
    """
    if canonical in _NODAL_SCALAR_TABLE:
        return _NODAL_SCALAR_TABLE[canonical]
    if "_" not in canonical:
        return None
    prefix, axis = canonical.rsplit("_", 1)
    entry = _NODAL_PREFIX_TABLE.get(prefix)
    if entry is None:
        return None
    ops_token, axis_kind = entry
    dof_table = _TRANS_AXIS_TO_DOF if axis_kind == "trans" else _ROT_AXIS_TO_DOF
    if axis not in dof_table:
        return None
    return (ops_token, dof_table[axis])


def group_node_components_by_ops_token(
    components: tuple[str, ...],
) -> dict[str, tuple[int, ...]]:
    """Group node components by their OpenSees token.

    Components sharing the same ``ops_recorder_token`` collapse into
    one ``recorder Node`` line with a ``-dof`` list of all their DOFs.
    For example, ``("displacement_x", "displacement_y", "reaction_force_z")``
    produces:

        {"disp": (1, 2), "reaction": (3,)}

    Returns
    -------
    Mapping from ops token to ordered tuple of DOFs (1-based, OpenSees
    convention). Order within each group preserves component
    declaration order; ops tokens themselves are returned in
    discovery order.

    Raises
    ------
    ValueError
        If any component is not a recognized node-level canonical
        (caller is responsible for validating against the canonical
        vocabulary upstream; this raise is a defensive backstop).
    """
    grouped: dict[str, list[int]] = {}
    for comp in components:
        translated = node_component_to_ops(comp)
        if translated is None:
            raise ValueError(
                f"_recorder_translate.group_node_components_by_ops_token: "
                f"{comp!r} is not a recognized node-level canonical."
            )
        ops_token, dof = translated
        grouped.setdefault(ops_token, []).append(dof)
    # Deduplicate DOFs per token while preserving insertion order.
    return {
        token: tuple(dict.fromkeys(dofs)) for token, dofs in grouped.items()
    }


# =====================================================================
# Element-level keyword resolution (Phase 9 commit 3b)
# =====================================================================
#
# Element-level recorders (``recorder Element ...``) take a *single*
# response token (or multi-token phrase like ``section force``) that
# applies to the whole record. The token depends on which component
# family the record carries:
#
#   - ``"stress_*"`` / shell membrane/curvature etc. → ``"stresses"``
#   - ``"strain_*"`` / shell strain etc.            → ``"strains"``
#   - ``"axial_force"`` etc. (line-stations)        → ``"section"`` + ``"force"``
#   - ``"nodal_resisting_force_*"`` (global frame)  → ``"globalForce"``
#   - ``"nodal_resisting_force_local_*"`` (local)   → ``"localForce"``
#
# The work-conjugate check rejects records that mix families
# (e.g. ``stress_xx`` + ``strain_yy`` in one ``gauss`` record) — those
# can't share a single ``ops.eleResponse`` call.


_ELEMENT_LEVEL_CATEGORIES: frozenset[str] = frozenset({
    "elements", "line_stations", "gauss",
})


def element_record_response_tokens(
    category: str,
    components: tuple[str, ...],
    *,
    record_name: str | None = None,
) -> Optional[tuple[str, ...]]:
    """Resolve the ``recorder Element`` response phrase for a record.

    Returns the response tokens as a tuple — usually one token
    (``("stresses",)``) but occasionally two (``("section", "force")``)
    that OpenSees parses as a multi-word response phrase. Returns
    ``None`` when no components route through the category's
    topology (caller skips silently).

    Raises ``ValueError`` if components mix work-conjugate families
    (stress + strain in one gauss record, global + local frame in one
    elements record).
    """
    if category not in _ELEMENT_LEVEL_CATEGORIES:
        raise ValueError(
            f"_recorder_translate.element_record_response_tokens: "
            f"category {category!r} is not element-level."
        )
    topology = {
        "gauss":         None,
        "line_stations": "line_stations",
        "elements":      "nodal_forces",
    }[category]

    # Lazy import to avoid a top-level cycle with the response catalog
    # module.
    from ._response_catalog import gauss_keyword_for_canonical

    keywords: set[str] = set()
    for comp in components:
        kw = gauss_keyword_for_canonical(comp, topology=topology)
        if kw is not None:
            keywords.add(kw)
    if not keywords:
        return None
    if len(keywords) > 1:
        rec_label = f"record {record_name!r}" if record_name else "record"
        raise ValueError(
            f"_recorder_translate.element_record_response_tokens: "
            f"{rec_label} (category={category!r}) mixes work-conjugate "
            f"families ({sorted(keywords)}); split into separate records "
            "(one per ops keyword)."
        )
    keyword = next(iter(keywords))
    # The line_stations keyword "section.force" is OpenSees-emitted as
    # two tokens ("section" "force"); same for ``section.deformation``.
    if "." in keyword:
        return tuple(keyword.split("."))
    return (keyword,)
