"""``/opensees/names`` sidecar — bridge-side name aliases persisted.

A name registered on the bridge (``ops.timeSeries.Linear(name="dead")``)
is an *alias*, not structure: it never enters the frozen primitive, the
lineage hash, or the canonical model identity.  But a name is useful
downstream — resolving ``"dead"`` back to its kind+tag in
``OpenSeesModel`` / ``Results`` / the viewer — so it is persisted as a
**sidecar** table here.

Layout (only written when at least one name is registered)::

    /opensees/names
        name   (vlen utf-8)  — the user-chosen alias
        kind   (vlen utf-8)  — the OpenSees command family
                               ("timeSeries", "uniaxialMaterial", ...)
        tag    (int64)       — the allocated per-kind tag

``names`` is in :data:`apeGmsh.opensees._internal.lineage.MODEL_HASH_EXCLUDED_CHILDREN`,
so relabelling never perturbs ``model_hash`` — same carve-out cuts /
sweeps / regions get (INV-4).  Empty ⇒ no group, preserving the
byte-equivalence of name-free files (mirrors ``write_cuts_into``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import h5py


__all__ = ["NAMES_GROUP", "write_names_into", "read_names"]


#: Group name under ``/opensees`` holding the alias table.
NAMES_GROUP: str = "names"


def write_names_into(
    f: "h5py.File",
    names: "Sequence[tuple[str, str, int]]",
    *,
    opensees_root: str = "opensees",
) -> None:
    """Write the ``(name, kind, tag)`` alias table under ``/opensees/names``.

    No-op when ``names`` is empty — the group is not created, so a
    model with no registered names is byte-identical to the pre-sidecar
    layout.  Records are written in caller order (the bridge sorts by
    name for determinism).
    """
    if not names:
        return

    import h5py

    grp = f.require_group(opensees_root)
    if NAMES_GROUP in grp:
        del grp[NAMES_GROUP]
    g = grp.create_group(NAMES_GROUP)

    str_dt = h5py.string_dtype(encoding="utf-8")
    g.create_dataset(
        "name", data=[n for n, _, _ in names], dtype=str_dt
    )
    g.create_dataset(
        "kind", data=[k for _, k, _ in names], dtype=str_dt
    )
    g.create_dataset(
        "tag", data=[int(t) for _, _, t in names], dtype="int64"
    )


def read_names(
    path: str,
    *,
    opensees_root: str = "/opensees",
) -> tuple[tuple[str, str, int], ...]:
    """Read the ``/opensees/names`` alias table.

    Returns an empty tuple when the file has no bridge zone or no names
    group (the common no-names case).  Uses the ``name in group`` probe
    per the repo's h5py optional-child convention — never ``Group.get``.
    """
    import h5py

    root = opensees_root.strip("/")
    with h5py.File(path, "r") as f:
        if root not in f:
            return ()
        grp = f[root]
        if NAMES_GROUP not in grp:
            return ()
        g = grp[NAMES_GROUP]
        raw_names = g["name"][()]
        raw_kinds = g["kind"][()]
        raw_tags = g["tag"][()]

    out: list[tuple[str, str, int]] = []
    for nm, kd, tg in zip(raw_names, raw_kinds, raw_tags):
        out.append((_as_str(nm), _as_str(kd), int(tg)))
    return tuple(out)


def _as_str(value: object) -> str:
    """Decode an h5py vlen-string element to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
