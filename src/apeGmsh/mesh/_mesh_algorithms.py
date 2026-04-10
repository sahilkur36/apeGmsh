"""
Algorithm / optimise constants + ``_normalize_algorithm`` helper.

Split out of the old monolithic ``Mesh.py`` so the generation
sub-composite can import the normaliser without pulling in the
main :class:`~apeGmsh.mesh.Mesh.Mesh` module.

The public names here are re-exported from ``apeGmsh.mesh.Mesh`` and
from the top-level ``apeGmsh`` package for backwards-compatible
imports.
"""
from __future__ import annotations

from enum import IntEnum


# ---------------------------------------------------------------------------
# Algorithm constants
# ---------------------------------------------------------------------------

class Algorithm2D(IntEnum):
    """
    2-D meshing algorithm selector (legacy IntEnum form).

    Prefer passing a string name to
    :meth:`apeGmsh.mesh.Mesh._Generation.set_algorithm` — see
    :data:`ALGORITHM_2D` and :class:`MeshAlgorithm2D` for the canonical
    names and the accepted aliases.
    """
    MESH_ADAPT             = 1
    AUTOMATIC              = 2
    INITIAL_MESH_ONLY      = 3
    DELAUNAY               = 5
    FRONTAL_DELAUNAY       = 6
    BAMG                   = 7
    FRONTAL_DELAUNAY_QUADS = 8
    PACKING_PARALLELOGRAMS = 9
    QUASI_STRUCTURED_QUAD  = 11


class Algorithm3D(IntEnum):
    """
    3-D meshing algorithm selector (legacy IntEnum form).
    """
    DELAUNAY          = 1
    INITIAL_MESH_ONLY = 3
    FRONTAL           = 4
    MMG3D             = 7
    R_TREE            = 9
    HXT               = 10


ALGORITHM_2D: dict[str, int] = {
    # canonical
    "mesh_adapt":             1,
    "automatic":              2,
    "initial_mesh_only":      3,
    "delaunay":               5,
    "frontal_delaunay":       6,
    "bamg":                   7,
    "frontal_delaunay_quads": 8,
    "packing_parallelograms": 9,
    "quasi_structured_quad":  11,
    # aliases
    "auto":                   2,
    "default":                2,
    "meshadapt":              1,
    "front":                  6,
    "frontal":                6,
    "quad":                   8,
    "quads":                  8,
    "tri":                    6,
    "tris":                   6,
    "pack":                   9,
    "packing":                9,
    "quasi_structured":       11,
    "qsq":                    11,
}

ALGORITHM_3D: dict[str, int] = {
    # canonical
    "delaunay":          1,
    "initial_mesh_only": 3,
    "frontal":           4,
    "mmg3d":             7,
    "r_tree":            9,
    "hxt":               10,
    # aliases
    "auto":              10,
    "default":           10,
    "automatic":         10,
    "rtree":             9,
    "mmg":               7,
}


class MeshAlgorithm2D:
    """Canonical 2-D algorithm names as string constants (IDE autocomplete)."""
    MESH_ADAPT             = "mesh_adapt"
    AUTOMATIC              = "automatic"
    AUTO                   = "automatic"
    INITIAL_MESH_ONLY      = "initial_mesh_only"
    DELAUNAY               = "delaunay"
    FRONTAL_DELAUNAY       = "frontal_delaunay"
    BAMG                   = "bamg"
    FRONTAL_DELAUNAY_QUADS = "frontal_delaunay_quads"
    QUAD                   = "frontal_delaunay_quads"
    QUADS                  = "frontal_delaunay_quads"
    PACKING_PARALLELOGRAMS = "packing_parallelograms"
    QUASI_STRUCTURED_QUAD  = "quasi_structured_quad"
    QSQ                    = "quasi_structured_quad"


class MeshAlgorithm3D:
    """Canonical 3-D algorithm names as string constants (IDE autocomplete)."""
    DELAUNAY          = "delaunay"
    INITIAL_MESH_ONLY = "initial_mesh_only"
    FRONTAL           = "frontal"
    MMG3D             = "mmg3d"
    R_TREE            = "r_tree"
    HXT               = "hxt"
    AUTO              = "hxt"


class OptimizeMethod:
    """Mesh optimisation method names — use with ``g.mesh.generation.optimize``."""
    DEFAULT                    = ""
    NETGEN                     = "Netgen"
    HIGH_ORDER                 = "HighOrder"
    HIGH_ORDER_ELASTIC         = "HighOrderElastic"
    HIGH_ORDER_FAST_CURVING    = "HighOrderFastCurving"
    LAPLACE_2D                 = "Laplace2D"
    RELOCATE_2D                = "Relocate2D"
    RELOCATE_3D                = "Relocate3D"
    QUAD_QUASI_STRUCTURED      = "QuadQuasiStructured"
    UNTANGLE_MESH_GEOMETRY     = "UntangleMeshGeometry"


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

def _normalize_algorithm(alg, dim: int) -> int:
    """
    Resolve ``alg`` (str | int | IntEnum) into the integer code Gmsh wants.

    Raises ``ValueError`` with the full list of canonical names on an
    unknown string, so users get immediate, actionable feedback instead
    of a silent Gmsh error later.
    """
    if isinstance(alg, bool):       # bool is an int subclass — reject early
        raise TypeError(f"set_algorithm: algorithm must not be a bool, got {alg!r}")
    if isinstance(alg, int):
        return int(alg)
    if isinstance(alg, str):
        table = ALGORITHM_2D if dim == 2 else ALGORITHM_3D
        key = alg.strip().lower().replace("-", "_").replace(" ", "_")
        if key in table:
            return table[key]
        canonical = sorted({
            name for name in table
            if name not in {"auto", "default", "automatic",
                            "meshadapt", "front", "quad", "quads",
                            "tri", "tris", "pack", "packing",
                            "quasi_structured", "qsq", "rtree", "mmg"}
        })
        raise ValueError(
            f"set_algorithm: unknown {dim}-D algorithm {alg!r}.\n"
            f"Known names (dim={dim}): {canonical}"
        )
    raise TypeError(
        f"set_algorithm: algorithm must be str, int, or IntEnum — got {type(alg).__name__}"
    )
