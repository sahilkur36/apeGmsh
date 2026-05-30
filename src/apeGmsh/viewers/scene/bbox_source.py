"""Gmsh → canonical :class:`BBox` provider (ADR 0045 S1).

The single place a ``gmsh.model.getBoundingBox`` result is turned into
the canonical :class:`~apeGmsh.viewers.scene_ir.BBox` value type. Every
viewer site that needs an entity or whole-model bounding box routes
through here, so there is one producer per source (INV-2: one bbox
*type*, per-substrate providers). BREP / mesh scenes are gmsh-sourced;
the results substrate (no gmsh model) gets its own provider in S4.
"""
from __future__ import annotations

import gmsh

from ..scene_ir import BBox


def gmsh_bbox(dim: int, tag: int) -> BBox:
    """Canonical :class:`BBox` for one gmsh entity ``(dim, tag)``."""
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    return BBox((xmin, ymin, zmin), (xmax, ymax, zmax))


def gmsh_model_bbox() -> BBox:
    """Canonical :class:`BBox` of the whole gmsh model (``-1, -1``)."""
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    return BBox((xmin, ymin, zmin), (xmax, ymax, zmax))


__all__ = ["gmsh_bbox", "gmsh_model_bbox"]
