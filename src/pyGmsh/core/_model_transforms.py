from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, DimTag, TagsLike

if TYPE_CHECKING:
    from .Model import Model


class _TransformsMixin:
    """Transform and extrusion/revolution methods extracted from Model."""

    # ------------------------------------------------------------------
    # Transforms  (all return self for chaining)
    # ------------------------------------------------------------------

    def translate(
        self,
        tags: TagsLike,
        dx: float, dy: float, dz: float,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Translate entities by (dx, dy, dz).

        Example
        -------
        ``g.model.translate(box, 5, 0, 0)``
        """
        gmsh.model.occ.translate(self._as_dimtags(tags, dim), dx, dy, dz)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"translate by ({dx}, {dy}, {dz})")
        return self

    def rotate(
        self,
        tags  : TagsLike,
        angle : float,
        *,
        ax: float = 0.0, ay: float = 0.0, az: float = 1.0,
        cx: float = 0.0, cy: float = 0.0, cz: float = 0.0,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Rotate entities around an axis through (cx, cy, cz) with direction
        (ax, ay, az) by ``angle`` radians.

        Example
        -------
        ``g.model.rotate(box, math.pi / 4, az=1)``
        """
        gmsh.model.occ.rotate(
            self._as_dimtags(tags, dim),
            cx, cy, cz,
            ax, ay, az,
            angle,
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._log(
            f"rotate {math.degrees(angle):.2f}\u00b0 about axis=({ax},{ay},{az}) "
            f"through ({cx},{cy},{cz})"
        )
        return self

    def scale(
        self,
        tags: TagsLike,
        sx: float, sy: float, sz: float,
        *,
        cx: float = 0.0, cy: float = 0.0, cz: float = 0.0,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Scale (dilate) entities by (sx, sy, sz) from centre (cx, cy, cz).

        Example
        -------
        ``g.model.scale(box, 2, 2, 2)``   # uniform double
        """
        gmsh.model.occ.dilate(
            self._as_dimtags(tags, dim),
            cx, cy, cz,
            sx, sy, sz,
        )
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"scale ({sx},{sy},{sz}) about ({cx},{cy},{cz})")
        return self

    def mirror(
        self,
        tags: TagsLike,
        a: float, b: float, c: float, d: float,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> Model:
        """
        Mirror entities through the plane ax + by + cz + d = 0.

        Example
        -------
        ``g.model.mirror(box, 1, 0, 0, 0)``   # reflect through YZ plane
        """
        gmsh.model.occ.mirror(self._as_dimtags(tags, dim), a, b, c, d)
        if sync:
            gmsh.model.occ.synchronize()
        self._log(f"mirror through plane {a}x + {b}y + {c}z + {d} = 0")
        return self

    def copy(
        self,
        tags: TagsLike,
        *,
        dim : int  = 3,
        sync: bool = True,
    ) -> list[Tag]:
        """
        Duplicate entities.  Returns the tags of the new copies.

        Example
        -------
        ``copies = g.model.copy([box, sphere])``
        """
        new_dimtags = gmsh.model.occ.copy(self._as_dimtags(tags, dim))
        if sync:
            gmsh.model.occ.synchronize()
        new_tags = [t for _, t in new_dimtags]
        self._log(f"copy \u2192 new tags {new_tags}")
        return new_tags

    # ------------------------------------------------------------------
    # Extrusion / Revolution  (generative sweep ops)
    # ------------------------------------------------------------------

    def extrude(
        self,
        tags: TagsLike,
        dx: float, dy: float, dz: float,
        *,
        dim          : int             = 2,
        num_elements : list[int] | None = None,
        heights      : list[float] | None = None,
        recombine    : bool            = False,
        sync         : bool            = True,
    ) -> list[DimTag]:
        """
        Linear extrusion \u2014 sweeps entities along (dx, dy, dz).

        Creates new geometry one dimension up: point \u2192 curve,
        curve \u2192 surface, surface \u2192 volume.

        Parameters
        ----------
        tags : entities to extrude.
        dx, dy, dz : extrusion vector.
        dim : default dimension for bare integer tags (default 2).
        num_elements : structured layer counts, e.g. ``[10]`` for
            10 layers.  Empty list (default) = unstructured.
        heights : relative heights per layer, e.g. ``[0.3, 0.7]``.
            Must sum to 1.0 when provided.  Empty = uniform layers.
        recombine : if True, produce hex/quad elements instead of
            tet/tri (requires structured layers).
        sync : synchronise OCC kernel after extrusion (default True).

        Returns
        -------
        list[DimTag]
            All generated (dim, tag) pairs.  For a surface \u2192 volume
            extrusion the list contains the top face, the volume, and
            the lateral faces \u2014 index into it to assign physical groups.

        Example
        -------
        ::

            surf = g.model.add_plane_surface(loop)
            out  = g.model.extrude(surf, 0, 0, 3.0, num_elements=[10])
            # out[0] = (2, top_face), out[1] = (3, volume), ...
        """
        dt = self._as_dimtags(tags, dim)
        ne = num_elements if num_elements is not None else []
        ht = heights if heights is not None else []
        result: list[tuple[int, int]] = gmsh.model.occ.extrude(
            dt, dx, dy, dz,
            numElements=ne,
            heights=ht,
            recombine=recombine,
        )
        if sync:
            gmsh.model.occ.synchronize()
        for d, t in result:
            self._register(d, t, None, 'extrude')
        self._log(
            f"extrude({dt}, ({dx},{dy},{dz})) \u2192 {len(result)} entities"
        )
        return result

    def revolve(
        self,
        tags  : TagsLike,
        angle : float,
        *,
        x : float = 0.0, y : float = 0.0, z : float = 0.0,
        ax: float = 0.0, ay: float = 0.0, az: float = 1.0,
        dim          : int             = 2,
        num_elements : list[int] | None = None,
        heights      : list[float] | None = None,
        recombine    : bool            = False,
        sync         : bool            = True,
    ) -> list[DimTag]:
        """
        Revolution \u2014 sweeps entities around an axis.

        Parameters
        ----------
        tags : entities to revolve.
        angle : sweep angle in radians (2\u03c0 for full revolution).
        x, y, z : point on the rotation axis.
        ax, ay, az : direction vector of the rotation axis.
        dim : default dimension for bare integer tags (default 2).
        num_elements, heights, recombine : same as :meth:`extrude`.
        sync : synchronise OCC kernel (default True).

        Returns
        -------
        list[DimTag]
            All generated (dim, tag) pairs.

        Example
        -------
        ::

            # Revolve a cross-section 360\u00b0 around the Y axis
            out = g.model.revolve(profile, 2 * math.pi, ay=1)
        """
        dt = self._as_dimtags(tags, dim)
        ne = num_elements if num_elements is not None else []
        ht = heights if heights is not None else []
        result: list[tuple[int, int]] = gmsh.model.occ.revolve(
            dt, x, y, z, ax, ay, az, angle,
            numElements=ne,
            heights=ht,
            recombine=recombine,
        )
        if sync:
            gmsh.model.occ.synchronize()
        for d, t in result:
            self._register(d, t, None, 'revolve')
        self._log(
            f"revolve({dt}, angle={math.degrees(angle):.1f}\u00b0, "
            f"axis=({ax},{ay},{az}) through ({x},{y},{z})) "
            f"\u2192 {len(result)} entities"
        )
        return result
