"""
NotebookPreview — inline interactive 3D preview for Colab / Jupyter.

Sibling of :mod:`apeGmsh.viz.Inspect`, :mod:`apeGmsh.viz.Plot`, and
:mod:`apeGmsh.viz.VTKExport`. Zero Qt dependency — renders via plotly's
``fig.show()`` which displays inline in Colab, JupyterLab, and VS Code
notebooks, and opens a browser tab in plain Python scripts.

Scene composition reuses the pure-function scene builders under
``apeGmsh.viewers.scene.*`` — so the data (points, cells, entity_tag
per cell) is identical to what ``ModelViewer`` / ``MeshViewer`` show.
Only the *render surface* differs: off-screen PyVista → triangulated
surfaces → Plotly ``Mesh3d`` / ``Scatter3d`` traces → self-contained
WebGL HTML with native hover tooltips.

Usage::

    import apeGmsh as g
    # Build geometry + mesh...
    g.model.preview()            # BRep, hover shows dim=D tag=T
    g.mesh.preview()             # Mesh, hover shows dim=D tag=T
    apeGmsh.preview(g)           # top-level convenience

Notes
-----
Requires ``plotly``. In Colab it's pre-installed; locally run
``pip install plotly``. PyVista is already a hard dependency of the
viewer stack.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from apeGmsh._types import DimTag


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "apeGmsh.preview / g.model.preview / g.mesh.preview require "
            "`plotly`. Install with `pip install plotly`. "
            "Plotly ships pre-installed in Google Colab."
        ) from err


def _theme_colors() -> dict[str, str]:
    """Palette colors as hex strings for the active theme."""
    from apeGmsh.viewers.ui.theme import THEME
    p = THEME.current

    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    return {
        "dim0": rgb_to_hex(p.dim_pt),
        "dim1": rgb_to_hex(p.dim_crv),
        "dim2": rgb_to_hex(p.dim_srf),
        "dim3": rgb_to_hex(p.dim_vol),
        "bg":   p.bg_top,
        "text": p.text,
        "origin_marker": p.origin_marker_color,
    }


# ======================================================================
# PG + label lookup
# ======================================================================

def _build_pg_label_lookup() -> tuple[
    dict[tuple[int, int], str],
    dict[tuple[int, int], str],
]:
    """Return ``(pg_names, labels)`` keyed by ``(dim, entity_tag)``.

    - ``pg_names``: user-facing physical groups (``_label:``-prefixed
      PGs are excluded); when an entity sits in multiple PGs the names
      are joined with ``", "``.
    - ``labels``: Tier 1 labels from ``apeGmsh.core.Labels`` — one entry
      per labelled entity, prefix stripped for display.

    Both dicts are empty for entities that carry no PG / no label.
    """
    import gmsh
    try:
        from apeGmsh.core.Labels import is_label_pg, strip_prefix
    except Exception:  # pragma: no cover — Labels is a core module
        def is_label_pg(name: str) -> bool:
            return name.startswith("_label:")

        def strip_prefix(name: str) -> str:
            return name[len("_label:"):] if name.startswith("_label:") else name

    pg_names: dict[tuple[int, int], list[str]] = {}
    labels: dict[tuple[int, int], str] = {}

    for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
        except Exception:
            continue
        if not name:
            continue
        try:
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
        except Exception:
            continue
        if is_label_pg(name):
            bare = strip_prefix(name)
            for t in ent_tags:
                labels[(pg_dim, int(t))] = bare
        else:
            for t in ent_tags:
                pg_names.setdefault((pg_dim, int(t)), []).append(name)

    pg_joined = {k: ", ".join(v) for k, v in pg_names.items()}
    return pg_joined, labels


# ======================================================================
# VTK → Plotly conversion helpers
# ======================================================================

def _hover_template() -> str:
    """Hover template string that renders dim, tag, pg, and label."""
    return (
        "dim=%{customdata[0]}, tag=%{customdata[1]}"
        "<br>pg: %{customdata[2]}<br>label: %{customdata[3]}"
        "<extra></extra>"
    )


def _cell_customdata(
    dim: int,
    tags: np.ndarray,
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
) -> list[list[Any]]:
    """Build per-cell customdata rows: [dim, tag, pg_name, label_name]."""
    rows: list[list[Any]] = []
    for t in tags:
        key = (dim, int(t))
        pg = pg_lookup.get(key, "—")
        lbl = label_lookup.get(key, "—")
        rows.append([dim, int(t), pg, lbl])
    return rows


def _dim2_surface_to_mesh3d(
    grid,
    name: str,
    color: str,
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
):
    """Convert a PyVista UnstructuredGrid of dim=2 cells → plotly Mesh3d."""
    import plotly.graph_objects as go
    # Extract as PolyData then triangulate everything (quads → 2 tris).
    surf = grid.extract_surface().triangulate()
    if surf.n_cells == 0:
        return None
    faces = np.asarray(surf.faces).reshape(-1, 4)     # [3, i, j, k, 3, i, j, k, ...]
    i, j, k = faces[:, 1], faces[:, 2], faces[:, 3]
    pts = np.asarray(surf.points)
    tags = surf.cell_data.get("entity_tag", None)
    if tags is None:
        tags = np.full(len(i), -1, dtype=np.int64)
    tags = np.asarray(tags)
    customdata = _cell_customdata(2, tags, pg_lookup, label_lookup)
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=1.0,
        name=name,
        showlegend=True,
        customdata=customdata,
        hovertemplate=_hover_template(),
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
    )


def _dim3_volume_to_mesh3d(
    grid,
    name: str,
    color: str,
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
):
    """Convert a dim=3 UnstructuredGrid → surface → triangles → plotly Mesh3d.

    Only external faces render (plotly's Mesh3d is a surface mesh).
    """
    import plotly.graph_objects as go
    surf = grid.extract_surface().triangulate()
    if surf.n_cells == 0:
        return None
    faces = np.asarray(surf.faces).reshape(-1, 4)
    i, j, k = faces[:, 1], faces[:, 2], faces[:, 3]
    pts = np.asarray(surf.points)
    tags = surf.cell_data.get("entity_tag", None)
    if tags is None:
        tags = np.full(len(i), -1, dtype=np.int64)
    tags = np.asarray(tags)
    customdata = _cell_customdata(3, tags, pg_lookup, label_lookup)
    return go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=1.0,
        name=name,
        showlegend=True,
        customdata=customdata,
        hovertemplate=_hover_template(),
        flatshading=True,
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
    )


def _dim1_lines_to_scatter3d(
    grid,
    name: str,
    color: str,
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
):
    """Convert a dim=1 grid → plotly Scatter3d line trace.

    Handles both ``pv.PolyData`` (connectivity on ``.lines``) and
    ``pv.UnstructuredGrid`` (connectivity on ``.cells``). For polyline
    cells with ``n > 2`` vertices, draws every consecutive pair so
    curved curves stay curved. ``None``-separated x/y/z sequences
    render each segment independently.
    """
    import plotly.graph_objects as go

    pts = np.asarray(grid.points)

    # PolyData stores line cells on ``.lines``; UnstructuredGrid on ``.cells``.
    cells_flat = getattr(grid, "lines", None)
    if cells_flat is None or len(cells_flat) == 0:
        cells_flat = getattr(grid, "cells", None)
    if cells_flat is None or len(cells_flat) == 0:
        return None
    cells = np.asarray(cells_flat)

    tags = np.asarray(grid.cell_data.get(
        "entity_tag", np.full(grid.n_cells, -1, dtype=np.int64)
    ))

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    cdata: list[list[Any]] = []

    offset = 0
    cell_idx = 0
    while offset < len(cells):
        n = int(cells[offset])
        indices = [int(cells[offset + 1 + k]) for k in range(n)]
        tag = int(tags[cell_idx]) if cell_idx < len(tags) else -1
        pg = pg_lookup.get((1, tag), "—")
        lbl = label_lookup.get((1, tag), "—")
        row = [1, tag, pg, lbl]
        # Draw one segment per consecutive pair so polyline curves
        # aren't collapsed to their chord.
        for p0, p1 in zip(indices[:-1], indices[1:]):
            xs += [pts[p0, 0], pts[p1, 0], None]
            ys += [pts[p0, 1], pts[p1, 1], None]
            zs += [pts[p0, 2], pts[p1, 2], None]
            cdata += [row, row, row]
        offset += n + 1
        cell_idx += 1

    if not xs:
        return None

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=4),
        name=name,
        showlegend=True,
        customdata=cdata,
        hovertemplate=_hover_template(),
    )


def _dim0_points_to_scatter3d(
    name: str,
    color: str,
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
):
    """Plotly ``Scatter3d`` markers for BRep dim=0 points.

    Matches the Qt model viewer: prefers the meshed node location
    (``gmsh.model.mesh.getNodes(dim=0, tag=tag)``) so un-meshed
    construction points are filtered out. Falls back to the BRep
    parametric location if the model hasn't been meshed yet.
    """
    import gmsh
    import plotly.graph_objects as go
    import numpy as np

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    tags: list[int] = []
    for _, tag in gmsh.model.getEntities(dim=0):
        coord = None
        try:
            ntags, ncoords, _ = gmsh.model.mesh.getNodes(dim=0, tag=tag)
            if len(ntags) > 0:
                coord = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)[0]
        except Exception:
            pass
        if coord is None:
            # Not meshed — use the BRep parametric location as a fallback
            try:
                coord = gmsh.model.getValue(0, tag, [])
            except Exception:
                continue
        xs.append(float(coord[0]))
        ys.append(float(coord[1]))
        zs.append(float(coord[2]))
        tags.append(int(tag))

    if not xs:
        return None

    customdata = _cell_customdata(0, np.asarray(tags), pg_lookup, label_lookup)
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=5, color=color),
        name=name,
        showlegend=True,
        customdata=customdata,
        hovertemplate=_hover_template(),
    )


# ======================================================================
# Scene wrappers — reuse existing scene builders
# ======================================================================

def _unshifted_grid(grid: Any, shift: np.ndarray) -> Any:
    """Return a copy of ``grid`` with ``shift`` added back to every point.

    The scene builders subtract the model's bbox center from every
    coordinate for numerical stability. We undo that here so plotly
    axis labels show real world coords — and, crucially, so the grid
    coords match the dim=0 point coords (which we query from Gmsh
    directly and are therefore un-shifted).
    """
    if not shift.any():
        return grid
    g = grid.copy()
    g.points = np.asarray(g.points, dtype=np.float64) + shift
    return g


def _build_brep_traces(dims: Iterable[int]) -> list[Any]:
    """Build BRep scene via viewers.scene.brep_scene and convert to plotly traces."""
    import gmsh
    import pyvista as pv
    from apeGmsh.viewers.scene.brep_scene import build_brep_scene

    gmsh.model.occ.synchronize()
    plotter = pv.Plotter(off_screen=True)
    try:
        registry = build_brep_scene(plotter, list(dims))
    except Exception as err:
        raise RuntimeError(
            f"Failed to build BRep scene for preview: {err}"
        ) from err

    shift = np.asarray(registry.origin_shift, dtype=np.float64)
    colors = _theme_colors()
    pg_lookup, label_lookup = _build_pg_label_lookup()
    traces: list[Any] = []
    for dim, grid in registry.dim_meshes.items():
        if grid is None or grid.n_cells == 0:
            continue
        world_grid = _unshifted_grid(grid, shift)
        trace = _convert_grid(world_grid, dim, colors, pg_lookup, label_lookup)
        if trace is not None:
            traces.append(trace)
    return traces


def _build_mesh_traces(
    dims: Iterable[int], *, show_nodes: bool = True,
) -> list[Any]:
    """Build mesh scene via viewers.scene.mesh_scene and convert to plotly traces."""
    import gmsh
    import pyvista as pv
    from apeGmsh.viewers.scene.mesh_scene import build_mesh_scene

    gmsh.model.occ.synchronize()
    plotter = pv.Plotter(off_screen=True)
    try:
        scene = build_mesh_scene(plotter, list(dims))
    except Exception as err:
        raise RuntimeError(
            f"Failed to build mesh scene for preview: {err}"
        ) from err

    shift = np.asarray(scene.registry.origin_shift, dtype=np.float64)
    colors = _theme_colors()
    pg_lookup, label_lookup = _build_pg_label_lookup()
    traces: list[Any] = []
    for dim, grid in scene.registry.dim_meshes.items():
        if grid is None or grid.n_cells == 0:
            continue
        world_grid = _unshifted_grid(grid, shift)
        trace = _convert_grid(world_grid, dim, colors, pg_lookup, label_lookup)
        if trace is not None:
            traces.append(trace)

    if show_nodes:
        node_trace = _mesh_nodes_to_scatter3d(
            scene.node_coords,
            scene.node_tags,
            shift,
            colors["dim0"],
        )
        if node_trace is not None:
            traces.append(node_trace)

    return traces


def _convert_grid(
    grid,
    dim: int,
    colors: dict[str, str],
    pg_lookup: dict[tuple[int, int], str],
    label_lookup: dict[tuple[int, int], str],
) -> Any | None:
    """Dispatch a per-dim grid to its plotly converter.

    Note: for ``dim=0`` we ignore the grid (it contains sphere-glyph
    tessellations, not point coords) and query Gmsh directly.

    Trace names encode the dimension explicitly ("Points (dim=0)",
    "Curves (dim=1)", ...) so the plotly legend doubles as a visibility
    filter: single-click a legend entry to hide it, double-click to
    isolate.
    """
    if dim == 0:
        return _dim0_points_to_scatter3d(
            "Points (dim=0)", colors["dim0"], pg_lookup, label_lookup,
        )
    if dim == 1:
        return _dim1_lines_to_scatter3d(
            grid, "Curves (dim=1)", colors["dim1"], pg_lookup, label_lookup,
        )
    if dim == 2:
        return _dim2_surface_to_mesh3d(
            grid, "Surfaces (dim=2)", colors["dim2"], pg_lookup, label_lookup,
        )
    if dim == 3:
        return _dim3_volume_to_mesh3d(
            grid, "Volumes (dim=3)", colors["dim3"], pg_lookup, label_lookup,
        )
    return None


def _mesh_nodes_to_scatter3d(
    node_coords: np.ndarray,
    node_tags: np.ndarray,
    shift: np.ndarray,
    color: str,
) -> Any | None:
    """Plotly Scatter3d trace for the mesh node cloud.

    ``node_coords`` comes from the mesh scene builder in shifted coords
    (origin subtracted for numerical stability); we add the shift back
    so the markers align with the un-shifted element traces.
    """
    if node_coords is None or len(node_coords) == 0:
        return None
    import plotly.graph_objects as go

    pts = np.asarray(node_coords, dtype=np.float64)
    if shift.any():
        pts = pts + shift
    tags = np.asarray(node_tags, dtype=np.int64) if node_tags is not None else None
    customdata = [[int(t)] for t in tags] if tags is not None else None
    hovertemplate = (
        "node=%{customdata[0]}<extra></extra>" if customdata is not None
        else "mesh node<extra></extra>"
    )
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=3, color=color),
        name="Mesh nodes",
        showlegend=True,
        customdata=customdata,
        hovertemplate=hovertemplate,
    )


# ======================================================================
# Public API
# ======================================================================

def _make_figure(traces: list[Any], title: str):
    import plotly.graph_objects as go
    colors = _theme_colors()
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            xaxis=dict(backgroundcolor=colors["bg"]),
            yaxis=dict(backgroundcolor=colors["bg"]),
            zaxis=dict(backgroundcolor=colors["bg"]),
            # Orthographic projection looking down +Z — XY plane view
            # by default, matching the convention for 2D structural
            # frames. Users can rotate in the browser and the projection
            # stays orthographic unless they change it.
            camera=dict(
                eye=dict(x=0.0, y=0.0, z=2.5),
                up=dict(x=0.0, y=1.0, z=0.0),
                center=dict(x=0.0, y=0.0, z=0.0),
                projection=dict(type="orthographic"),
            ),
        ),
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font=dict(color=colors["text"]),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig


def _display_figure(fig: Any) -> None:
    """Display the figure inline using the most robust path available.

    Tries in order:
    1. ``IPython.display.display`` with an ``HTML`` wrapper around the
       figure's self-contained HTML (CDN plotly.js). This works in VS
       Code / JupyterLab / classic Jupyter / Colab without requiring
       ``ipywidgets`` or any renderer configuration.
    2. ``fig.show()`` — plotly's native renderer dispatch. Opens a
       browser tab when no notebook is detected.
    """
    try:
        from IPython.display import display, HTML
        html_text = fig.to_html(include_plotlyjs="cdn", full_html=False)
        display(HTML(html_text))
        return
    except Exception:
        pass
    # Fallback: plain-Python / no IPython
    fig.show()


def _open_in_browser(fig: Any, title: str) -> str:
    """Write the figure to a temp HTML file and open the default browser.

    Returns the file path so the caller can print / log it. The temp
    file persists for the session so the browser tab keeps working
    after this function returns; the OS cleans up the temp dir.
    """
    import pathlib
    import tempfile
    import webbrowser

    html_text = fig.to_html(include_plotlyjs="cdn", full_html=True)
    # Use a prefix so the temp file is easy to spot if left behind.
    fd, path = tempfile.mkstemp(
        prefix="apeGmsh_preview_",
        suffix=".html",
        text=True,
    )
    import os
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(html_text)

    webbrowser.open(pathlib.Path(path).as_uri())
    return path


def preview_model(
    session: Any = None,
    *,
    dims: list[int] | None = None,
    browser: bool = False,
    return_fig: bool = False,
) -> Any:
    """Interactive WebGL preview of the BRep geometry.

    By default displays inline in the current notebook cell. Pass
    ``browser=True`` to open the preview in a new browser tab instead
    — useful when the notebook output is cluttered or you want a
    dedicated window.

    Parameters
    ----------
    session : apeGmsh session, optional
        Accepted for call-site symmetry with ``ModelViewer(session)``;
        not required — the scene builder reads Gmsh globals directly.
    dims : list of int, optional
        BRep dimensions to render (default: ``[0, 1, 2, 3]``).
    browser : bool
        If ``True``, write the preview to a temp HTML file and open the
        default web browser instead of rendering inline. The tab stays
        open after the Python call returns.
    return_fig : bool
        If ``True``, skip display entirely and return the raw
        :class:`plotly.graph_objects.Figure` instead. Useful for saving
        to HTML with ``fig.write_html('path.html')`` or composing a
        larger notebook layout. Takes precedence over ``browser``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        ``None`` unless ``return_fig=True``.
    """
    _require_plotly()
    traces = _build_brep_traces(dims or [0, 1, 2, 3])
    fig = _make_figure(traces, title="apeGmsh — BRep preview")
    if return_fig:
        return fig
    if browser:
        _open_in_browser(fig, "apeGmsh — BRep preview")
        return None
    _display_figure(fig)
    return None


def preview_mesh(
    session: Any = None,
    *,
    dims: list[int] | None = None,
    show_nodes: bool = True,
    browser: bool = False,
    return_fig: bool = False,
) -> Any:
    """Interactive WebGL preview of the mesh.

    Parameters match :func:`preview_model`; ``dims`` default is
    ``[1, 2, 3]`` since meshes rarely render dim=0 element cells.

    A dedicated "Mesh nodes" trace is added by default so every mesh
    node is visible (the element traces alone only show cells, not the
    node cloud). Pass ``show_nodes=False`` to suppress it.

    The plotly legend doubles as an interactive filter — single-click
    a legend entry to hide a trace, double-click to isolate it.
    """
    _require_plotly()
    traces = _build_mesh_traces(dims or [1, 2, 3], show_nodes=show_nodes)
    fig = _make_figure(traces, title="apeGmsh — Mesh preview")
    if return_fig:
        return fig
    if browser:
        _open_in_browser(fig, "apeGmsh — Mesh preview")
        return None
    _display_figure(fig)
    return None


def preview(
    session: Any = None,
    *,
    mode: str = "mesh",
    dims: list[int] | None = None,
    show_nodes: bool = True,
    browser: bool = False,
    return_fig: bool = False,
) -> Any:
    """Unified entry point — routes to ``preview_model`` or ``preview_mesh``.

    Parameters
    ----------
    mode : {"model", "mesh"}
        Which scene to render. Default ``"mesh"``.
    show_nodes : bool
        Mesh mode only — render the full mesh-node cloud as a
        separate trace. Ignored in model mode.
    browser : bool
        Open in a new browser tab instead of rendering inline.
    return_fig : bool
        Skip display and return the raw plotly ``Figure``.
    """
    if mode == "model":
        return preview_model(
            session, dims=dims, browser=browser, return_fig=return_fig,
        )
    if mode == "mesh":
        return preview_mesh(
            session, dims=dims, show_nodes=show_nodes,
            browser=browser, return_fig=return_fig,
        )
    raise ValueError(f"Unknown preview mode: {mode!r} (expected 'model' or 'mesh')")
