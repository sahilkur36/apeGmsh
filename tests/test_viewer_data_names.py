"""ViewerData surfaces bridge-side name aliases through the h5 read seam.

Names flow viewer-ward only via ``H5Model.names()`` (ADR 0026 — the
viewers' single legal import).  ``ViewerData.name_for`` lets a viewer
label a primitive by its human name instead of the bare tag.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.section.fiber import FiberPoint
from apeGmsh.viewers.data import ViewerData

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _named_model_h5(tmp_path: Path) -> Path:
    fem = build_simple_frame_fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01, name="rebar")
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
        name="col_sec",
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    out = tmp_path / "named.h5"
    ops.h5(str(out))
    return out


def test_h5model_names_reads_sidecar(tmp_path: Path) -> None:
    p = _named_model_h5(tmp_path)
    with h5_reader.open(str(p)) as m:
        lut = {n: (k, t) for n, k, t in m.names()}
    assert lut["rebar"][0] == "uniaxialMaterial"
    assert lut["col_sec"][0] == "section"


def test_viewerdata_name_for(tmp_path: Path) -> None:
    p = _named_model_h5(tmp_path)
    vd = ViewerData.from_h5(str(p))
    kind, tag = next((k, t) for n, k, t in vd.names if n == "rebar")
    assert kind == "uniaxialMaterial"
    assert vd.name_for(kind, tag) == "rebar"
    assert vd.name_for("uniaxialMaterial", 9999) is None


def test_viewerdata_from_fem_has_no_names() -> None:
    # Live-FEM snapshots carry no bridge names (emit-time concept).
    vd = ViewerData.from_fem(build_simple_frame_fem())
    assert vd.names == ()
    assert vd.name_for("uniaxialMaterial", 1) is None
