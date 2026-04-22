"""Unit tests for g.opensees.elements.vecxz helper."""
import math

import pytest

from apeGmsh.solvers._opensees_elements import _Elements


vecxz = _Elements.vecxz


def _close(a, b, tol=1e-12):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class TestVecxzDefault:
    def test_horizontal_beam_default_local_z_is_plus_Z(self):
        assert _close(vecxz(axis=(1, 0, 0)), (0.0, 0.0, 1.0))

    def test_normalizes_non_unit_axis(self):
        # axis magnitude should not affect the result
        assert _close(vecxz(axis=(5, 0, 0)), (0.0, 0.0, 1.0))

    def test_normalizes_non_unit_local_z(self):
        assert _close(vecxz(axis=(1, 0, 0), local_z=(0, 0, 7)),
                      (0.0, 0.0, 1.0))

    def test_custom_local_z_direction(self):
        # Beam along +X, section oriented so local-z = +Y
        assert _close(vecxz(axis=(1, 0, 0), local_z=(0, 1, 0)),
                      (0.0, 1.0, 0.0))


class TestVecxzRoll:
    def test_roll_90_around_X_rotates_plus_Z_to_minus_Y(self):
        # Right-hand rule about +X: +Z → -Y at 90°
        assert _close(vecxz(axis=(1, 0, 0), roll_deg=90),
                      (0.0, -1.0, 0.0), tol=1e-9)

    def test_roll_180_flips_sign(self):
        assert _close(vecxz(axis=(1, 0, 0), roll_deg=180),
                      (0.0, 0.0, -1.0), tol=1e-9)

    def test_roll_360_is_identity(self):
        assert _close(vecxz(axis=(1, 0, 0), roll_deg=360),
                      (0.0, 0.0, 1.0), tol=1e-9)

    def test_negative_roll_opposite_direction(self):
        # About +X, -90° takes +Z to +Y
        assert _close(vecxz(axis=(1, 0, 0), roll_deg=-90),
                      (0.0, 1.0, 0.0), tol=1e-9)

    def test_roll_preserves_unit_length(self):
        for angle in (13.7, 45.0, 217.3):
            v = vecxz(axis=(1, 2, 3), roll_deg=angle)
            mag = math.sqrt(sum(x * x for x in v))
            assert abs(mag - 1.0) < 1e-12

    def test_roll_vector_stays_perpendicular_to_axis(self):
        # local_z was perpendicular to axis; rotating about axis keeps
        # it perpendicular.
        axis = (1, 0, 0)
        for angle in (0, 30, 45, 90, 123):
            v = vecxz(axis=axis, roll_deg=angle)
            dot = sum(a * b for a, b in zip(axis, v))
            assert abs(dot) < 1e-12


class TestVecxzGuards:
    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            vecxz(axis=(0, 0, 0))

    def test_zero_local_z_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            vecxz(axis=(1, 0, 0), local_z=(0, 0, 0))

    def test_collinear_local_z_and_axis_raises(self):
        with pytest.raises(ValueError, match="collinear"):
            vecxz(axis=(1, 0, 0), local_z=(1, 0, 0))

    def test_anti_parallel_local_z_and_axis_raises(self):
        with pytest.raises(ValueError, match="collinear"):
            vecxz(axis=(1, 0, 0), local_z=(-2, 0, 0))


class TestVecxzIntegrationWithAddGeomTransf:
    """vecxz output should plug directly into add_geom_transf."""

    def test_plugs_into_add_geom_transf(self, g):
        vxz = g.opensees.elements.vecxz(axis=(1, 0, 0))
        g.opensees.set_model(ndm=3, ndf=6)
        g.opensees.elements.add_geom_transf(
            "Beams", "Linear", vecxz=list(vxz),
        )
        stored = g.opensees._geom_transfs["Beams"]
        assert stored["transf_type"] == "Linear"
        assert _close(stored["vecxz"], (0.0, 0.0, 1.0))
