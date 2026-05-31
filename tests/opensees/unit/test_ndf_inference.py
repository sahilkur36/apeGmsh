"""ADR 0048 P2 — the per-node ndf inference engine (pure core + ndm guard).

Tests the rule algebra validated by the P0 spike, now as real code raising
``BridgeError``:

    candidate = max(required_floor) over incident elements
    VALID iff candidate in ndf_ok(elem) for EVERY incident element
    empty/incompatible -> BridgeError -> separate node + equalDOF

The fem-walking adapter ``infer_node_ndf`` is exercised in the P2b wiring
suite against a real broker; here we lock the pure core + the ndm guard.
"""
import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    _infer_ndf_from_incidence,
    assert_ndm_compatible,
)


@pytest.mark.parametrize(
    "classes, ndm, expected",
    [
        (["stdBrick", "stdBrick"], 3, 3),            # homogeneous 3D solid
        (["quad", "quad"], 2, 2),                    # homogeneous 2D plane
        (["elasticBeamColumn"], 3, 6),               # 3D frame
        (["elasticBeamColumn"], 2, 3),               # 2D frame
        (["truss"], 3, 3),                           # 3D truss alone
        (["truss", "elasticBeamColumn"], 3, 6),      # truss adapts to the beam's 6
        (["ShellMITC4", "ShellMITC4"], 3, 6),        # homogeneous shell
    ],
)
def test_infer_node_ndf_ok_cases(classes, ndm, expected):
    out = _infer_ndf_from_incidence({7: classes}, ndm)
    assert out == {7: expected}


def test_quad_plus_beam_2d_must_not_share():
    """Frame-on-plane-strain: quad{2} cannot host the beam's ndf=3."""
    with pytest.raises(BridgeError, match="equal_dof|equalDOF"):
        _infer_ndf_from_incidence({1: ["quad", "elasticBeamColumn"]}, 2)


def test_shell_plus_solid_3d_must_not_share():
    """ADR 0046: stdBrick{3} cannot host the shell's ndf=6."""
    with pytest.raises(BridgeError, match="equal_dof|equalDOF"):
        _infer_ndf_from_incidence({1: ["ShellMITC4", "stdBrick"]}, 3)


def test_beam_plus_solid_3d_is_caught_stricter_than_disjoint_guard():
    """The ∩ gate is stricter than ndf_ok-disjoint: beam{3,6} and brick{3}
    are NOT disjoint, but the node still needs ndf=6 which the brick rejects.
    validate_node_ndf_element_compat misses this; inference catches it."""
    with pytest.raises(BridgeError, match="require ndf=6"):
        _infer_ndf_from_incidence({1: ["elasticBeamColumn", "stdBrick"]}, 3)


def test_unclassifiable_element_fails_loud():
    with pytest.raises(BridgeError, match="not in the capability registry"):
        _infer_ndf_from_incidence({1: ["NotAnElement"]}, 3)


def test_multiple_nodes_independent():
    out = _infer_ndf_from_incidence(
        {1: ["stdBrick"], 2: ["stdBrick"], 3: ["stdBrick"]}, 3,
    )
    assert out == {1: 3, 2: 3, 3: 3}


# ---- ndm compatibility guard ----------------------------------------------

def test_assert_ndm_compatible_passes_for_consistent_model():
    assert_ndm_compatible(["stdBrick", "elasticBeamColumn", "truss"], 3)
    assert_ndm_compatible(["quad", "elasticBeamColumn"], 2)


def test_assert_ndm_mix_2d_3d_elements_raises():
    with pytest.raises(BridgeError, match="mix 2D and 3D"):
        assert_ndm_compatible(["quad", "stdBrick"], 2)


def test_assert_ndm_value_outside_common_set_raises():
    # all elements support {3} only, but the user asked for ndm=2
    with pytest.raises(BridgeError, match="common ndm"):
        assert_ndm_compatible(["stdBrick", "ShellMITC4"], 2)


def test_assert_ndm_skips_unclassifiable():
    # unknown class is skipped; the brick pins ndm=3 and passes
    assert_ndm_compatible(["stdBrick", "MysteryElement"], 3)
