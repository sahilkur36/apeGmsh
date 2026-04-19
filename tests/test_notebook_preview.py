"""Tests for apeGmsh.viz.NotebookPreview — Colab-friendly WebGL preview."""
from __future__ import annotations

import pytest

from apeGmsh.viz import NotebookPreview as np_mod


def test_helpful_error_when_plotly_missing(monkeypatch):
    """Without plotly the public preview functions raise a clear ImportError.

    The real project dependency is optional (plotly ships pre-installed on
    Colab; users without it locally need `pip install plotly`). The error
    message must guide them, not blow up with a bare ModuleNotFoundError.
    """
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "plotly.graph_objects" or name.startswith("plotly"):
            raise ImportError("No module named plotly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError) as exc:
        np_mod._require_plotly()
    assert "plotly" in str(exc.value).lower()
    assert "pip install plotly" in str(exc.value)


def test_public_api_is_exposed():
    """The three entry points are exported at the module level."""
    assert callable(np_mod.preview_model)
    assert callable(np_mod.preview_mesh)
    assert callable(np_mod.preview)


def test_top_level_preview_is_reexported():
    """``apeGmsh.preview`` is the canonical convenience handle."""
    import apeGmsh
    assert callable(apeGmsh.preview)
    # Don't compare identity — test ordering in the full suite can
    # reimport NotebookPreview. Compare the qualified origin instead.
    assert apeGmsh.preview.__module__ == "apeGmsh.viz.NotebookPreview"
    assert apeGmsh.preview.__name__ == "preview"


def test_mode_dispatch_rejects_unknown():
    """``preview(mode=...)`` validates the mode argument."""
    # Should fail before hitting the plotly import (fast-fail)
    pytest.importorskip("plotly")
    with pytest.raises(ValueError) as exc:
        np_mod.preview(mode="blueprint")
    assert "model" in str(exc.value)
    assert "mesh" in str(exc.value)


def test_theme_colors_returns_hex_strings():
    """_theme_colors serializes palette tuples to hex for plotly."""
    colors = np_mod._theme_colors()
    for key in ("dim0", "dim1", "dim2", "dim3"):
        assert colors[key].startswith("#"), (
            f"{key} should be a hex string, got {colors[key]!r}"
        )
        assert len(colors[key]) == 7


def test_session_methods_exist():
    """Model.preview and Mesh.preview are reachable via the session tree."""
    # Verify the attribute is present — no runtime invocation needed
    from apeGmsh.core.Model import Model
    from apeGmsh.mesh.Mesh import Mesh
    assert hasattr(Model, "preview")
    assert hasattr(Mesh, "preview")
    assert callable(getattr(Model, "preview"))
    assert callable(getattr(Mesh, "preview"))
