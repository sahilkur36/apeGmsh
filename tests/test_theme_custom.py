"""Tests for custom-theme persistence and built-in protection."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from apeGmsh.viewers.ui import theme as theme_mod


@pytest.fixture(autouse=True)
def _isolate_custom_palettes():
    """Remove any non-builtin entries from PALETTES after each test.

    The theme editor / save machinery mutates the module-level ``PALETTES``
    dict; without this cleanup, test order matters.
    """
    yield
    for name in list(theme_mod.PALETTES.keys()):
        if name not in theme_mod._BUILTIN_THEME_IDS:
            del theme_mod.PALETTES[name]


def test_all_builtin_themes_load():
    expected = {
        "catppuccin_mocha", "catppuccin_latte", "neutral_studio", "paper",
        "solarized_dark", "solarized_light", "nord", "tokyo_night",
        "gruvbox_dark", "high_contrast",
    }
    assert expected.issubset(set(theme_mod.PALETTES.keys()))
    for name in expected:
        assert name in theme_mod._BUILTIN_THEME_IDS


def test_save_custom_theme_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        theme_mod.ThemeManager, "themes_dir",
        classmethod(lambda cls: tmp_path),
    )
    draft = replace(theme_mod.PALETTE_CATPPUCCIN_MOCHA, name="my_custom")
    path = theme_mod.ThemeManager.save_custom_theme(draft)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "my_custom"
    assert theme_mod.PALETTES["my_custom"] is draft


def test_save_custom_theme_refuses_builtin_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        theme_mod.ThemeManager, "themes_dir",
        classmethod(lambda cls: tmp_path),
    )
    draft = replace(theme_mod.PALETTE_CATPPUCCIN_MOCHA, name="paper")
    with pytest.raises(ValueError):
        theme_mod.ThemeManager.save_custom_theme(draft)


def test_delete_custom_theme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        theme_mod.ThemeManager, "themes_dir",
        classmethod(lambda cls: tmp_path),
    )
    draft = replace(theme_mod.PALETTE_NORD, name="my_nord")
    theme_mod.ThemeManager.save_custom_theme(draft)
    assert "my_nord" in theme_mod.PALETTES
    removed = theme_mod.ThemeManager.delete_custom_theme("my_nord")
    assert removed
    assert "my_nord" not in theme_mod.PALETTES
    assert not (tmp_path / "my_nord.json").exists()


def test_delete_builtin_theme_forbidden():
    with pytest.raises(ValueError):
        theme_mod.ThemeManager.delete_custom_theme("paper")


def test_load_custom_themes_skips_builtin_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        theme_mod.ThemeManager, "themes_dir",
        classmethod(lambda cls: tmp_path),
    )
    # Attempt to override a built-in via an on-disk file.
    payload = asdict(replace(theme_mod.PALETTE_CATPPUCCIN_MOCHA))
    payload["text"] = "#ff00ff"  # obviously different
    (tmp_path / "paper.json").write_text(json.dumps(payload))
    theme_mod.ThemeManager._load_custom_themes()
    # Built-in paper should be untouched.
    assert theme_mod.PALETTES["paper"] is theme_mod.PALETTE_PAPER


def test_settings_and_theme_editor_exported():
    import apeGmsh
    assert callable(apeGmsh.settings)
    assert callable(apeGmsh.theme_editor)
    from apeGmsh import viewers
    assert apeGmsh.theme_editor is viewers.theme_editor
