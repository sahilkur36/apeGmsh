"""
Build a `.ipynb` from a `.py` source with cell markers.

Cell markers are lines starting with ``# %% [markdown]`` or ``# %%``
(a standard widely supported by VS Code, Spyder, Jupytext, etc.).

Usage::

    python _build_notebook.py <source.py> <target.ipynb>

The target notebook has no cached outputs and no execution counts —
run it in Jupyter to populate those.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build(source_py: Path, target_ipynb: Path) -> None:
    raw = source_py.read_text(encoding="utf-8").splitlines(keepends=True)

    cells: list[dict] = []
    cur_kind = "code"
    cur_lines: list[str] = []

    def flush() -> None:
        if not cur_lines:
            return
        # Strip leading/trailing blank lines
        while cur_lines and cur_lines[0].strip() == "":
            cur_lines.pop(0)
        while cur_lines and cur_lines[-1].strip() == "":
            cur_lines.pop()
        if not cur_lines:
            return
        if cur_kind == "markdown":
            # Strip the leading `# ` from each comment line
            source = [
                (ln[2:] if ln.startswith("# ") else (ln[1:] if ln.startswith("#") else ln))
                for ln in cur_lines
            ]
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cur_lines[:],
            })

    for line in raw:
        stripped = line.strip()
        if stripped.startswith("# %% [markdown]"):
            flush()
            cur_kind = "markdown"
            cur_lines = []
        elif stripped.startswith("# %%"):
            flush()
            cur_kind = "code"
            cur_lines = []
        else:
            cur_lines.append(line)
    flush()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "opensees_venv",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"wrote {sys.argv[2]}")
