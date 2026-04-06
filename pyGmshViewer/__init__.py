"""pyGmshViewer — Standalone 3D viewer for pyGmsh meshes and FEM results."""

__version__ = "0.1.0"


def show(*filepaths, blocking=True):
    """Launch the pyGmshViewer from a notebook or script.

    Parameters
    ----------
    *filepaths : str or Path
        One or more VTU/PVD/MSH files to open on launch.
    blocking : bool
        If True (default), runs the Qt event loop and blocks until the
        window is closed — safe for scripts and ``if __name__ == '__main__'``.
        If False, launches in a subprocess so the notebook keeps running.

    Examples
    --------
    From a notebook cell (non-blocking, recommended in Jupyter)::

        from pyGmshViewer import show
        show("results.vtu", blocking=False)

    From a script (blocking)::

        from pyGmshViewer import show
        show("results.vtu")

    Multiple files::

        show("mesh.vtu", "modes.pvd", blocking=False)
    """
    from pathlib import Path

    paths = [str(Path(f).resolve()) for f in filepaths]

    if not blocking:
        # Non-blocking: launch as a subprocess so the notebook stays alive
        import sys, subprocess
        cmd = [sys.executable, "-m", "pyGmshViewer"] + paths
        subprocess.Popen(cmd)
        return

    # Blocking: run the Qt event loop in this process
    import pyvista as pv
    pv.set_plot_theme("dark")
    pv.global_theme.font.color = "white"

    import sys
    from PySide6.QtWidgets import QApplication
    from pyGmshViewer.main_window import MainWindow

    app = QApplication.instance()
    own_app = app is None
    if own_app:
        app = QApplication(sys.argv)

    window = MainWindow()
    for p in paths:
        try:
            window.load_file(p)
        except Exception as e:
            print(f"Warning: could not load '{p}': {e}")
    window.show()

    if own_app:
        app.exec()
    else:
        # Inside an existing event loop (e.g. IPython with %gui qt)
        # Just show the window; the existing loop will handle events
        pass
