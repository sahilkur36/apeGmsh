"""
sections — Fiber section generators for OpenSeesPy.
====================================================

Standalone package that uses Gmsh internally to mesh cross-sections
into fibers and injects them directly into an active OpenSeesPy model.

Usage::

    from sections import RectangularColumnSection

    sec = RectangularColumnSection(
        b=400, h=600, cover=40,
        top_bars=(3, 25), bot_bars=(3, 25),
        fc=30, fy=420,
    )
    sec.build(sec_tag=1)          # creates materials + fiber section in OpenSeesPy
    sec.plot()                    # visualise the discretisation
    fibers = sec.get_fibers()     # raw fiber data if needed
"""

from sections._rectangular import RectangularColumnSection

__all__ = [
    "RectangularColumnSection",
]
