# First steps with apeGmsh

A conversational walkthrough for someone opening apeGmsh for the first
time. It intentionally moves slowly — the goal is to build a mental
model, not to dump the full API. For the complete reference see the
other guides (`guide_basics.md`, `guide_meshing.md`, …) and the
API pages.

This document is built up lesson-by-lesson from real teaching
sessions. Each lesson is short, answers one question, and leaves you
ready for the next.


## Lesson 1 — What apeGmsh is, and why it exists

The one-sentence version:

> apeGmsh lets you describe a structural model once — geometry,
> loads, constraints, masses — and hand it off to a solver (OpenSees
> today) without writing Gmsh API calls by hand.

Two ideas do all the work.

### 1.1  A session owns one model

You open a session with a `with` block. Inside it, the object `g`
exposes everything — geometry, meshing, loads, the solver bridge. No
global state, no stray `gmsh.initialize()` calls scattered around your
script.

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="hello", verbose=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
```

That's a full meshed cube in three lines.

### 1.2  Two ways to open a session

`apeGmsh` (and `Part`, which we'll meet later) supports two equivalent
lifecycle patterns. Pick whichever matches the situation.

**Style A — context manager.** Recommended for scripts. The session
closes cleanly on exit *and* on exceptions.

```python
with apeGmsh(model_name="hello") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
```

**Style B — explicit `begin()` / `end()`.** Recommended for Jupyter
notebooks, where you want the session to stay alive across cells so
you can inspect `g` between steps.

```python
g = apeGmsh(model_name="hello").begin()

g.model.geometry.add_box(0, 0, 0, 1, 1, 3)
# ... more cells, more operations ...

g.end()
```

Both styles produce the same result. The `with` form is safer because
it closes the Gmsh state even if something raises in the middle; the
explicit form is more ergonomic when you are exploring interactively.

### 1.3  Composites group related actions

`g` is not a flat object with two hundred methods. It has ~16
attributes, each a focused slice of the API:

| Access | Purpose |
|---|---|
| `g.model` | Geometry — points, boxes, booleans, STEP I/O |
| `g.mesh` | Meshing — sizing, generation, structured, queries |
| `g.physical` | Named groups (`"Base"`, `"Body"`) that survive the pipeline |
| `g.constraints`, `g.loads`, `g.masses` | Pre-mesh declarations |
| `g.opensees` | The solver bridge |

And the big composites have **sub-composites**. You always write

```python
g.mesh.generation.generate(dim=3)     # correct
```

not

```python
g.mesh.generate(dim=3)                # will not work
```

This is deliberate: the prefix tells you *what kind* of operation it
is (generation vs. sizing vs. structured vs. queries) and keeps the
namespaces honest.

---

_More lessons will be appended here as the guide grows._
