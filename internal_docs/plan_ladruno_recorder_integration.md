# Plan — Ladruno recorder integration (`ops.recorder.Ladruno` + `Results.from_ladruno` + energy)

**Status:** proposed (2026-05-31) · **Owner:** nmora · **Scope:** apeGmsh-side
consumption of the OpenSees *Ladruno fork*'s canonical recorder.

This is the apeGmsh half of items **#1 (Ladruno recorder)** and **#2 (energy
balance)** from the fork's apeGmsh-facing contract
(`nmorabowen/OpenSees@ladruno:Ladruno_implementation/ladruno_apegmsh_contract.md`).
The fork-side work is shipped/verified; this plan is what apeGmsh must build to
*emit* the recorder and *read* `.ladruno` back.

> **Why this first.** The Ladruno recorder is the fork's *canonical* recorder. It
> (a) subsumes the energy-balance recorder — energy lands inside `.ladruno` via
> `-G energy <regions>`, and (b) **solves beam orientation** by writing real
> `MODEL/LOCAL_AXES` quaternions, retiring the `.mpco` "no beam vecxz" workaround
> (apegmsh-helper §7.2) for wired classes. Highest leverage of the five fork
> features.

---

## Canonicity invariant (the whole point of the recorder)

**`.ladruno` is canonical — READ-DIRECT, NEVER TRANSCODE.** The recorder exists to
be the single, self-sufficient source of truth (schema Principle 0: "there is no
privileged native path; this *is* the native path"). The apeGmsh reader therefore
**must not** transcode `.ladruno` into any derived/cached representation on disk
(no `NativeWriter`, no sidecar `.h5`, no "parse once, cache native" step). It reads
HDF5 groups → the in-memory `ResultsReader` protocol **lazily and on demand**,
exactly as `_native.py`/`_mpco.py` do today (their caches are in-memory
memoization, *not* disk transcode — verified). The `.ladruno` bytes stay
authoritative.

The only permitted "transforms" are **read-time interpretation at the API
boundary**, which do not re-encode the file:
1. **Neutral-vocabulary aliasing** — surfacing a file component as `axial_force` /
   `shear_y` so plotting is solver-agnostic (a label alias).
2. **Geometry reconstruction** via the file's **own** `BASIS` descriptor
   (`x(ξ)=Σ Rᵢ(ξ)·Xᵢ`) — the file dictates the map; the reader imposes nothing.

If a clean read would otherwise need a transform, prefer a **fork-side writer
change** (see "Fork-side asks") over an apeGmsh-side transcode.

## Governing constraints (non-negotiable)

1. **Fork is opt-in; vanilla never breaks.** apeGmsh must keep running on stock
   `openseespy`. The Ladruno recorder is unavailable there — gate **at the point
   of use**, never force the fork, never fail at import.
   - *Emit:* `ops.recorder.Ladruno(...)` produces deck text on **any** build —
     emission is just a `recorder ladruno ...` line. The fork requirement bites
     only at `ops.run()` (clear "requires the Ladruno fork build" error).
   - *Read:* `Results.from_ladruno(...)` needs only `h5py` — **no fork at read
     time**. It keys on `INFO/GENERATOR="Ladruno"`.
2. **Class tags read live, never hardcoded.** Fork tags moved to a private
   **≥33000 band**; the old sub-300 values (272/273/61/63/26) are dead. Read from
   the fork `classTags.h` / `LEDGER_implementations.md`. *(For the reader this
   barely matters — see "self-describing" below — but it matters for the response
   catalog if/when we register fork elements.)*
3. **Schema is actively evolving — pin to `FORMAT_VERSION`.** `ladruno_schema_v1`
   is `status: draft`; the live writer is already **ahead** of it (chunked
   time-series, `LOCAL_AXES` wired for `ElasticBeam3d` only, energy `ON_DOMAIN`
   landed). **Verify against a real fixture from the current fork build, not the
   schema doc.** Reader accepts a two-`FORMAT_VERSION` window (mirror ADR 0023).

---

## What's different from MPCO (read this before mirroring blindly)

The reader is **not** a line-for-line clone of `_mpco.py`. Two divergences:

- **Self-describing geometry.** Every `MODEL/ELEMENTS/<classTag>-<ClassName>[g]`
  group carries a `BASIS` descriptor (`TOPOLOGY/FAMILY/ORDER/PARAM_DOMAIN/
  RATIONAL/NUM_CTRL/NUM_GP`) + `QUADRATURE/{GP_PARAM,GP_WEIGHT}`. The reader
  reconstructs `x(ξ)=Σ Rᵢ(ξ)·Xᵢ` from the file — **no per-class shape-function
  table**. apeGmsh implements `B(ξ; FAMILY, ORDER)` **once per family**
  (line/quad/tri/tet/hex × lagrange/serendipity/bernstein), not once per element
  class. This is *less* than MPCO needs, and it's the seam the bezier elements
  (#3/#4) also ride.
- **Component names from the file.** v1 replaces MPCO's flattened `META`
  `;`-string with a structured column map (`COLUMN_MAP`/`COMP_NAMES`
  authoritative). The reader reads component names from the `.ladruno`, so the
  `_response_catalog` is needed only for *neutral-vocabulary* mapping
  (`axial_force`, `shear_y/z`, …), not for raw column decode.

Everything else mirrors MPCO: `MODEL_STAGE[<stamp>]` staging, partition
discovery, the `ResultsReader` protocol, the `LineStationSlab` shape.

---

## On-disk `.ladruno` layout (the reader contract)

Source: `ladruno_schema_v1.md` + the handoff deltas. Apply the deltas — the doc
lags the writer.

```
/INFO
  GENERATOR="Ladruno"  FORMAT_VERSION=1  SOLVER_NAME  SOLVER_VERSION[3]  SPATIAL_DIM
  PARTITIONED  PARTITION_ID  NUM_PARTITIONS              ← partition manifest
/MODEL_STAGE[<stamp>]            attrs STEP,TIME,KIND("transient"|"static"|"eigen")
  /MODEL
    /NODES        { ID[nN], COORDINATES[nN×ndim] }       (control points for HO elems)
    /ELEMENTS/<classTag>-<ClassName>[g]
        CONNECTIVITY[nE×(1+NUM_CTRL)]  +  BASIS attrs  +  QUADRATURE/{GP_PARAM,GP_WEIGHT}
        CTRL_WEIGHT[nE×NUM_CTRL]       (rational only)
    /LOCAL_AXES/<classTag>-<ClassName>[g]/{ FRAME[nE×4 quaternion], ID[nE] }   ← the unlock
    /SECTION_ASSIGNMENTS/SECTION_<tag>[<Class>]/{ ASSIGNMENT, FIBER_DATA, … }
    /SETS/SET_<regionTag>/{ NODES, ELEMENTS }            ← self-contained regions
  /RESULTS
    /ON_NODES/<RESULT>                                   chunked DATA[T×nIds×nComp]+STEP[T]+TIME[T]
    /ON_ELEMENTS/<result>                                ← incl. optional per-step LOCAL_AXES
    /ON_DOMAIN/energyBalance                             ← #2 (KE/IE/DW/ULW/RES/ERR)
    /ON_REGIONS/…                                        ← per-region energy
```

**Time-series is chunked** `[T×nIds×nComp]` with `STEP[T]`/`TIME[T]` axes
(handoff D3), replacing per-step `DATA/STEP_<k>`. Reader must handle **chunked
and legacy**. Partitions: `<stem>.part-<N>.ladruno`, 0-indexed, contiguous, glob
regex `^(?P<stem>.+?)\.part-(?P<idx>\d+)\.ladruno$` (error on a gap).

### Verified layout — live build `605affeb`, FORMAT_VERSION 1 (deltas vs the doc)

Captured by dumping real fixtures (`tests/fixtures/ladruno/*.ladruno`,
regenerable via `_generate_fixtures.py`). **These supersede the schema doc where
they differ:**

- **`INFO`** carries `GENERATOR="Ladruno"`, `FORMAT_VERSION=1`, `SOLVER_NAME`,
  `SOLVER_VERSION[3]`, `SPATIAL_DIM`, `PARTITIONED`/`PARTITION_ID`/`NUM_PARTITIONS`
  — **plus undocumented `STORED_PRECISION="f64"`** and an empty `INFO/PROVENANCE/`
  subgroup. (Reader: key on GENERATOR+FORMAT_VERSION; tolerate extra INFO attrs.)
- **`MODEL_STAGE[1]`** uses an **integer index** (`[1]`, `[2]`, …), *not* a
  timestamp. attrs `KIND` (`static`/`transient`/`eigen`), `STEP`, `TIME`.
- **`MODEL/ELEMENTS/<classTag>-<ClassName>[<colon-suffix>]`** — e.g.
  `12-Truss[1:0]`. **Legacy `GEOMETRY`/`INTEGRATION_RULE` attrs ARE written
  alongside** the new self-describing `FAMILY/TOPOLOGY/ORDER/PARAM_DOMAIN/RATIONAL/
  NUM_CTRL/NUM_GP/NDIR` + `QUADRATURE/{GP_PARAM,GP_WEIGHT}` + `GLOBAL_GP_COORDS`
  (belt-and-suspenders — reader can prefer BASIS+GP_PARAM and ignore the legacy pair).
- **`RESULTS/ON_NODES/<RESULT>`** uses an UPPERCASE key (`DISPLACEMENT`) with attrs
  `COMPONENTS="Ux,Uy"`, `DIMENSION`, `DISPLAY_NAME`, `TYPE`/`DATA_TYPE`; datasets
  `ID[n,1]`, `DATA[T,n,nComp]`, `TIME[T]`, `STEP[T]` (chunked confirmed).
- **`RESULTS/ON_ELEMENTS/<token>/<classTag>-<ClassName>[..]/`** carries the
  **structured `COLUMN_MAP`** (`COMP_NAMES="N"` + `LEVELS/GAUSS_ID/SECTION_TAG/
  FIBER_ID/NUM_COMP/MULTIPLICITY` datasets). **Component names come from the file
  here** — this is the seam that sidesteps the Tri6 GP-order trap.
- **L3 — `MODEL/LOCAL_AXES/<classTag>-<ClassName>/{ID[nE], FRAME[nE,4]}`** (e.g.
  `5-ElasticBeam3d`, **no `[g]` suffix here** — tolerate with/without). `FRAME` =
  unit quaternion. ElasticBeam3d wired; other beams pending (fork-side ask #1).
- **L4 — `-G energy <regionTag>`** accepted → both `RESULTS/ON_DOMAIN/energyBalance`
  (whole model, always) **and** `RESULTS/ON_REGIONS/energyBalance` (per region),
  `COMPONENTS="KE,IE,DW,ULW,RES,ERR"`, `DATA[T,nIds,6]`, `DIMENSION="F*L"`.
- **L1 emit grammar confirmed** against the real recorder: `-N`/`-E`/`-T dt $v`/
  `-T nsteps $n` all accepted (matches MPCO, as assumed).

**Quaternion convention quirk (carry forward):** the frozen `quatFromMat` stores
the **transpose** convention → reconstructed axes are the **rows** of the matrix.
The fork checker documents this; the apeGmsh reader must match it (and a fixture
must lock it).

---

## Seam map (apeGmsh files to touch)

Grounded against current `src/`. Anchors from the seam exploration.

| Seam | File(s) | Mirror of | Change |
|---|---|---|---|
| Emitter primitive | `src/apeGmsh/opensees/recorder.py` (`class MPCO`, ~339–717) | `MPCO` dataclass + `_emit` | New frozen `class Ladruno(Recorder)`; `_emit` → `emitter.recorder("ladruno", …)` |
| Emitter factory | `src/apeGmsh/opensees/_internal/ns/recorder.py` (`_RecorderNS.MPCO`, ~210–246) | `_RecorderNS.MPCO` | New `_RecorderNS.Ladruno(...)` → `_register` |
| Reader entry | `src/apeGmsh/results/Results.py` (`from_mpco`, ~367–455) | `from_mpco` | New `from_ladruno(path, *, fem=None, model_h5=None)`; required `model_h5=` |
| Partition discovery | `src/apeGmsh/results/readers/_mpco_multi.py` (~57–102) | `discover_partition_files` + regex | New `.ladruno` discovery (or parametrize the existing one) |
| Reader core | new `src/apeGmsh/results/readers/_ladruno.py` | `_mpco.py` `MPCOReader` | `LadrunoReader(ResultsReader)`: INFO identity, stages, NODES, self-describing ELEMENTS, chunked ON_NODES/ON_ELEMENTS |
| Family basis lib | new `src/apeGmsh/results/readers/_ladruno_basis.py` | (none — new) | `B(ξ; FAMILY, ORDER)` per family: line2 / quad4 / quad9 / tri3 / tri6-bernstein / tet4 / tet10 / hex8 / hex20. **Shared with bezier #3/#4.** |
| Schema window | `src/apeGmsh/results/schema/_versions.py` (+ a `_native.py`-style validator) | ADR 0023 window | `LADRUNO_FORMAT_VERSION` + two-version acceptance |
| Orientation | `src/apeGmsh/results/readers/_mpco_line_io.py` + `_slabs.py` `LineStationSlab.local_axes_quaternion` | native-reader quaternion path (`_native.py` ~494–694) | Populate `local_axes_quaternion` from `MODEL/LOCAL_AXES`; fall back to native `vecxz` when absent |
| Vocabulary map | `src/apeGmsh/_vocabulary.py` (`LINE_DIAGRAMS`) + `_response_catalog.py` | existing neutral names | Map Ladruno component names → `axial_force/shear_y/z/torsion/bending_moment_y/z` |
| Energy accessor | `src/apeGmsh/results/Results.py` + `_composites.py` | (none — new) | `r.energy(region=None)` → DataFrame `KE/IE/DW/ULW/RES/ERR` from `ON_DOMAIN`/`ON_REGIONS` |
| Energy text reader | new tiny `src/apeGmsh/results/readers/_energy_text.py` | (none) | standalone `EnergyBalance` sidecar → DataFrame (routed by a `text` discriminator, **not** HDF5) |
| Docs/skill | `.claude/skills/apegmsh-helper/references/ladruno.md`, `references/results.md` | — | Flip §7.2 note once orientation lands; document `from_ladruno` |

---

## Testing strategy (no fork at test time)

The blocker is that apeGmsh runs py3.11 + stock `openseespy`; the fork build is
py3.12. **Resolution: commit small `.ladruno` fixtures**, generated once from the
fork build (or the fork's `make_synthetic.py`), into `tests/fixtures/ladruno/`.
apeGmsh reader tests read fixtures — zero fork dependency at test time. This is
the same shape as the existing `.mpco` fixtures.

Minimum fixture set:
- `truss_*.ladruno` — nodal + element value channels (parity vs a sibling `.mpco`
  → the fork already proves 1e-12; we assert the apeGmsh read matches).
- `beam3d_*.ladruno` — `MODEL/LOCAL_AXES` non-identity frame (locks orientation +
  the transpose-convention quirk).
- `energy_*.ladruno` — `ON_DOMAIN/energyBalance` + one region.
- `*.part-0/1.ladruno` — partition merge.
- A higher-order element fixture (tri6 or quad9) — exercises the self-describing
  basis path (and de-risks bezier #3/#4).

Emitter tests need **no** fixture or fork — assert the emitted tcl/py deck
contains the exact `recorder ladruno …` line.

---

## Phased delivery

Each phase is independently shippable (its own PR), verifiable, and ordered so
value lands early.

### L1 — Emitter `ops.recorder.Ladruno(...)`  *(no fork, no fixture)* — ✅ DONE
- `class Ladruno(Recorder)` (`recorder.py`) + `_RecorderNS.Ladruno(...)`
  (`_internal/ns/recorder.py`). Whole-model **value channels**:
  `recorder ladruno <file> [-N <nodal…>] [-E <elem…>] [-T dt $dt | -T nsteps $n]`,
  mirroring `MPCO`'s `_emit` with kind token `"ladruno"`. The emitters need **no**
  change — `recorder(kind, *args)` is generic.
- **Shipped:** 16 unit tests (`tests/opensees/unit/primitives/test_ladruno_recorder.py`)
  green — construction, validation (≥1 response; dT⊻nsteps), `_emit` via
  `RecordingEmitter`, literal tcl + py deck lines, `dependencies()==()`, namespace
  registration. 65 existing recorder tests + mypy on both files clean.
- **Refined scope (deferred *out* of L1, decided during impl):**
  - **`-R` region filter** (the `nodes=`/`elements=` → region fan-out) — deferred;
    `.ladruno` is self-sufficient and the common case is whole-model. Mirror MPCO's
    filter machinery in a later slice only if a user needs sub-model recording.
  - **`-G energy` channel** — moved to **L4** (with the reader), because the exact
    fork grammar + region-tag handling (vs the no-raw-tags rule) needs verification
    against a real `.ladruno` fixture.
  - **Run-gate friendly error** (fork-required at `ops.run()`) — its own small step,
    pending a venv fork build that actually *rejects* an unknown recorder. The
    current venv build (`288f6d0f1`) **predates** the `ladruno` recorder, so the live
    round-trip can't be exercised yet; verify openseespy's unknown-recorder behavior
    on the updated build before wiring the gate (don't ship unverified error-handling).

### L2 — Reader core `Results.from_ladruno(...)`  *(fixtures)*

**L2a — `LadrunoReader` core + nodal reads — ✅ DONE.** Tested *directly*
against committed fork fixtures (`tests/fixtures/ladruno/*.ladruno`), no factory
/ model needed:
- `src/apeGmsh/results/readers/_ladruno.py` `LadrunoReader` — INFO identity +
  `LADRUNO_FORMAT_VERSION` window (`schema/_versions.py`), integer-indexed
  `MODEL_STAGE[k]` enumeration, `KIND`→`mode/static/transient` map, `TIME`-dataset
  time vectors, `partitions()`, **self-describing `fem()`**, `opensees_model()→None`,
  `available_components(NODES)` + `read_nodes()` (chunked `DATA[T,n,nComp]`, reusing
  MPCO's `canonical_node_component` — ladruno shares MPCO result names). Element /
  gauss / line / fiber / layer / spring reads return **empty slabs** (L2b).
- `src/apeGmsh/mesh/_femdata_ladruno_io.py` `read_fem_from_ladruno` +
  `FEMData.from_ladruno_model` — reads nested `<grp>/CONNECTIVITY` + BASIS
  `TOPOLOGY`/`ORDER` (mirrors `from_mpco_model`).
- **Shipped:** 14 reader tests green (identity rejects non-ladruno / wrong-GENERATOR
  / unsupported version; stages/time/fem/nodal read + node-filter + time-slice +
  unknown-component-empty; beam3d fem). New source mypy-clean; no regression (95
  reader+recorder tests).
- **End-to-end round-trip VERIFIED** (worktree on `PYTHONPATH`, fork build
  `605affeb`): two-column frame → `ops.recorder.Ladruno(...)` → live build →
  `analyze` (rc 0) → `LadrunoReader.read_nodes("stage_0","displacement_x")` ==
  live `nodeDisp` to **1e-12**. Closes the L1 emit→run gap *and* proves L2a reads an
  apeGmsh-emitted `.ladruno`. (Codify as a `@pytest.mark.live` test in L2b once the
  `from_ladruno` factory exists, so it asserts on `Results` not the bare reader.)

**L2b-1 — `from_ladruno` factory + self-sufficient model — ✅ DONE.**
- `Results.from_ladruno(path, *, fem=None, model_h5=None)`
  (`results/Results.py`). **Decision REVERSED in favour of the canonical design:**
  `model_h5=` is **OPTIONAL** (not required like `from_mpco`). The self-sufficient
  path turned out *cheaper* than requiring a sibling — `LadrunoReader.opensees_model()`
  builds a **minimal in-memory broker** from the file's `MODEL` group (geometry +
  `ndm` from `INFO/SPATIAL_DIM`, empty bridge zones), honoring schema Principle 0
  ("this *is* the native path, no sibling file"). It's read-time interpretation, not
  a transcode. `model_h5=` still accepted for richer lineage / bridge records (+
  composed-model tag translation). **`ndf=ndm`** in the minimal broker — a `.ladruno`
  does *not* record ndf (DISPLACEMENT is always 3 translations); pass `model_h5=` for
  the exact ndf on rotational models.
- **Shipped:** 5 factory tests green — self-sufficient (no `model_h5`),
  `r.nodes.get(component="displacement_x")` public-API read, minimal-broker ndm/ndf,
  rejects non-ladruno, time-slice. Reader + Results mypy-clean (zero new errors);
  100-test results-suite regression green.

**L2b-2 — element value channels — ✅ DONE.**
- **Element IO engine** `src/apeGmsh/results/readers/_ladruno_element_io.py` — parses
  each `ON_ELEMENTS/<token>/<class>[…]` bucket's structured `COLUMN_MAP` (one row per
  output block; `LEVELS` 0=Element/1=GP/2=Section/4=NdMaterial, `GAUSS_ID`) +
  `COMP_NAMES` (newline-separated, one CSV per block) into blocks, slices the block-major
  `DATA`, and maps tokens to neutral vocabulary **from the file** (no per-class catalog):
  `sigmaIJ`/`etaIJ`→`stress_ij`/`strain_ij`, `N`/`Vy`/…(`_<station>`)→`axial_force`/…
  Continuum tokens
  handle **both** vocabularies in the wild: digit form `sigma11`/`eta11` (stock
  FourNodeQuad) **and** axis form `sigma_xx`/`eps_xx`/`gamma_xy` (the real fork
  BezierTri6 + the contract's recommended naming) — `gamma_xy` (engineering shear,
  same as digit `eta12`) → `strain_xy`.
- **Reads** wired in `_ladruno.py`: `read_gauss` (continuum stress/strain, neutral-canonical
  `stress_ij`/`strain_ij`, natural_coords from `GP_PARAM[gauss_id]`), `read_line_stations`
  (neutral-canonical `axial_force`/… — `localForce` 2-station with the MPCO-parity end-force
  sign flip + `basicForce` 1-station at ξ=0), and **token-driven** `read_elements`:
  `component` IS the file's `ON_ELEMENTS/<token>` key (`basicForce`/`localForce`/`force`/
  `globalForce`) and the `ElementSlab` carries the raw `(T, E, NUM_COLUMNS)` block in
  `COMP_NAMES` column order (no neutral remap at the element level — the neutral views live on
  the line/gauss levels). `available_components(ELEMENTS)` lists the element-level (`LEVELS==0`)
  token keys; `_ids_to_ops`/`_index_to_fem` tag translation mirrored from MPCO. **API note:**
  this makes `results.elements.get(component=…)` take a file token for Ladruno vs a neutral name
  for MPCO — a deliberate cross-backend asymmetry (Ladruno is file-driven; the neutral beam view
  is `line_stations`). Aligned with the parallel effort's token-driven stance; the three pinned
  micro-details (npe=raw NUM_COLUMNS; ELEMENTS lists LEVELS==0 tokens; `ElementSlab` unextended)
  are flagged for reconciliation.
- **Partition merge** `_ladruno_multi.py` `LadrunoMultiPartitionReader` +
  `discover_partition_files` (`.part-<N>.ladruno`); reuses the solver-neutral stitch
  helpers from `_mpco_multi`. Wired into `Results.from_ladruno` (auto-discovery +
  `merge_partitions=` + list form), mirroring `from_mpco`.
- **Basis lib** `src/apeGmsh/_basis.py` — the decided neutral `B(ξ; FAMILY, ORDER, TOPOLOGY)`
  front-door. **Non-duplicative:** delegates *lagrange* to the existing Gmsh-keyed
  `fem/_shape_functions.py`, and **adds** the genuinely-missing *bernstein* (Bézier)
  evaluators that `plan_bezier_elements_integration.md` B4 imports unchanged. The
  **Basis-function correctness** is proven by the **formula cross-check** against the
  reference elements (`C:\Users\nmora\Github\bezierFEM`, Kadapa 2018) at random *interior*
  points (2.2e-16) — non-degenerate, i.e. it genuinely distinguishes Bernstein from
  Lagrange. The tet10 mid-edge order carries the reference's **Larenas N9↔N10 Gmsh swap** —
  edges `(1-2,2-3,1-3,1-4,3-4,2-4)`, *not* the naive `…1-4,2-4,3-4` (the earlier contract
  text was wrong here; a wrong order silently corrupts `x_global`). Simplex `xi` accepts the
  file's `GP_PARAM` **free**-coord form (2 for tri / 3 for tet; last coord derived) as well
  as full barycentric.
  **Verified against the REAL fork BezierTri6** (tag 33000, fork build `605affeb`): a bezier
  `.ladruno` writes **no `GLOBAL_GP_COORDS`** (unlike Truss/quad), so the basis lib is a
  genuine *runtime* dependency for bezier world-coord reconstruction. The committed
  `bezier_tri6.ladruno` fixture is **straight-sided** (mid-edge nodes at the edge midpoints —
  apeGmsh's Gmsh pipeline does not emit curved/high-order-snapped geometry), so the element
  is **affine**: control points `P = X` and `x = B(ξ)·X` reproduces `eleResponse(e,"gpCoord")`
  to 0.0. **Caveat (degeneracy):** on affine geometry the Bernstein and Lagrange maps
  coincide, so the 0.0 reconstruction proves only the reader's `GP_PARAM`/`CONNECTIVITY`/
  `COORDINATES` *plumbing*, not the basis — basis correctness is the formula cross-check
  above. For a *genuinely curved* Bézier element the maps diverge and `x = B·X` (nodes) ≠
  `x = B·P` (control points); that path is **unreachable** while the pipeline is straight-
  sided. Revisit (persist control-point coords / `GLOBAL_GP_COORDS`, or fail loud on
  `RATIONAL=1`) only if curved high-order meshing (`SetOrder` + `HighOrderOptimize` / curved
  CAD) is ever enabled. The fixture also locks the `sigma_xx`/`eps_xx` Gauss read fork-free.
- **Fixtures** (`_generate_fixtures.py`): `quad2d.ladruno` (FourNodeQuad `stress`/`strain`
  → Gauss + `force` → element) + `truss2d.part-0/1.ladruno` (hand-split from the
  single-partition truss with pure h5py — real `mpiexec` is unavailable in the dev env,
  so the synth faithfully exercises the node-union + element-concat stitch path).
- **Shipped & verified:** 12 basis tests (partition-of-unity all families; lagrange
  Kronecker-delta; **fork-fixture reconstruction `x=B@X`==`GLOBAL_GP_COORDS` to 0.0** for
  line2+quad4) + reader/factory tests (gauss/element/line reads, element filter,
  partition merge) + a `@pytest.mark.live` round-trip (`test_ladruno_live_roundtrip.py`):
  emit→run→`Results.from_ladruno` element reads == live `ops.basicForce`/`ops.eleResponse`
  to 1e-12/1e-10. All green; new source mypy-clean (zero new errors); 97-test
  MPCO-multi/parity/recorder/shape-fn regression green.
- **Deferred to L2b-3 (no fixture / no consumer yet):** `read_fibers`/`read_layers`/
  `read_springs` (still empty slabs — fiber/layer/spring buckets need their own fixtures);
  level-2 `section.force` line stations (force-based beams — needs a DispBeamColumn
  fixture; the engine already parses the `N_i` grammar, only the per-GP station ξ from
  `GP_PARAM` is untested); runtime wiring of `_basis.py` into a self-describing
  `GaussSlab.global_coords` for HO/bezier families — the Gmsh-code-keyed `compute_global_coords`
  covers the linear fixtures but, for a bezier slab, falls through to the centroid+bbox
  approximation (*correct values, approximate world placement*); closing it needs the GaussSlab
  to carry the file's BASIS/connectivity so it can route through `_basis`, an ADR-level seam owned
  by the bezier plan (#3/#4), which imports `_basis.basis_values` (proven 0.0 vs `gpCoord`). No
  sibling-`.mpco` parity fixture was added — the live round-trip against `ops.*` is the stronger
  ground truth.

### L3 — Beam orientation from `MODEL/LOCAL_AXES`  *(the unlock)* — ✅ DONE
- `LocalAxes` result type (`results/_slabs.py`) — per-element scalar-first
  quaternions + `.matrices`/`.x_axis`/`.y_axis`/`.z_axis` (local axes are the **ROWS**
  of each matrix; OpenSees `quatFromMat` transpose). `LadrunoReader.read_local_axes`
  flattens `MODEL/LOCAL_AXES/<class>/{ID,FRAME}` → `{id: quat}` (identity fallback).
  Public **`results.elements.local_axes(...)`** (`TypeError` on non-Ladruno).
  `LocalAxes` exported from `apeGmsh.results`. Shipped on **PR #509**.
- **Verified:** beam3d `.x_axis` == node1→node2 direction to 1e-6 + orthonormal frame;
  element filter; absent-frames empty. mypy-clean; regression green. Skill
  `ladruno.md` updated (canonical + synced mirror) — the §7.2 "no beam vecxz" stale
  note is now superseded for `.ladruno`.
- **Deferred (needs L2b-2 line reads):** auto-populating
  `LineStationSlab.local_axes_quaternion` so `results.plot.line_force(...)` orients
  itself — wire `read_local_axes` into the line-station read when L2b-2 lands. The
  frames are already exposed via the accessor; this is the plumbing into diagrams.

### L4 — Energy accessor `r.energy(region=...)`  *(#2)* — ✅ DONE (HDF5 path)
- `LadrunoReader.read_energy(stage_id, *, region=)` reads
  `ON_DOMAIN/energyBalance` (region=None) / `ON_REGIONS/energyBalance` (region=tag);
  `Results.energy(*, region=, stage=)` → pandas DataFrame `KE/IE/DW/ULW/RES/ERR`
  indexed by time. `TypeError` on non-Ladruno results; `ValueError` on
  absent-energy / unknown-region. Shipped on **PR #509**.
- **Shipped:** 4 tests (whole-domain, per-region, unknown-region, energy-absent).
  mypy-clean; results regression green.
- **Deferred:** the standalone `EnergyBalance` **text-sidecar** → DataFrame reader
  (a `text` discriminator, separate from the HDF5 path) — only needed when a user
  runs the standalone `recorder EnergyBalance` instead of the `.ladruno` `-G energy`
  channel. Surfacing `ERR%` as a viewer/history-plot badge is also deferred (UI).

### L5 — Polish / docs / parity gate
- Pin the two-version window; document `from_ladruno` in `references/results.md`
  + the contract row; CHANGELOG. Confirm the family basis lib is the one the
  bezier work will import (avoid a second copy).

---

## Open questions

> [!decided] **Family basis library lives NEUTRAL at `src/apeGmsh/_basis.py`** —
> *not* reader-local. **Settled up-front** (before L2 ships), because there are
> **two** independent consumers: the `.ladruno` reader *and* the bezier read path
> (`plan_bezier_elements_integration.md` B4 needs `B(ξ; bernstein, 2)` for
> `tri6-bernstein` + `tet10`). Resolving it lazily would let L2 ship reader-local and
> force bezier B4 into a cross-package "promote to neutral" move under time pressure.
> So L2 puts `B(ξ; FAMILY, ORDER)` in `src/apeGmsh/_basis.py` and `_ladruno*.py`
> imports it; the bezier plan imports the **same** module. One copy, no duplication.

> [!question] **`ON_DOMAIN` energy presence** — `ladruno_schema_v1` §7.3 marks it
> *DEFERRED*, but the handoff says energy landed via `-G energy`. **Verify against
> a current-fork fixture before scoping L4's reader paths.** If the live layout
> differs from `ON_DOMAIN/energyBalance` + `ON_REGIONS`, L4 adjusts.

> [!question] **Per-step `LOCAL_AXES`** (co-rotational / Belytschko beams,
> `RESULTS/ON_ELEMENTS/LOCAL_AXES`) — out of scope for L3 (static frame only).
> Revisit when a co-rotational element ships fork-side.

> [!question] **`fem=` vs `model_h5=`** — `.ladruno` is *self-sufficient*
> (geometry, regions, local axes all in-file), unlike `.mpco` which leans on the
> model. Does `from_ladruno` still *require* `model_h5=` for the lineage chain, or
> can it run model-less off the file alone? Default to mirroring `from_mpco`
> (require `model_h5=`) for lineage consistency; revisit if a model-less path is
> wanted.

---

## Fork-side asks (request from the Ladruno team, not work around)

The recorder team is reachable — when a clean apeGmsh read wants a writer change,
prefer asking over an apeGmsh-side workaround (keeps `.ladruno` canonical). Open
candidates, in dependency order:

1. **Remaining-beam `localAxes`** (gates L3 fully). Only `ElasticBeam3d` is wired
   today; `ElasticBeam2d` / `DispBeamColumn2d/3d` / `ForceBeamColumn2d/3d` need the
   same `"localAxes"` response so `MODEL/LOCAL_AXES` covers all wired beams (2D
   fills z=(0,0,1)). The fork handoff already lists this as next work.
2. **Authoritative `COLUMN_MAP`/`COMP_NAMES` on every `ON_ELEMENTS` result** —
   confirm the structured column map is emitted universally, so the reader needs
   **zero** per-class component decode. (If any result still ships a flattened
   string, ask to structure it.)
3. **Energy layout confirmation** — confirm `ON_DOMAIN/energyBalance` +
   `ON_REGIONS` shape on a current-fork fixture (schema doc marks deferred;
   handoff says landed). Adjust L4 to whatever the writer actually emits.
4. **Fixture generation** — a stable `make_synthetic.py` / sample-export recipe we
   can re-run when the schema bumps, so apeGmsh fixtures track the writer.

## Out of scope (this plan)

- Bezier elements #3/#4 (separate plan — but L2's basis lib de-risks them).
- Explicit integrator / auto-dt surface (separate plan).
- Profiler #5 (separate, optional).
- Tier 2 parallel energy `Allreduce` and Tier 3 envelopes (`ENVELOPES/…`) —
  fork-side still in flight; add reader support when the writer lands.

## References
- Fork contract: `nmorabowen/OpenSees@ladruno:Ladruno_implementation/ladruno_apegmsh_contract.md`
- Schema: `…/ladruno_schema_v1.md` · Handoff: `…/ladruno_handoff.md` · Element grammar: `…/ladruno_element_contract.md`
- apeGmsh skill ref: `.claude/skills/apegmsh-helper/references/ladruno.md`
- Mirror seams: `from_mpco` (`results/Results.py`), `_mpco*.py`, `recorder.py` `MPCO`, ADR 0023 (`results/schema/_versions.py`)
