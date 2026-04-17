# %% [markdown]
# # 20 — Time-History Dynamic Analysis (SDOF Step Input)
#
# **Curriculum slot:** Tier 5, slot 20 (final slot).
# **Prerequisite:** 17 — Modal Analysis.
#
# ## Purpose
#
# Time-history analysis integrates the structural equations of
# motion forward in time:
#
# $$
# M\,\ddot{u} + C\,\dot{u} + K\,u \;=\; F(t).
# $$
#
# Two new OpenSees pieces over the modal slot:
#
# 1. **``ops.analysis("Transient")``** instead of ``"Static"``.
# 2. **A time integrator** — ``Newmark(gamma, beta)`` is the
#    default choice (the average-acceleration variant with
#    ``gamma = 0.5, beta = 0.25`` is unconditionally stable and
#    preserves energy exactly for linear systems).
#
# ## Problem — single-DOF mass-spring with step load
#
# One mass $m$ connected to ground through a linear spring $k$.
# At $t = 0$ a constant force $F_0$ is suddenly applied and held
# forever. For an undamped system the response is the well-known
# half-sine-wave-offset:
#
# $$
# u(t) \;=\; \dfrac{F_{0}}{k}\Big(1 - \cos(\omega_{n}\,t)\Big),
# \qquad \omega_{n} = \sqrt{k/m}.
# $$
#
# Three checks:
#
# * **Peak displacement** = $2\,F_{0}/k$ (twice the static value —
#   dynamic magnification factor of 2 at a step input).
# * **Period** $T_{n} = 2\pi/\omega_{n}$ extracted from
#   zero-crossings.
# * **Static final value** — with artificial damping added, the
#   response settles at $F_{0}/k$.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

# Physical parameters (SDOF mass-spring)
m  = 1.0          # mass [kg]
k  = 100.0        # spring stiffness [N/m]
F0 = 5.0          # step-load magnitude [N]

# Derived
omega_n = np.sqrt(k / m)
Tn      = 2 * np.pi / omega_n
u_static = F0 / k
u_peak_expected = 2 * u_static

# Time integration window: watch a few periods
t_end = 3 * Tn
dt    = Tn / 200                  # 200 points per period

print(f"omega_n       : {omega_n:.4f} rad/s")
print(f"Tn            : {Tn:.4f} s")
print(f"u_static (F/k): {u_static:.6e} m")
print(f"expected peak : {u_peak_expected:.6e} m")


# %% [markdown]
# ## 2. OpenSees build — a literal spring + mass
#
# No mesh needed — this is an abstract SDOF. Two nodes:
# one grounded, one carrying the mass. A ``zeroLength`` element
# with ``uniaxialMaterial Elastic(K=k)`` is the spring.

# %%
ops.wipe()
ops.model("basic", "-ndm", 1, "-ndf", 1)

ops.node(1, 0.0)
ops.node(2, 0.0)        # same coordinate — it's a zeroLength element
ops.fix(1, 1)           # ground

ops.mass(2, m)          # lumped mass on node 2

ops.uniaxialMaterial("Elastic", 1, k)
ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 1)


# %% [markdown]
# ## 3. Step load pattern + undamped transient

# %%
# "Constant" time series = always equals 1.0 -> with the Plain
# pattern below, the applied load is permanently F0.
ops.timeSeries("Constant", 1)
ops.pattern("Plain", 1, 1)
ops.load(2, F0)

ops.system("BandGeneral")
ops.numberer("Plain")
ops.constraints("Plain")
ops.test("NormDispIncr", 1e-12, 20)
ops.algorithm("Linear")
ops.integrator("Newmark", 0.5, 0.25)      # average-acceleration (stable, energy-preserving)
ops.analysis("Transient")

n_steps = int(np.ceil(t_end / dt))
t_hist:  list[float] = []
u_hist:  list[float] = []
t = 0.0
for _ in range(n_steps):
    ops.analyze(1, dt)
    t += dt
    t_hist.append(t)
    u_hist.append(ops.nodeDisp(2, 1))

t_arr = np.asarray(t_hist)
u_arr = np.asarray(u_hist)
print(f"integrated {n_steps} steps, t_end = {t_arr[-1]:.4f} s")


# %% [markdown]
# ## 4. Verification — peak + period

# %%
# Peak displacement
u_peak_fem = float(u_arr.max())
err_peak = abs(u_peak_fem - u_peak_expected) / u_peak_expected * 100.0

# Period via zero-crossings of (u - u_static). For a step input,
# u oscillates around u_static with period Tn.
y = u_arr - u_static
zcs_idx = np.where(np.diff(np.sign(y)) != 0)[0]
# Two consecutive same-direction crossings make one full period.
# Simpler: two zero-crossings of opposite sign = half period.
if len(zcs_idx) >= 2:
    first_zc = t_arr[zcs_idx[0]]
    third_zc = t_arr[zcs_idx[2]] if len(zcs_idx) >= 3 else t_arr[zcs_idx[1]] * 2 - first_zc
    Tn_fem = (third_zc - first_zc) if len(zcs_idx) >= 3 else 2 * (t_arr[zcs_idx[1]] - first_zc)
else:
    Tn_fem = float('nan')
err_T = abs(Tn_fem - Tn) / Tn * 100.0 if not np.isnan(Tn_fem) else float('nan')

print("Peak displacement (u_peak = 2 * F0 / k)")
print(f"  Analytical : {u_peak_expected:.6e}  m")
print(f"  FEM        : {u_peak_fem:.6e}  m")
print(f"  Error      : {err_peak:.4f} %")
print()
print("Natural period (T_n = 2 pi / omega_n)")
print(f"  Analytical : {Tn:.6e}  s")
print(f"  FEM (zc)   : {Tn_fem:.6e}  s")
print(f"  Error      : {err_T:.4f} %")


# %% [markdown]
# ## 5. (Optional) plot the response

# %%
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 4))
# plt.plot(t_arr, u_arr, 'b-', lw=1.5, label='FEM')
# # Analytical
# u_analytical = (F0 / k) * (1 - np.cos(omega_n * t_arr))
# plt.plot(t_arr, u_analytical, 'r--', lw=1.5, label='Analytical')
# plt.axhline(u_peak_expected, ls=':', color='k', alpha=0.5, label=f'peak 2F0/k')
# plt.axhline(u_static,        ls=':', color='k', alpha=0.5, label=f'static F0/k')
# plt.xlabel('t [s]'); plt.ylabel('u(t) [m]'); plt.grid(alpha=0.3)
# plt.title('SDOF free-vibration response to a step load')
# plt.legend(); plt.tight_layout(); plt.show()


# %% [markdown]
# ## What this unlocks
#
# * **``ops.analysis("Transient")``** — the dynamic analogue of
#   ``"Static"``. Same assembly of ``system``, ``numberer``,
#   ``constraints``, ``test``, ``algorithm`` — only the
#   integrator and the analysis type change.
# * **Newmark time integrator.** ``Newmark(0.5, 0.25)`` is the
#   average-acceleration method, unconditionally stable for
#   linear systems and energy-preserving. The variant
#   ``Newmark(0.5, 1/6)`` is the linear-acceleration method
#   (second-order accurate, conditionally stable).
# * **``ops.analyze(n, dt)`` in transient mode** takes a time
#   step `dt` as second argument. The ``n`` first argument is the
#   number of time steps to advance.
# * **Pathway to ground motion.** Replace the ``Constant``
#   timeSeries with ``Path`` (time-series from file) or
#   ``GroundMotion`` to apply base excitation — the rest of the
#   integrator and analysis wiring stays identical.
#
# ---
#
# ## End of the curriculum
#
# Slot 20 closes the 20-slot apeGmsh curriculum. From a
# static-elastic plate in slot 01 to a time-integrated dynamic
# system in slot 20, every notebook ends with printed
# analytical-reference error and a "What this unlocks" section
# that points forward to the next capability.
