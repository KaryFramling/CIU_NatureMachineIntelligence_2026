"""
CIU vs SHAP: Wealth toy example
================================
Accompanies: "Importance Is Not Influence: ..."
             Främling, K. (Nature Machine Intelligence, submitted)

Produces the following figure files (in ./figures/):
  io_salary.pdf       -- IO plot, salary feature
  io_savings.pdf      -- IO plot, savings feature
  io_debt.pdf         -- IO plot, debt feature
  io_all.pdf          -- All three IO plots in one row
  pi_plot_wealth.pdf  -- CIU Potential Influence (PI) plot
  shap_wealth.pdf     -- SHAP attributions
  pi_vs_shap.pdf      -- PI plot + SHAP side by side

Requires:  numpy, matplotlib  (pip install numpy matplotlib)

Run:  python ciu_vs_shap_wealth.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name: str, fig=None):
    """Save current (or given) figure as PDF and PNG."""
    f = fig or plt.gcf()
    f.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches="tight")
    f.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches="tight", dpi=150)
    print(f"  saved  figures/{name}.pdf  +  .png")


# ── colour palette (matches ciu R package + TikZ figure) ─────────────────────
# CIU package convention:
#   PI background bar: light gray (potential range)
#   PI solid bar:      green = high CU (good), red = low CU (bad)
#   IO curve:          blue
#   Instance dot:      orange
C_BLUE      = "#378ADD"
C_ORANGE    = "#E07B00"
C_GREEN     = "#1D9E75"   # matches ciu package
C_RED       = "#E24B4A"   # matches ciu package
C_GRAY      = "#888780"
C_PI_BG     = "#CCCCCC"   # light gray for CI potential-range bar (ciu package style)

# Matplotlib style: clean, paper-ready
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.dpi":        150,
})


# =============================================================================
# 1.  MODEL
# =============================================================================

def wealth_model(X: np.ndarray) -> np.ndarray:
    """
    Linear wealth model.

    wealth = 0.6*salary + 0.3*savings - 0.4*debt

    Feature ranges (thousands):
      salary  in [20,  200]
      savings in [0,   500]
      debt    in [0,   200]
    """
    return 0.6 * X[:, 0] + 0.3 * X[:, 1] - 0.4 * X[:, 2]


FEATURE_NAMES  = ["salary", "savings", "debt"]
FEATURE_RANGES = {
    "salary":  (20,  200),
    "savings": (0,   500),
    "debt":    (0,   200),
}
FEATURE_LABELS = {
    "salary":  "salary (k)",
    "savings": "savings (k)",
    "debt":    "debt (k)",
}

# Analytical global extremes (linear model: extremes at corners of input space)
Y_GLOBAL_MIN = wealth_model(np.array([[20,   0, 200]]))[0]   # = -68 k
Y_GLOBAL_MAX = wealth_model(np.array([[200, 500,   0]]))[0]  # = 270 k


# =============================================================================
# 2.  UTILITY TRANSFORMATION  u(y) in [0, 1]
# =============================================================================

def u(y):
    """
    Affine utility transformation: y -> [0, 1].

    Maps [Y_GLOBAL_MIN, Y_GLOBAL_MAX] -> [0, 1] linearly.
    By Corollary 1 (Framling 2022), when {i} = {I} = all features,
    the CI denominator equals exactly 1, so we never need to
    estimate it by sampling.
    """
    return (y - Y_GLOBAL_MIN) / (Y_GLOBAL_MAX - Y_GLOBAL_MIN)


# =============================================================================
# 3.  CIU  (model-agnostic perturbation estimator)
# =============================================================================

def ciu_explain(
    instance: np.ndarray,
    model,
    feature_names  = FEATURE_NAMES,
    feature_ranges = FEATURE_RANGES,
    n_samples: int = 2000,
    cu_ref: float  = 0.5,
) -> dict:
    """
    Compute CI, CU and contextual influence (phi) for each feature.

    Definitions (Framling 2022, 2024):
      CI  = umax_i - umin_i        (denominator = 1 by Corollary 1)
      CU  = (u(y_instance) - umin_i) / CI
      phi = CI * (CU - cu_ref)

    Parameters
    ----------
    instance      : shape (d,) feature vector of the explained instance
    model         : callable X (n, d) -> y (n,)
    feature_names : ordered list of feature names
    feature_ranges: dict name -> (lo, hi)
    n_samples     : perturbation samples per feature
    cu_ref        : contrastive reference utility (0.5 = neutral)

    Returns
    -------
    dict  feature_name -> {CI, CU, phi, umin_i, umax_i, ymin_i, ymax_i, xs, ys}
    """
    u_inst = float(u(model(instance.reshape(1, -1))[0]))
    results = {}

    for idx, fname in enumerate(feature_names):
        lo, hi = feature_ranges[fname]

        xs = np.linspace(lo, hi, n_samples)
        X_vary = np.tile(instance, (n_samples, 1))
        X_vary[:, idx] = xs
        ys = model(X_vary)

        umin_i = float(u(ys.min()))
        umax_i = float(u(ys.max()))

        ci  = umax_i - umin_i
        cu  = (u_inst - umin_i) / ci  if ci > 1e-12 else 0.0
        phi = ci * (cu - cu_ref)

        results[fname] = dict(
            CI=ci, CU=cu, phi=phi,
            umin_i=umin_i, umax_i=umax_i,
            ymin_i=float(ys.min()), ymax_i=float(ys.max()),
            xs=xs, ys=ys,
        )

    return results


# =============================================================================
# 4.  SHAP  (exact formula for linear models)
#
#     For y = w . x + b, the Shapley value of feature i is:
#       phi_i = w_i * (x_i - E[x_i])
#     At x_i = E[x_i]  =>  phi_i = 0  exactly, regardless of w_i.
# =============================================================================

def shap_linear_exact(instance, coefs, pop_means):
    """Exact Shapley values for a linear model (no library required)."""
    return coefs * (instance - pop_means)


# =============================================================================
# 5.  DATA
# =============================================================================

np.random.seed(42)
N_POP       = 10_000
salary_pop  = np.random.uniform(20,  200, N_POP)
savings_pop = np.random.uniform(0,   500, N_POP)
debt_pop    = np.random.uniform(0,   200, N_POP)
pop_means   = np.array([salary_pop.mean(), savings_pop.mean(), debt_pop.mean()])

# Explained instance: salary exactly at population mean
instance = np.array([pop_means[0], 50.0, 30.0])
y_inst   = wealth_model(instance.reshape(1, -1))[0]
u_inst   = float(u(y_inst))

coefs     = np.array([0.6, 0.3, -0.4])
shap_vals = shap_linear_exact(instance, coefs, pop_means)

# ── console summary ───────────────────────────────────────────────────────────
print("=" * 62)
print("Wealth model:  wealth = 0.6*salary + 0.3*savings - 0.4*debt")
print(f"Global range:  [{Y_GLOBAL_MIN:.0f}k, {Y_GLOBAL_MAX:.0f}k]")
print(f"Pop. means:    salary={pop_means[0]:.1f}k  "
      f"savings={pop_means[1]:.1f}k  debt={pop_means[2]:.1f}k")
print(f"Instance:      salary={instance[0]:.1f}k (=mean)  "
      f"savings={instance[1]:.0f}k  debt={instance[2]:.0f}k")
print(f"Predicted:     wealth={y_inst:.1f}k   u(wealth)={u_inst:.3f}")
print("=" * 62)

ciu = ciu_explain(instance, wealth_model)

print(f"\n{'':12} {'CI':>6}  {'CU':>6}  {'phi':>8}   SHAP")
for fname in FEATURE_NAMES:
    r  = ciu[fname]
    sv = shap_vals[FEATURE_NAMES.index(fname)]
    print(f"  {fname:<10} {r['CI']:>6.3f}  {r['CU']:>6.3f}  "
          f"{r['phi']:>8.4f}   {sv:>+8.2f}k")
print(f"\n  Salary SHAP = 0 exactly  (salary = population mean by construction)")


# =============================================================================
# 6.  PLOTTING HELPERS
# =============================================================================

def _solid_color(cu):
    if cu <= 0.35: return C_RED
    if cu >= 0.65: return C_GREEN
    return C_GRAY


def _io_ax(ax, fname: str, r: dict, instance_val: float):
    """
    Draw a single IO plot on ax.

    CI and CU are geometrically readable:
      CI  = (ymax_i - ymin_i) / (Y_GLOBAL_MAX - Y_GLOBAL_MIN)
      CU  = position of the instance dot within [ymin_i, ymax_i]
    """
    xs, ys = r["xs"], r["ys"]

    # IO curve
    ax.plot(xs, ys, color=C_BLUE, lw=1.8, zorder=3)

    # ymin / ymax bracket lines
    ax.axhline(r["ymax_i"], color=C_GREEN, lw=1.2, ls="--", zorder=2)
    ax.axhline(r["ymin_i"], color=C_RED,   lw=1.2, ls="--", zorder=2)

    # Neutral line at contextual midpoint of [ymin_i, ymax_i]  (CU = 0.5 for this feature)
    # This is where CU = 0.5 *given the current context*, not the global midpoint.
    y_neutral = r["ymin_i"] + 0.5 * (r["ymax_i"] - r["ymin_i"])
    ax.axhline(y_neutral, color=C_GRAY, lw=0.8, ls="--", zorder=2, alpha=0.6)

    # Vertical dashed line at instance x value
    ax.axvline(instance_val, color=C_GRAY, lw=0.8, ls="--", zorder=2)

    # Instance dot
    ax.scatter([instance_val], [y_inst], color=C_ORANGE,
               s=42, zorder=5, edgecolors="white", linewidths=0.8)

    # Right-side CI bracket arrow
    x_span = xs[-1] - xs[0]
    xr = xs[-1] + 0.06 * x_span
    ax.annotate("", xy=(xr, r["ymax_i"]), xytext=(xr, r["ymin_i"]),
                arrowprops=dict(arrowstyle="<->", color=C_BLUE, lw=1.4),
                annotation_clip=False)

    # CI / CU text next to bracket
    mid_y = (r["ymax_i"] + r["ymin_i"]) / 2
    ax.text(xr + 0.02 * x_span, mid_y,
            f"CI={r['CI']:.2f}\nCU={r['CU']:.2f}",
            va="center", ha="left", fontsize=7.5,
            color=C_BLUE, clip_on=False)

    ax.set_xlabel(FEATURE_LABELS[fname], fontsize=8)
    ax.set_ylabel("wealth (k)", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))

    # y-axis limits: bracket + padding
    pad = 0.15 * max(r["ymax_i"] - r["ymin_i"], 10)
    ax.set_ylim(r["ymin_i"] - pad, r["ymax_i"] + pad)


# IO plot shared legend elements
IO_LEGEND = [
    plt.Line2D([0], [0], color=C_BLUE,   lw=1.8,          label="f(x)"),
    plt.Line2D([0], [0], color=C_GREEN,  lw=1.2, ls="--", label="ymax"),
    plt.Line2D([0], [0], color=C_RED,    lw=1.2, ls="--", label="ymin"),
    plt.Line2D([0], [0], color=C_GRAY,   lw=0.8, ls="--", label="neutral (CU=0.5)"),
    plt.scatter([], [], color=C_ORANGE,  s=30,             label="instance"),
]

PI_LEGEND = [
    mpatches.Patch(facecolor=C_PI_BG, edgecolor="#999999",
                   label="CI — potential range"),
    mpatches.Patch(facecolor=C_GREEN, alpha=0.88,
                   label="CI x CU — current utility (green=good, red=bad)"),
]


# =============================================================================
# 7.  IO PLOTS
# =============================================================================

print("\nGenerating IO plots ...")

# Individual panels
for fname in FEATURE_NAMES:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _io_ax(ax, fname, ciu[fname], instance[FEATURE_NAMES.index(fname)])
    fig.legend(handles=IO_LEGEND, loc="lower center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    savefig(f"io_{fname}", fig)
    plt.close(fig)

# Combined row
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
for ax, fname in zip(axes, FEATURE_NAMES):
    _io_ax(ax, fname, ciu[fname], instance[FEATURE_NAMES.index(fname)])
    ax.set_title(fname, fontsize=9)
    if ax is not axes[0]:
        ax.set_ylabel("")
fig.legend(handles=IO_LEGEND, loc="lower center", ncol=6,
           fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
savefig("io_all", fig)
plt.close(fig)


# =============================================================================
# 8.  PI PLOT
# =============================================================================

print("Generating PI plot ...")

CI_vals = [ciu[f]["CI"] for f in FEATURE_NAMES]
CU_vals = [ciu[f]["CU"] for f in FEATURE_NAMES]
y_pos   = np.arange(len(FEATURE_NAMES))

fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.barh(y_pos, CI_vals, height=0.45,
        color=C_PI_BG, alpha=1.0, edgecolor="#999999", linewidth=0.8)
ax.barh(y_pos, [ci * cu for ci, cu in zip(CI_vals, CU_vals)],
        height=0.45,
        color=[_solid_color(cu) for cu in CU_vals], alpha=0.88)
ax.set_yticks(y_pos)
ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax.set_xlim(0, 0.75)
ax.set_xlabel("CI  in  [0, 1]", fontsize=8)
ax.axvline(0, color="black", lw=0.5)
for i, fname in enumerate(FEATURE_NAMES):
    r = ciu[fname]
    ax.text(r["CI"] + 0.01, i,
            f"CI={r['CI']:.2f}, CU={r['CU']:.2f}",
            va="center", fontsize=7.5, color="#333")
ax.legend(handles=PI_LEGEND, fontsize=7.5, loc="upper right", frameon=False)
fig.tight_layout()
savefig("pi_plot_wealth", fig)
plt.close(fig)


# =============================================================================
# 9.  SHAP PLOT
# =============================================================================

print("Generating SHAP plot ...")

shap_colors = [C_GREEN if v > 0.5 else C_RED if v < -0.5 else C_GRAY
               for v in shap_vals]

fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.barh(y_pos, shap_vals, height=0.45, color=shap_colors, alpha=0.88)
ax.set_yticks(y_pos)
ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax.set_xlabel("attribution (k)", fontsize=8)
ax.axvline(0, color="black", lw=0.8)
for i, (fname, sv) in enumerate(zip(FEATURE_NAMES, shap_vals)):
    if abs(sv) < 0.5:
        ax.text(2, i, "0.00  (salary = pop. mean)",
                va="center", ha="left", fontsize=7.5, color="#333")
    elif sv > 0:
        ax.text(sv + 1, i, f"{sv:.1f}k",
                va="center", ha="left", fontsize=7.5, color="#333")
    else:
        ax.text(sv - 1, i, f"{sv:.1f}k",
                va="center", ha="right", fontsize=7.5, color="#333")
fig.tight_layout()
savefig("shap_wealth", fig)
plt.close(fig)


# =============================================================================
# 10. PI + SHAP  SIDE BY SIDE
# =============================================================================

print("Generating combined PI vs SHAP figure ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2.8))

# PI (left)
ax1.set_title("CIU — Potential Influence (PI) plot", fontsize=9)
ax1.barh(y_pos, CI_vals, height=0.45,
         color=C_PI_BG, alpha=1.0, edgecolor="#999999", linewidth=0.8)
ax1.barh(y_pos, [ci * cu for ci, cu in zip(CI_vals, CU_vals)],
         height=0.45,
         color=[_solid_color(cu) for cu in CU_vals], alpha=0.88)
ax1.set_yticks(y_pos); ax1.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax1.set_xlim(0, 0.75); ax1.set_xlabel("CI  in  [0, 1]", fontsize=8)
ax1.axvline(0, color="black", lw=0.5)
for i, fname in enumerate(FEATURE_NAMES):
    r = ciu[fname]
    ax1.text(r["CI"] + 0.005, i,
             f"CI={r['CI']:.2f}, CU={r['CU']:.2f}",
             va="center", fontsize=7.5, color="#333")
ax1.legend(handles=PI_LEGEND, fontsize=7, loc="upper right", frameon=False)

# SHAP (right)
ax2.set_title("SHAP — feature attributions", fontsize=9)
ax2.barh(y_pos, shap_vals, height=0.45, color=shap_colors, alpha=0.88)
ax2.set_yticks(y_pos); ax2.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax2.set_xlabel("attribution (k)", fontsize=8)
ax2.axvline(0, color="black", lw=0.8)
for i, (fname, sv) in enumerate(zip(FEATURE_NAMES, shap_vals)):
    if abs(sv) < 0.5:
        ax2.text(2, i, "0.00  (salary = pop. mean)",
                 va="center", ha="left", fontsize=7.5, color="#333")
    elif sv > 0:
        ax2.text(sv + 1, i, f"{sv:.1f}k",
                 va="center", ha="left", fontsize=7.5, color="#333")
    else:
        ax2.text(sv - 1, i, f"{sv:.1f}k",
                 va="center", ha="right", fontsize=7.5, color="#333")

fig.tight_layout()
savefig("pi_vs_shap", fig)
plt.close(fig)

print("\nDone. All figures written to ./figures/")
