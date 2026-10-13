#!/usr/bin/env python3
"""Build supplementary Figures S1--S9 from analysis result tables."""

from __future__ import annotations

import argparse
import sys
from statistics import NormalDist
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402

FP = "Frontoparietal"
SITES = ["MGH", "UCLA", "UMinn", "WashU"]
SITE_MARKERS = {"MGH": "o", "UCLA": "s", "UMinn": "^", "WashU": "v"}
STATES5 = fs.STATES + ["EQUAL"]
STATE5_LABELS = {**fs.STATE_LABELS, "EQUAL": "Equal-state"}

INPUTS = {
    # key: (directory key, relative path)
    "pair_ratio": ("a10", "04_pairwise_normalization.csv"),
    "spectra": ("a11", "03_eigenvalue_spectra.csv"),
    "spectrum_summary": ("a10", "02_matrix_structure.csv"),
    "a10_slopes": ("a10", "20_longitudinal_slopes.csv"),
    "s05": ("supp", "Table_S05_network_profile_and_convergence.csv"),
    "s06": ("supp", "Table_S06_FP_network_comparisons.csv"),
    "s08": ("supp", "Table_S08_scanner_site_reference_QC_and_partial_covariance.csv"),
    "conv_boot": ("res", "06_final_checks/tables/profile_convergence_bootstrap.csv"),
    "resid_frac": ("res", "02_primary_profile_and_measurement/tables/gate1_residual_fraction_intervals.csv"),
    "loso": ("res", "03_fp_specificity_and_robustness/tables/fp_relative_leave_one_site_out_slopes.csv"),
    "sens_profile": ("res", "02_primary_profile_and_measurement/tables/sensitivity_network_slopes.csv"),
    "site_profile": ("res", "07_site_profile_shape_diagnostic/tables/03_site_equal_state_profile_slopes.csv"),
    "pooled_profile": ("res", "07_site_profile_shape_diagnostic/tables/03b_pooled_equal_state_profile_slopes.csv"),
    "site_sim": ("res", "07_site_profile_shape_diagnostic/tables/04_site_equal_state_similarity.csv"),
    "site_mag": ("res", "07_site_profile_shape_diagnostic/tables/05_site_equal_state_magnitude.csv"),
    "site_fp": ("res", "07_site_profile_shape_diagnostic/tables/06_site_FP_relative_slopes.csv"),
    "site_state_sim": ("res", "07_site_profile_shape_diagnostic/tables/07_site_state_similarity.csv"),
    "struct_slopes": ("res", "04_structural_sensitivity/tables/structural_slopes.csv"),
    "area_slope": ("res", "04_structural_sensitivity/tables/surface_area_relative_slope.csv"),
    "struct_fmri": ("res", "04_structural_sensitivity/tables/fmri_state_slopes.csv"),
    "s11": ("supp", "Table_S11_cognition_precision_context_and_selection.csv"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--supplement-tables", type=Path, help="Default: RESULTS/manuscript_outputs/tables")
    p.add_argument("--analysis10-tables", type=Path, help="Default: RESULTS/10_noise_amplitude_and_normalization/tables")
    p.add_argument("--analysis11-tables", type=Path, help="Default: RESULTS/11_normalization_followup/tables")
    p.add_argument("--output-dir", type=Path, help="Default: RESULTS/manuscript_outputs/figures")
    p.add_argument("--only", nargs="*", help="Subset of figures, e.g. S1 S4")
    a = p.parse_args()
    r = a.results_dir.expanduser().resolve()
    a.results_dir = r
    a.output_dir = (a.output_dir or r / "manuscript_outputs" / "figures").expanduser().resolve()
    a.dirs = {
        "res": r,
        "supp": a.supplement_tables or r / "manuscript_outputs" / "tables",
        "a10": a.analysis10_tables or r / "10_noise_amplitude_and_normalization" / "tables",
        "a11": a.analysis11_tables or r / "11_normalization_followup" / "tables",
    }
    return a


def load(a, key: str) -> pd.DataFrame:
    d, rel = INPUTS[key]
    path = a.dirs[d] / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing input table for {key}: {path}")
    return pd.read_csv(path)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------
# Shared drawing helpers
# --------------------------------------------------------------------------
def zero_line(ax, axis="h", value=0.0):
    kw = dict(color=fs.ZERO, lw=0.7, ls=(0, (3, 2)), zorder=1)
    (ax.axhline if axis == "h" else ax.axvline)(value, **kw)


def vinterval(ax, x, est, lo, hi, color, marker="o", filled=True, size=4.0, label=None, lw=1.0):
    ax.vlines(x, lo, hi, color=color, lw=lw, zorder=2)
    ax.plot(x, est, ls="none", marker=marker, ms=size, mec=color, mew=0.9,
            mfc=color if filled else "white", zorder=3, label=label)


def hinterval(ax, y, est, lo, hi, color, marker="o", filled=True, size=4.0, label=None, lw=1.0):
    ax.hlines(y, lo, hi, color=color, lw=lw, zorder=2)
    ax.plot(est, y, ls="none", marker=marker, ms=size, mec=color, mew=0.9,
            mfc=color if filled else "white", zorder=3, label=label)


def state_xaxis(ax, labels, rotate=True):
    ax.set_xticks(range(len(labels)))
    if rotate:
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    fs.open_axes(ax)


def network_order(a) -> list[str]:
    s5 = load(a, "s05")
    nets = list(dict.fromkeys(s5[s5.table_component.eq("network_profile")].network))
    check(len(nets) == 10 and FP in nets, "Expected ten networks including Frontoparietal")
    return nets


def legend_below(ax, handles=None, ncol=4, y=-0.25, **kw):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              handletextpad=0.3, columnspacing=1.0, **kw)


# --------------------------------------------------------------------------
# S1  Measurement construction
# --------------------------------------------------------------------------
def figure_s1(a) -> plt.Figure:
    ratios = load(a, "pair_ratio")
    spectra = load(a, "spectra")
    summary = load(a, "spectrum_summary")
    sessions = [f"{s}_{i:02d}" for s in fs.STATES for i in (1, 2, 3)]
    check(set(sessions) <= set(ratios.diagnostic_session), "Pairwise table lacks diagnostic sessions")

    rep_style = {
        "SIGNAL_NORMALIZED": (fs.COLORS["signal_norm"], "o", r"$C_{pq}/(a_pa_q)$  signal"),
        "CANDIDATE_TOTAL_NORMALIZED": (fs.COLORS["total"], "D", r"$C_{pq}/(t_pt_q)$  candidate total"),
    }
    mat_style = {
        "RELEASED_SIGNAL_DIAGONAL": (fs.COLORS["signal_norm"], "Released covariance"),
        "CANDIDATE_TOTAL": (fs.COLORS["total"], r"Released covariance + diag($n_p^2$)"),
    }

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 4.9))
    top = fig.add_gridspec(1, 1, left=0.085, right=0.985, top=0.90, bottom=0.62)
    bot = fig.add_gridspec(1, 4, left=0.085, right=0.985, top=0.47, bottom=0.12, wspace=0.12)

    # A: ratio ranges per session
    ax = fig.add_subplot(top[0, 0])
    xs = []
    for j, sess in enumerate(sessions):
        x0 = j + (j // 3) * 0.8
        xs.append(x0)
        for k, (rep, (col, mk, _)) in enumerate(rep_style.items()):
            r = ratios[(ratios.diagnostic_session == sess) & (ratios.representation == rep)].iloc[0]
            x = x0 + (k - 0.5) * 0.36
            ax.vlines(x, r.ratio_min, r.ratio_max, color=col, lw=0.7, zorder=2)
            ax.vlines(x, r.ratio_p001, r.ratio_p999, color=col, lw=2.6, zorder=2)
            ax.plot(x, r.ratio_median, marker=mk, ms=3.0, mfc="white", mec=col, mew=0.8, ls="none", zorder=3)
    ax.axhspan(-1, 1, color="#EDEDED", zorder=0, lw=0)
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_yticks([-30, -10, -3, -1, 0, 1, 3, 10, 30])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-40, 70)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.split("_")[1] for s in sessions])
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    for i, s in enumerate(fs.STATES):
        ax.text(np.mean(xs[3 * i:3 * i + 3]), -0.15, fs.STATE_LABELS[s], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=fs.BASE)
    ax.set_ylabel("Parcel-pair ratio")
    fs.open_axes(ax)
    ax.tick_params(axis="x", length=0)
    handles = [Line2D([], [], color=c, lw=2.6, marker=m, ms=3.0, mfc="white", mec=c, label=l)
               for c, m, l in rep_style.values()]
    handles.append(Rectangle((0, 0), 1, 1, color="#EDEDED", label="[−1, 1]"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2,
              handletextpad=0.4, columnspacing=1.4)

    # B: lowest part of each eigenvalue spectrum, one panel per state
    k_show = 300
    axes = [fig.add_subplot(bot[0, i]) for i in range(4)]
    for ax_s, state in zip(axes, fs.STATES):
        for sess_i in (1, 2, 3):
            sess = f"{state}_{sess_i:02d}"
            for mat, (col, _) in mat_style.items():
                e = spectra[(spectra.diagnostic_session == sess) & (spectra.matrix == mat)].sort_values("ascending_index")
                check(len(e) == 360, f"Expected 360 eigenvalues for {sess} {mat}")
                srow = summary[(summary.diagnostic_session == sess) & (summary.matrix == mat)].iloc[0]
                check(np.isclose(e.eigenvalue_over_spectral_scale.min(), srow.eigenvalue_min_over_spectral_scale,
                                 rtol=1e-6, atol=1e-12), f"Spectrum minimum mismatch for {sess} {mat}")
                e = e.head(k_show)
                ax_s.plot(e.ascending_index, e.eigenvalue_over_spectral_scale, color=col, lw=0.9, alpha=0.9)
        ax_s.axhline(0, color=fs.TEXT, lw=0.5, zorder=1)
        ax_s.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
        ax_s.set_ylim(-0.4, 0.05)
        ax_s.set_yticks([-0.1, -0.01, -0.001, 0, 0.001, 0.01])
        ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}".replace("-", "\u2212")))
        ax_s.yaxis.set_minor_locator(mticker.NullLocator())
        ax_s.set_xlim(0, k_show)
        ax_s.set_xticks([1, 100, 200, 300])
        ax_s.set_title(fs.STATE_LABELS[state], fontsize=fs.BASE, pad=3)
        fs.open_axes(ax_s)
        if ax_s is not axes[0]:
            ax_s.tick_params(labelleft=False)
    axes[0].set_ylabel(r"Eigenvalue / largest eigenvalue")
    fig.text(0.535, 0.045, "Eigenvalue rank (ascending; lowest 300 of 360)", ha="center", fontsize=fs.BASE)
    handles = [Line2D([], [], color=c, lw=1.2, label=l) for c, l in mat_style.values()]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.505), ncol=2, handlelength=1.4,
               columnspacing=1.4, borderaxespad=0.0)

    fig.canvas.draw()
    fs.panel_label(fig, fig.axes[0], "A", "Parcel-pair ratios in 12 diagnostic sessions", y=0.955)
    fs.panel_label(fig, axes[0], "B", "Lowest eigenvalues", y=0.525)
    return fig


# --------------------------------------------------------------------------
# S2  Fitted-profile convergence
# --------------------------------------------------------------------------
def figure_s2(a) -> plt.Figure:
    boot = load(a, "conv_boot").replace({"state": {"EQUAL_STATE_AVERAGE": "EQUAL"}})
    s5 = load(a, "s05")
    conv = s5[s5.table_component.eq("convergence")].replace({"state": {"EQUAL_STATE_AVERAGE": "EQUAL"}})
    metrics = [("baseline_level_slope_r", "Baseline level–slope correlation", 0.0, (-1.0, 0.4)),
               ("sd_ratio_5y_over_baseline", "Across-network SD, 5 years / baseline", 1.0, (0.84, 1.06))]

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.5))
    grid = fig.add_gridspec(1, 2, left=0.11, right=0.985, top=0.85, bottom=0.2, wspace=0.12)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    ys = np.arange(len(STATES5))[::-1].astype(float)
    ys[-1] -= 0.35  # separate the equal-state row
    for ax, (metric, label, null, lim) in zip(axes, metrics):
        for y, st in zip(ys, STATES5):
            draws = boot.loc[boot.state.eq(st), metric].dropna().to_numpy(float)
            check(len(draws) == 4999, f"Expected 4,999 draws for {st} {metric}")
            r = conv[(conv.state == st) & (conv.metric == metric)].iloc[0]
            col = fs.COLORS["contrast"] if st == "EQUAL" else fs.COLORS["signal_norm"]
            vp = ax.violinplot(draws, positions=[y], vert=False, widths=0.75, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(col)
                body.set_alpha(0.18)
                body.set_edgecolor("none")
            hinterval(ax, y, r.observed, r.bootstrap_q025, r.bootstrap_q975, col, size=4.2)
        zero_line(ax, "v", null)
        ax.set_xlim(*lim)
        ax.set_xlabel(label)
        ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([STATE5_LABELS[s] for s in STATES5] if ax is axes[0] else [])
        fs.open_axes(ax)
        ax.tick_params(axis="y", length=0)
    fig.canvas.draw()
    fs.panel_label(fig, axes[0], "A", "Level–slope correlation", dx=-0.09)
    fs.panel_label(fig, axes[1], "B", "Profile spread ratio", dx=-0.02)
    return fig


# --------------------------------------------------------------------------
# S3  Fixed-weight decomposition and residual fractions
# --------------------------------------------------------------------------
def figure_s3(a) -> plt.Figure:
    q = load(a, "resid_frac")
    qs = q[q.state.isin(fs.STATES)].set_index("state").loc[fs.STATES]
    qmax = q[q.state.eq("Q_MAX")].iloc[0]
    native = qs.raw_fixed_weight_slope_per_5y.to_numpy(float)
    std = qs.standardized_fixed_weight_slope_per_5y.to_numpy(float)
    amp = native - std
    check(np.allclose(np.abs(std / native), qs.q_point.to_numpy(float), atol=1e-6), "q_s != |beta_STD / beta_RAW|")

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.55))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=0.085, right=0.985, top=0.80, bottom=0.25, wspace=0.35)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    x = np.arange(4)
    w = 0.24
    for k, (vals, key, label) in enumerate([(native, "native", r"Native $\beta^{\mathrm{RAW}}$"),
                                             (amp, "signal_amp", r"Amplitude $\beta^{\mathrm{A}}$"),
                                             (std, "signal_norm", r"Signal-normalized $\beta^{\mathrm{STD}}$")]):
        ax_a.bar(x + (k - 1) * w, vals, width=w * 0.92, color=fs.COLORS[key], label=label, zorder=2)
    ax_a.axhline(0, color=fs.TEXT, lw=0.6, zorder=3)
    state_xaxis(ax_a, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_a.set_ylabel("Common log slope per 5 years")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2,
                handletextpad=0.3, columnspacing=0.9, handlelength=1.0)

    for i, s in enumerate(fs.STATES):
        r = qs.loc[s]
        vinterval(ax_b, i, r.q_point, r.simultaneous_95_ci_low, r.simultaneous_95_ci_high, fs.COLORS["signal_norm"])
    for v, ls in ((0.5, (0, (3, 2))), (1.0, (0, (1, 1.5)))):
        ax_b.axhline(v, color=fs.GRAY, lw=0.7, ls=ls, zorder=1)
    state_xaxis(ax_b, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_b.set_ylim(0, max(1.2, float(qmax.Qmax_upper_95) + 0.05))
    ax_b.set_ylabel(r"Residual fraction $q_s$")

    fig.canvas.draw()
    y = ax_a.get_position().y1 + 0.12
    fs.panel_label(fig, ax_a, "A", "Fixed-weight decomposition", y=y)
    fs.panel_label(fig, ax_b, "B", r"Residual fraction, $q_s=|\beta^{\mathrm{STD}}_s/\beta^{\mathrm{RAW}}_s|$",
                   dx=-0.06, y=y)
    return fig


# --------------------------------------------------------------------------
# S4  Profile under both normalizations
# --------------------------------------------------------------------------
def _profile_matrix(slopes: pd.DataFrame, component: str, nets: list[str]):
    t = slopes[(slopes.family == "PROFILE") & (slopes.target == "NETWORK_RELATIVE") & (slopes.component == component)]
    check(len(t) == 40, f"Expected 40 network-relative slopes for {component}")
    est = t.pivot(index="network", columns="state", values="slope_per_5y").loc[nets, fs.STATES]
    lo = t.pivot(index="network", columns="state", values="simultaneous_ci_low").loc[nets, fs.STATES]
    hi = t.pivot(index="network", columns="state", values="simultaneous_ci_high").loc[nets, fs.STATES]
    return est, (lo > 0) | (hi < 0)


def _heatmap(ax, est, excl, nets, norm, cmap, show_labels):
    values = np.column_stack([est.to_numpy(float), est.mean(axis=1).to_numpy(float)])
    xpos = np.array([0, 1, 2, 3, 4.35])
    vmax = norm.vmax
    for j, x in enumerate(xpos):
        for i in range(len(nets)):
            v = values[i, j]
            ax.add_patch(Rectangle((x - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            ax.text(x, i, f"{v:+.3f}", ha="center", va="center", fontsize=5.6,
                    color="white" if abs(v) > 0.6 * vmax else fs.TEXT)
            if j < 4 and bool(excl.iloc[i, j]):
                ax.add_patch(Rectangle((x - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=fs.TEXT, lw=1.0))
    ax.set_xlim(-0.5, 4.85)
    ax.set_ylim(len(nets) - 0.5, -0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([fs.STATE_LABELS[s] for s in fs.STATES] + ["Mean"], rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(len(nets)))
    ax.set_yticklabels([fs.NETWORK_ABBREV[n] for n in nets] if show_labels else [])
    if show_labels:
        lab = ax.get_yticklabels()[nets.index(FP)]
        lab.set_color(fs.COLORS["fp"])
        lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def figure_s4(a) -> plt.Figure:
    nets = network_order(a)
    slopes = load(a, "a10_slopes")
    sig, sig_x = _profile_matrix(slopes, "SIGNAL", nets)
    tot, tot_x = _profile_matrix(slopes, "TOTAL", nets)
    s5 = load(a, "s05")
    ref = s5[s5.table_component.eq("network_profile")].pivot(index="network", columns="state",
                                                              values="relative_profile_slope_per_5y")
    check(np.allclose(ref.loc[nets, fs.STATES].to_numpy(float), sig.to_numpy(float), atol=1e-5),
          "Script-10 signal profile does not reproduce Table S05")

    vmax = float(np.ceil(max(np.abs(sig.to_numpy()).max(), np.abs(tot.to_numpy()).max()) * 100) / 100)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN * 0.78, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], left=0.08, right=0.9,
                            top=0.86, bottom=0.2, wspace=0.12)
    ax_a, ax_b, ax_cb = (fig.add_subplot(grid[0, i]) for i in range(3))
    _heatmap(ax_a, sig, sig_x, nets, norm, cmap, True)
    _heatmap(ax_b, tot, tot_x, nets, norm, cmap, False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.outline.set_linewidth(0.5)
    step = 0.04 if vmax >= 0.08 else 0.02
    cb.set_ticks(np.arange(-np.floor(vmax / step) * step, vmax + 1e-9, step))
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
    cb.ax.tick_params(labelsize=fs.SMALL, length=2)

    fig.canvas.draw()
    ytitle = ax_a.get_position().y1 + 0.05
    fs.panel_label(fig, ax_a, "A", "Signal normalization", dx=-0.06, y=ytitle)
    fs.panel_label(fig, ax_b, "B", "Candidate total normalization", dx=-0.01, y=ytitle)
    return fig


# --------------------------------------------------------------------------
# S5  Leave-one-site-out FP-relative slopes
# --------------------------------------------------------------------------
def figure_s5(a) -> plt.Figure:
    loso = load(a, "loso")
    s6 = load(a, "s06")
    full = s6[s6.table_component.eq("fp_state_slopes")]
    panels = [("RAW_FULL", "native", "Native covariance"),
              ("AMPLITUDE_STANDARDIZED", "signal_norm", "Signal normalization")]
    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.6))
    grid = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.83, bottom=0.3, wspace=0.22)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    offsets = np.linspace(-0.3, 0.3, 5)
    for ax, (rep, key, _) in zip(axes, panels):
        col = fs.COLORS[key]
        f = full[full.scenario.eq(rep)].set_index("state").loc[fs.STATES]
        for i, s in enumerate(fs.STATES):
            vinterval(ax, i + offsets[0], f.loc[s, "slope_per_5y"], f.loc[s, "simultaneous_ci_low"],
                      f.loc[s, "simultaneous_ci_high"], col, size=4.4)
            for j, site in enumerate(SITES, start=1):
                r = loso[(loso.representation == rep) & (loso.state == s) & (loso.site_left_out == site)].iloc[0]
                vinterval(ax, i + offsets[j], r.slope_per_5y, r.simultaneous_ci_low, r.simultaneous_ci_high,
                          col, marker=SITE_MARKERS[site], filled=False, size=3.4, lw=0.8)
        zero_line(ax)
        state_xaxis(ax, [fs.STATE_LABELS[s] for s in fs.STATES])
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(min(lo, -0.02), hi)
    axes[0].set_ylabel("FP-relative log slope per 5 years")
    handles = [Line2D([], [], marker="o", ls="none", ms=4.4, mfc=fs.TEXT, mec=fs.TEXT, label="All sites")]
    handles += [Line2D([], [], marker=SITE_MARKERS[s], ls="none", ms=3.4, mfc="white", mec=fs.TEXT,
                       label=f"Without {s}") for s in SITES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=5, handletextpad=0.3,
               columnspacing=1.2)
    fig.canvas.draw()
    for ax, (_, _, title), lab in zip(axes, panels, "AB"):
        fs.panel_label(fig, ax, lab, title, dx=-0.06 if lab == "A" else -0.035)
    return fig


# --------------------------------------------------------------------------
# S6  Reference deletion and partial covariance
# --------------------------------------------------------------------------
def figure_s6(a) -> plt.Figure:
    s8 = load(a, "s08")
    ref = s8[s8.table_component.eq("reference_sensitivity")].copy()
    sens = load(a, "sens_profile")
    z = NormalDist().inv_cdf(1 - 0.05 / 8)
    def fp_rel(sc):
        t = sens[(sens.scenario == sc) & (sens.network == FP)].set_index("state").loc[fs.STATES]
        est, se = t.relative_profile_slope_per_5y * 10 / 9, t.se * 10 / 9  # FP minus mean of other nine
        return est, est - z * se, est + z * se
    nets = network_order(a)
    state_cols = [f"{s}_slope" for s in fs.STATES]

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.75))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], left=0.12, right=0.985, top=0.86, bottom=0.27, wspace=0.3)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])

    rows = ["ALL_NINE"] + [f"LEAVE_OUT_{n}" for n in nets if n != FP]
    labels = ["All nine"] + [f"Without {fs.NETWORK_ABBREV[n]}" for n in nets if n != FP]
    ref = ref.set_index("reference_choice").loc[rows]
    base = float(ref.loc["ALL_NINE", "mean_four_state_slope"])
    for y, (rid, r) in enumerate(ref.iterrows()):
        sv = r[state_cols].to_numpy(float)
        ax_a.hlines(y, sv.min(), sv.max(), color=fs.COLORS["signal_norm"], lw=0.8, alpha=0.6)
        ax_a.plot(sv, [y] * 4, "|", ms=4, color=fs.COLORS["signal_norm"], alpha=0.6)
        ax_a.plot(r.mean_four_state_slope, y, "o", ms=4.2, color=fs.COLORS["fp"] if rid == "ALL_NINE"
                  else fs.COLORS["signal_norm"])
    ax_a.axvline(base, color=fs.COLORS["fp"], lw=0.7, ls=(0, (3, 2)))
    ax_a.set_yticks(range(len(rows)))
    ax_a.set_yticklabels(labels)
    ax_a.set_ylim(len(rows) - 0.5, -0.6)
    ax_a.set_xlim(0, None)
    ax_a.set_xlabel("FP-relative log slope per 5 years")
    fs.open_axes(ax_a)
    ax_a.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="none", ms=4.2, color=fs.COLORS["signal_norm"], label="Equal-state mean"),
               Line2D([], [], marker="|", ls="-", lw=0.8, ms=4, color=fs.COLORS["signal_norm"], alpha=0.6,
                      label="State-specific range")]
    legend_below(ax_a, handles, ncol=2, y=-0.2)

    x = np.arange(4)
    vinterval(ax_b, x - 0.13, *fp_rel("FULL_PARTIAL_MATCHED"), fs.COLORS["native"], label="Full covariance")
    vinterval(ax_b, x + 0.13, *fp_rel("PARTIAL_MATCHED"), fs.COLORS["native"], marker="s", filled=False,
              label="Partial covariance")
    zero_line(ax_b)
    state_xaxis(ax_b, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_b.set_ylabel("FP-relative log slope per 5 years")
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.1, handletextpad=0.3)

    fig.canvas.draw()
    y = ax_a.get_position().y1 + 0.09
    fs.panel_label(fig, ax_a, "A", "Reference-network deletion", dx=-0.1, y=y)
    fs.panel_label(fig, ax_b, "B", "Native full vs. partial covariance, same rows", dx=-0.06, y=y)
    return fig


# --------------------------------------------------------------------------
# S7  Site profile shape
# --------------------------------------------------------------------------
def figure_s7(a) -> plt.Figure:
    nets = network_order(a)
    prof = load(a, "site_profile")
    pooled = load(a, "pooled_profile").set_index("network").loc[nets]
    sim = load(a, "site_sim")
    mag = load(a, "site_mag").set_index("site").loc[SITES]
    sfp = load(a, "site_fp").set_index("site").loc[SITES]
    ssim = load(a, "site_state_sim")

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 5.0))
    top = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], left=0.085, right=0.985, top=0.92, bottom=0.55, wspace=0.28)
    bot = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.40, bottom=0.08, wspace=0.45)
    ax_a, ax_d = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    ax_b, ax_c, ax_e = (fig.add_subplot(bot[0, i]) for i in range(3))

    # A: equal-state profiles
    y = np.arange(len(nets))
    ax_a.barh(y, pooled.slope_per_5y, height=0.62, color="#DADADA", zorder=1, label="Pooled")
    for k, site in enumerate(SITES):
        p = prof[prof.site.eq(site)].set_index("network").loc[nets]
        ax_a.plot(p.slope_per_5y, y + (k - 1.5) * 0.14, ls="none", marker=SITE_MARKERS[site], ms=3.2,
                  mfc="white", mec=fs.TEXT, mew=0.7, zorder=3, label=site)
    zero_line(ax_a, "v")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([fs.NETWORK_ABBREV[n] for n in nets])
    lab = ax_a.get_yticklabels()[nets.index(FP)]
    lab.set_color(fs.COLORS["fp"])
    lab.set_fontweight("bold")
    ax_a.set_ylim(len(nets) - 0.5, -0.5)
    ax_a.set_xlabel("Equal-state relative log slope per 5 years")
    fs.open_axes(ax_a)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(loc="lower right", fontsize=6.2, borderaxespad=0.1, handletextpad=0.2, labelspacing=0.25)

    # D: state-resolved similarity
    offs = np.linspace(-0.27, 0.27, 4)
    for i, s in enumerate(fs.STATES):
        for k, site in enumerate(SITES):
            r = ssim[(ssim.site == site) & (ssim.state == s)].iloc[0]
            vinterval(ax_d, i + offs[k], r.r_with_other_three_state_profile, r.simultaneous_95_low,
                      r.simultaneous_95_high, fs.COLORS["signal_norm"], marker=SITE_MARKERS[site], filled=False,
                      size=3.4, lw=0.8)
    zero_line(ax_d)
    state_xaxis(ax_d, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_d.set_ylim(-0.7, 1.02)
    ax_d.set_ylabel("r with other three sites")
    ax_d.legend(handles=[Line2D([], [], marker=SITE_MARKERS[s], ls="none", ms=3.4, mfc="white",
                                mec=fs.COLORS["signal_norm"], label=s) for s in SITES],
                loc="lower left", ncol=2, borderaxespad=0.2, handletextpad=0.2, columnspacing=0.8)

    # B: equal-state similarity; C: magnitude; E: FP-relative slope
    for ax, est, lo, hi, null, ylab in [
        (ax_b, [sim.set_index("label").loc[f"{s}__vs__OTHER_THREE", "estimate"] for s in SITES],
         [sim.set_index("label").loc[f"{s}__vs__OTHER_THREE", "simultaneous_95_low"] for s in SITES],
         [sim.set_index("label").loc[f"{s}__vs__OTHER_THREE", "simultaneous_95_high"] for s in SITES],
         0.0, "r with other three sites"),
        (ax_c, mag.ratio_estimate, mag.ratio_simultaneous_95_low, mag.ratio_simultaneous_95_high, 1.0,
         "Site / pooled RMS"),
        (ax_e, sfp.fp_relative_slope_per_5y, sfp.simultaneous_95_low, sfp.simultaneous_95_high, 0.0,
         "FP-relative log slope per 5 years"),
    ]:
        for i, site in enumerate(SITES):
            col = fs.COLORS["fp"] if ax is ax_e else fs.COLORS["signal_norm"]
            vinterval(ax, i, np.asarray(est)[i], np.asarray(lo)[i], np.asarray(hi)[i], col,
                      marker=SITE_MARKERS[site], size=4.0)
        zero_line(ax, "h", null)
        state_xaxis(ax, SITES, rotate=False)
        ax.set_ylabel(ylab)
    ax_b.set_ylim(0, 1.02)
    ax_e.set_ylim(0, None)

    fig.canvas.draw()
    fs.panel_label(fig, ax_a, "A", "Equal-state site profiles", dx=-0.07, y=0.965)
    fs.panel_label(fig, ax_d, "D", "State-resolved similarity", dx=-0.06, y=0.965)
    yb = ax_b.get_position().y1 + 0.04
    fs.panel_label(fig, ax_b, "B", "Profile similarity", dx=-0.07, y=yb)
    fs.panel_label(fig, ax_c, "C", "Profile magnitude", dx=-0.07, y=yb)
    fs.panel_label(fig, ax_e, "E", "FP-relative slope", dx=-0.07, y=yb)
    return fig


# --------------------------------------------------------------------------
# S8  Structure
# --------------------------------------------------------------------------
def figure_s8(a) -> plt.Figure:
    st = load(a, "struct_slopes").set_index("outcome")
    area = load(a, "area_slope").iloc[0]
    fm = load(a, "struct_fmri")
    scen = [("THICKNESS_MATCHED_ORIGINAL", "o", True, "No structural terms"),
            ("THICKNESS_WITHIN_BETWEEN", "D", False, "+ thickness"),
            ("VOLUME_WITHIN_BETWEEN", "s", False, "+ volume"),
            ("THICKNESS_FINAL_SAME_SCANNER", "^", False, "+ thickness, same scanner")]

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.33, 0.55, 1.6], left=0.08, right=0.985, top=0.84, bottom=0.3,
                            wspace=0.55)
    ax_a1, ax_a2, ax_b = (fig.add_subplot(grid[0, i]) for i in range(3))

    t = st.loc["Y_THICK_REL"]
    vinterval(ax_a1, 0, t.slope_per_5y, t.ci_low, t.ci_high, fs.COLORS["fp"])
    ax_a1.set_ylabel("mm per 5 years")
    state_xaxis(ax_a1, ["Thickness"], rotate=True)
    ax_a1.set_ylim(0, 0.025)
    zero_line(ax_a1)
    v = st.loc["Y_VOLUME_REL"]
    rows = [(v.slope_per_5y, v.ci_low, v.ci_high), (area.slope_per_5y, area.ci_low, area.ci_high)]
    for i, (e, lo, hi) in enumerate(rows):
        vinterval(ax_a2, i, e, lo, hi, fs.COLORS["fp"])
    zero_line(ax_a2)
    state_xaxis(ax_a2, ["Volume", "Area"], rotate=True)
    ax_a2.set_ylabel("Log units per 5 years")

    offs = np.linspace(-0.27, 0.27, len(scen))
    for k, (sc, mk, filled, label) in enumerate(scen):
        t = fm[fm.scenario.eq(sc)].set_index("state").loc[fs.STATES]
        vinterval(ax_b, np.arange(4) + offs[k], t.slope_per_5y, t.simultaneous_ci_low, t.simultaneous_ci_high,
                  fs.COLORS["fp"], marker=mk, filled=filled, size=3.8, lw=0.9, label=label)
    zero_line(ax_b)
    state_xaxis(ax_b, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_b.set_ylabel("FP-relative fMRI log slope\nper 5 years")
    legend_below(ax_b, ncol=2, y=-0.3)

    fig.canvas.draw()
    y = ax_b.get_position().y1 + 0.06
    fs.panel_label(fig, ax_a1, "A", "FP-relative structure", dx=-0.07, y=y)
    fs.panel_label(fig, ax_b, "B", "FP-relative fMRI slopes with structural adjustment", dx=-0.07, y=y)
    return fig


# --------------------------------------------------------------------------
# S9  Cognition
# --------------------------------------------------------------------------
SELECTION_ROWS = [
    ("n_eligible_fmri_visits", None, "Eligible fMRI visits"),
    ("fmri_followup_years", None, "Imaging follow-up"),
    ("baseline_age", None, "Baseline age"),
    ("sex", "F", "Female"),
    ("first_equal_state_mean_fp_relative", None, "First FP-relative, equal-state"),
    ("first_fp_relative_REST", None, "First FP-relative, REST"),
    ("first_fp_relative_CARIT", None, "First FP-relative, CARIT"),
    ("first_fp_relative_FACENAME", None, "First FP-relative, FaceName"),
    ("first_fp_relative_VISMOTOR", None, "First FP-relative, visual–motor"),
]


def figure_s9(a) -> plt.Figure:
    s11 = load(a, "s11")
    wb = s11[s11.table_component.eq("within_between_coefficients")]
    bal = s11[s11.table_component.eq("selection_balance")]

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.085, right=0.985, top=0.84, bottom=0.27, wspace=0.55)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for k, (comp, filled, label) in enumerate([("WITHIN_PERSON", True, "Within person"),
                                               ("BETWEEN_PERSON", False, "Between person")]):
        t = wb[wb.component.eq(comp)].set_index("state").loc[fs.STATES]
        vinterval(ax_a, np.arange(4) + (k - 0.5) * 0.26, t.coefficient_per_0_1_log_unit,
                  t.simultaneous_ci_low / 10.0, t.simultaneous_ci_high / 10.0, fs.COLORS["fp"],
                  filled=filled, label=label)
        check(np.allclose(t.coefficient_per_1_log_unit / 10.0, t.coefficient_per_0_1_log_unit), "Scale mismatch")
    zero_line(ax_a)
    state_xaxis(ax_a, [fs.STATE_LABELS[s] for s in fs.STATES])
    ax_a.set_ylabel("FluidIQ per 0.1 FP-relative log unit")
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.1, handletextpad=0.3)

    vals, labels = [], []
    for var, level, lab in SELECTION_ROWS:
        r = bal[bal.variable.eq(var)]
        r = r[r.level.eq(level)] if level else r[r.level.isna()]
        check(len(r) == 1, f"Selection row not unique: {var} {level}")
        vals.append(float(r.standardized_difference.iloc[0]))
        labels.append(lab)
    y = np.arange(len(vals))
    ax_b.hlines(y, 0, vals, color=fs.GRAY, lw=1.0)
    ax_b.plot(vals, y, "o", ms=4.2, color=fs.TEXT)
    ax_b.axvline(0, color=fs.TEXT, lw=0.6)
    for v in (-0.1, 0.1):
        ax_b.axvline(v, color=fs.ZERO, lw=0.6, ls=(0, (3, 2)))
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.set_ylim(len(vals) - 0.5, -0.5)
    ax_b.set_xlabel("Standardized difference, included − outside")
    fs.open_axes(ax_b)
    ax_b.tick_params(axis="y", length=0)

    fig.canvas.draw()
    y = ax_a.get_position().y1 + 0.1
    fs.panel_label(fig, ax_a, "A", "FP-relative coupling with FluidIQ", dx=-0.07, y=y)
    fs.panel_label(fig, ax_b, "B", "Repeated-cognition selection", dx=-0.17, y=y)
    return fig


FIGURES = {
    "S1": (figure_s1, "Figure_S1_measurement_construction"),
    "S2": (figure_s2, "Figure_S2_profile_convergence"),
    "S3": (figure_s3, "Figure_S3_residual_fraction"),
    "S4": (figure_s4, "Figure_S4_profile_two_normalizations"),
    "S5": (figure_s7, "Figure_S5_site_profile"),
    "S6": (figure_s5, "Figure_S6_leave_one_site_out"),
    "S7": (figure_s6, "Figure_S7_reference_and_partial_covariance"),
    "S8": (figure_s8, "Figure_S8_structure"),
    "S9": (figure_s9, "Figure_S9_cognition"),
}


def main() -> None:
    a = parse_args()
    fs.configure()
    for key, (fn, name) in FIGURES.items():
        if a.only and key not in a.only:
            continue
        fig = fn(a)
        paths = fs.save(fig, a.output_dir, name)
        plt.close(fig)
        print(f"[{key}] wrote {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
