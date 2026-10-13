#!/usr/bin/env python3
"""Build Figures 2--4 from assembled source CSVs and normalization results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs  # noqa: E402

FP = "Frontoparietal"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--supplement-tables", type=Path, help="Default: RESULTS/manuscript_outputs/tables")
    p.add_argument("--analysis10-tables", type=Path, help="Default: RESULTS/10_noise_amplitude_and_normalization/tables")
    p.add_argument("--analysis11-tables", type=Path, help="Default: RESULTS/11_normalization_followup/tables")
    p.add_argument("--output-dir", type=Path, help="Default: RESULTS/manuscript_outputs/figures")
    a = p.parse_args()
    r = a.results_dir.expanduser().resolve()
    a.results_dir = r
    a.output_dir = (a.output_dir or r / "manuscript_outputs" / "figures").expanduser().resolve()
    a.supplement_tables = a.supplement_tables or r / "manuscript_outputs" / "tables"
    a.analysis10_tables = a.analysis10_tables or r / "10_noise_amplitude_and_normalization" / "tables"
    a.analysis11_tables = a.analysis11_tables or r / "11_normalization_followup" / "tables"
    return a


def need(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path)


def by_state(df: pd.DataFrame, states=fs.STATES) -> pd.DataFrame:
    out = df.set_index("state").reindex(states)
    if out["slope_per_5y"].isna().any():
        raise ValueError(f"Missing states in table: {sorted(set(states) - set(df.state))}")
    return out


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def interval_series(ax, x, table, color, marker, label=None, filled=True, size=4.2):
    b = table["slope_per_5y"].to_numpy(float)
    lo = table["simultaneous_ci_low"].to_numpy(float)
    hi = table["simultaneous_ci_high"].to_numpy(float)
    ax.vlines(x, lo, hi, color=color, lw=1.0, zorder=2)
    ax.plot(x, b, linestyle="none", marker=marker, ms=size, mfc=color if filled else "white",
            mec=color, mew=0.9, zorder=3, label=label)


def state_axis(ax, extra_mean: bool = False):
    labels = [fs.STATE_LABELS[s] for s in fs.STATES] + (["Mean"] if extra_mean else [])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.axhline(0, color=fs.ZERO, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if extra_mean:
        ax.axvline(3.5, color="#D0D0D0", lw=0.6, zorder=0)
    fs.open_axes(ax)


# --------------------------------------------------------------------------
# Figure 2
# --------------------------------------------------------------------------
def figure2(a) -> plt.Figure:
    s3 = need(a.supplement_tables / "Table_S03_common_covariance_and_pair_amplitude_models.csv")
    native = by_state(s3[s3.scenario.eq("RAW_MATCHED")])
    amp = need(a.analysis11_tables / "20_common_amplitude_slopes.csv")
    paired = need(a.analysis11_tables / "22_signal_minus_noise_paired_slopes.csv")
    states5 = fs.STATES + ["EQUAL_STATE_MEAN"]

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.25, 0.85], left=0.075, right=0.985,
                            bottom=0.27, top=0.76, wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, i]) for i in range(3))

    # A: native common covariance
    interval_series(ax_a, np.arange(4), native, fs.COLORS["native"], "o")
    state_axis(ax_a)
    ax_a.set_ylabel("Common log slope per 5 years")

    # B: pair-amplitude components, same y scale as A
    series = [("SIGNAL_PAIR_AMPLITUDE", "signal_amp", "Signal"),
              ("TOTAL_PAIR_AMPLITUDE", "total", "Candidate total"),
              ("NOISE_PAIR_AMPLITUDE", "noise", "Noise")]
    for k, (metric, key, label) in enumerate(series):
        t = by_state(amp[amp.metric.eq(metric)], states5)
        interval_series(ax_b, np.arange(5) + (k - 1) * 0.22, t, fs.COLORS[key], fs.MARKERS[key], label=label)
    state_axis(ax_b, extra_mean=True)
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.3,
                columnspacing=1.0, borderaxespad=0.2)

    lo = min(ax_a.get_ylim()[0], ax_b.get_ylim()[0])
    hi = max(ax_a.get_ylim()[1], ax_b.get_ylim()[1], 0.05)
    for ax in (ax_a, ax_b):
        ax.set_ylim(lo, hi)

    # C: paired signal-minus-noise contrast
    t = by_state(paired, states5)
    interval_series(ax_c, np.arange(5), t, fs.COLORS["contrast"], "o")
    state_axis(ax_c, extra_mean=True)
    ax_c.set_ylabel("Log slope per 5 years")

    fig.canvas.draw()
    ytitle = ax_a.get_position().y1 + 0.13
    fs.panel_label(fig, ax_a, "A", "Native common covariance", y=ytitle)
    fs.panel_label(fig, ax_b, "B", "Pair amplitude", y=ytitle)
    fs.panel_label(fig, ax_c, "C", "Signal − noise pair amplitude", y=ytitle)
    # Move B's legend below the title line.
    return fig


# --------------------------------------------------------------------------
# Figure 3
# --------------------------------------------------------------------------
def figure3(a) -> plt.Figure:
    s5 = need(a.supplement_tables / "Table_S05_network_profile_and_convergence.csv")
    prof = s5[s5.table_component.eq("network_profile")].copy()
    networks = list(dict.fromkeys(prof.network))
    if FP not in networks or len(networks) != 10:
        raise ValueError("Expected ten networks including Frontoparietal in Table S05")
    slopes = prof.pivot(index="network", columns="state", values="relative_profile_slope_per_5y").loc[networks, fs.STATES]
    low = prof.pivot(index="network", columns="state", values="simultaneous_ci_low").loc[networks, fs.STATES]
    high = prof.pivot(index="network", columns="state", values="simultaneous_ci_high").loc[networks, fs.STATES]
    mean = prof.groupby("network").equal_state_mean_slope.first().loc[networks]
    excludes = (low > 0) | (high < 0)

    s6 = need(a.supplement_tables / "Table_S06_FP_network_comparisons.csv")
    comp = s6[s6.table_component.eq("equal_state_simultaneous_intervals")].set_index("comparator")

    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 3.05))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.045, 0.95], left=0.085, right=0.985,
                            bottom=0.22, top=0.86, wspace=0.17)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_cb = fig.add_subplot(grid[0, 1])
    ax_b = fig.add_subplot(grid[0, 2], sharey=ax_a)

    # A: heatmap with a separated equal-state mean column
    values = np.column_stack([slopes.to_numpy(float), mean.to_numpy(float)])
    vmax = float(np.ceil(np.nanmax(np.abs(values)) * 100) / 100)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    xpos = np.array([0, 1, 2, 3, 4.35])
    for j, x in enumerate(xpos):
        for i in range(len(networks)):
            v = values[i, j]
            ax_a.add_patch(Rectangle((x - 0.5, i - 0.5), 1, 1, facecolor=cmap(norm(v)), edgecolor="white", lw=1.0))
            ink = "white" if abs(v) > 0.6 * vmax else fs.TEXT
            ax_a.text(x, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.0, color=ink)
            if j < 4 and bool(excludes.iloc[i, j]):
                ax_a.add_patch(Rectangle((x - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=fs.TEXT, lw=1.0))
    ax_a.set_xlim(-0.5, 4.85)
    ax_a.set_ylim(len(networks) - 0.5, -0.5)
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels([fs.STATE_LABELS[s] for s in fs.STATES] + ["Mean"], rotation=30, ha="right",
                         rotation_mode="anchor")
    ax_a.set_yticks(range(len(networks)))
    ax_a.set_yticklabels([fs.NETWORK_ABBREV[n] for n in networks])
    fp_row = networks.index(FP)
    ax_a.get_yticklabels()[fp_row].set_color(fs.COLORS["fp"])
    ax_a.get_yticklabels()[fp_row].set_fontweight("bold")
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    ax_a.tick_params(length=0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.outline.set_linewidth(0.5)
    step = 0.04 if vmax >= 0.08 else 0.02
    cb.set_ticks(np.arange(-np.floor(vmax / step) * step, vmax + 1e-9, step))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%+.2f"))
    cb.ax.tick_params(labelsize=fs.SMALL, length=2)
    cb.set_label("Relative log slope per 5 years", fontsize=fs.SMALL)

    # B: FP minus each comparator, rows aligned with A
    for i, network in enumerate(networks):
        if network == FP:
            continue
        r = comp.loc[network]
        excl = bool(r.simultaneous_ci_excludes_zero) if not isinstance(r.simultaneous_ci_excludes_zero, str) \
            else r.simultaneous_ci_excludes_zero.strip().lower() == "true"
        ax_b.hlines(i, r.simultaneous_ci_low, r.simultaneous_ci_high, color=fs.COLORS["fp"], lw=1.0)
        ax_b.plot(r.equal_state_mean_difference, i, marker="o", ms=4.2, mec=fs.COLORS["fp"], mew=0.9,
                  mfc=fs.COLORS["fp"] if excl else "white", linestyle="none")
    ax_b.axvline(0, color=fs.ZERO, lw=0.7, ls=(0, (3, 2)))
    ax_b.set_xlabel("FP − comparator, log slope per 5 years")
    plt.setp(ax_b.get_yticklabels(), visible=False)
    fs.open_axes(ax_b, left=False)
    ax_b.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="none", ms=4.2, mec=fs.COLORS["fp"], mfc=fs.COLORS["fp"],
                      label="interval excludes 0"),
               Line2D([], [], marker="o", ls="none", ms=4.2, mec=fs.COLORS["fp"], mfc="white",
                      label="interval includes 0")]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
                handletextpad=0.3, columnspacing=1.2)

    fig.canvas.draw()
    fs.panel_label(fig, ax_a, "A", "Network-relative slopes", dx=-0.07)
    fs.panel_label(fig, ax_b, "B", "FP versus each comparator (equal-state)", dx=0.0,
                   y=ax_a.get_position().y1 + 0.035)
    return fig


# --------------------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------------------
def figure4(a) -> plt.Figure:
    lon = need(a.analysis10_tables / "20_longitudinal_slopes.csv")
    fig = plt.figure(figsize=(fs.DOUBLE_COLUMN_IN, 2.55))
    grid = fig.add_gridspec(1, 2, left=0.085, right=0.985, bottom=0.27, top=0.80, wspace=0.30)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    specs = [("COMMON_LEVEL", "Common log slope per 5 years"),
             ("FP_RELATIVE", "FP-relative log slope per 5 years")]
    components = [("SIGNAL", "signal_norm", "o", "Signal-normalized"),
                  ("TOTAL", "total", "D", "Candidate total-normalized")]
    for ax, (target, ylabel) in zip(axes, specs):
        for k, (component, key, marker, label) in enumerate(components):
            t = by_state(lon[lon.target.eq(target) & lon.component.eq(component)])
            interval_series(ax, np.arange(4) + (k - 0.5) * 0.24, t, fs.COLORS[key], marker, label=label)
        state_axis(ax)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3,
                   columnspacing=1.0, borderaxespad=0.2)
    fig.canvas.draw()
    fs.panel_label(fig, axes[0], "A", "Common level", dx=-0.06, y=axes[0].get_position().y1 + 0.10)
    fs.panel_label(fig, axes[1], "B", "FP relative to the other nine networks", dx=-0.06,
                   y=axes[0].get_position().y1 + 0.10)
    return fig


CAPTION_NOTES = {
    "Figure2": "A: simultaneous 95% intervals across four states (Table S03, outcome-specific fit). "
               "B: Bonferroni across 20 slopes (script 11, fixed common-model weights). "
               "C: Bonferroni across 5 paired contrasts.",
    "Figure3": "A: outlined cells have 40-cell Bonferroni intervals excluding zero; inference rests on the "
               "profile omnibus tests. B: max-|t| simultaneous intervals across nine comparators.",
    "Figure4": "Bonferroni across four states within each representation; FP selection not adjusted. "
               "Total normalization uses the candidate reconstruction t = sqrt(a^2 + n^2).",
}


def main() -> None:
    a = parse_args()
    fs.configure()
    for name, builder in [("Figure2", figure2), ("Figure3", figure3), ("Figure4", figure4)]:
        fig = builder(a)
        for path in fs.save(fig, a.output_dir, name):
            print(path.resolve())
        plt.close(fig)
        print(f"  caption note: {CAPTION_NOTES[name]}")


if __name__ == "__main__":
    main()
