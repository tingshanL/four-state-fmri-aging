#!/usr/bin/env python3
"""Build manuscript figures 1--4, supplementary figures, and source tables from analyses 01--07."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
STATE_LABELS = {
    "REST": "Rest",
    "CARIT": "CARIT",
    "FACENAME": "FaceName",
    "VISMOTOR": "Visual–motor",
}
FP_NETWORK = "Frontoparietal"

COLORS = {
    "native": "#3B5B92",
    "amplitude": "#C77C2E",
    "standardized": "#3A8A72",
    "fp": "#B24745",
    "neutral": "#6E7781",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manuscript figures and tables from analyses 01--07."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results root containing analysis outputs 01 through 07.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default: RESULTS_DIR/manuscript_outputs.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path, sources: list[dict]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required result table is missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    sources.append({"file": str(path), "rows": int(len(frame))})
    return frame


def read_json(path: Path, sources: list[dict]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    sources.append({"file": str(path), "rows": np.nan})
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_state_order"] = pd.Categorical(out["state"], STATES, ordered=True)
    return out.sort_values("_state_order").drop(columns="_state_order")


def one_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    selected = frame.loc[frame[column].astype(str).eq(value)]
    if len(selected) != 1:
        raise AssertionError(
            f"Expected one row where {column}={value!r}; found {len(selected)}."
        )
    return selected.iloc[0]


def to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def concat_parts(parts: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    return pd.concat(
        [frame.assign(table_component=label) for label, frame in parts],
        ignore_index=True,
        sort=False,
    )


def save_figure(
    fig: plt.Figure,
    directory: Path,
    stem: str,
    *,
    include_svg: bool = False,
    png_dpi: int = 300,
    vector_dpi: int | None = None,
    pad_inches: float | None = None,
) -> None:
    formats = [(".png", {"dpi": png_dpi}), (".pdf", {})]
    if include_svg:
        formats.append((".svg", {}))
    for suffix, options in formats:
        if vector_dpi is not None and suffix in {".pdf", ".svg"}:
            options["dpi"] = vector_dpi
        if pad_inches is not None:
            options["pad_inches"] = pad_inches
        target = directory / f"{stem}{suffix}"
        temporary = directory / f"{stem}.tmp{suffix}"
        fig.savefig(temporary, bbox_inches="tight", **options)
        if temporary.stat().st_size == 0:
            raise OSError(f"Empty figure file: {temporary}")
        temporary.replace(target)
        temporary.unlink(missing_ok=True)
    plt.close(fig)


# Figure 1: frozen cohort sampling (Panel A) and illustrative representation
# schematic (Panels B/C). Panels B/C do not show longitudinal observations;
# Panel C compares two representations of the same illustrative measurements.
# This is the plotting implementation from make_figure1_study_design.py.
# --------------------------------------------------------------------------
# Palette shared with Figures 2-4
# --------------------------------------------------------------------------
NATIVE = "#0072B2"
AMPLITUDE = "#E69F00"
AMPLITUDE_TEXT = "#B97A00"  # darker amplitude hue for small text on white
STANDARDIZED = "#009E73"
FP = "#7B3294"
GRAY = "#777777"
NOTE = "#4D4D4D"
TEXT = "#1A1A1A"

NETWORKS = ["VIS1", "VIS2", "SM", "CO", "DA", "LANG", "FP", "AUD", "DMN", "PMM"]
FP_INDEX = NETWORKS.index("FP")

SMALL = 6.8  # smallest text in the figure
BASE = 7.0


def configure_figure1_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "sans",
            "mathtext.it": "sans:italic",
            "mathtext.bf": "sans:bold",
            "mathtext.fallback": "cm",
            "font.size": BASE,
            "axes.labelsize": BASE,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_header(fig: plt.Figure, x: float, label: str, title: str, note: str | None = None) -> None:
    fig.text(x, 0.965, label, fontsize=9.5, fontweight="bold", ha="left", va="baseline")
    fig.text(x + 0.022, 0.965, title, fontsize=7.5, ha="left", va="baseline", color=TEXT)
    if note:
        fig.text(x + 0.022, 0.928, note, fontsize=SMALL, ha="left", va="baseline", color=GRAY)


def open_axes(ax: plt.Axes, left: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.tick_params(direction="out")


# --------------------------------------------------------------------------
# Panel A: longitudinal sampling (cohort data)
# --------------------------------------------------------------------------
def load_figure1_cohort(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reuse the frozen primary-cohort tables already read for Table 1."""
    persons = tables["person_summary"]
    cohort_ids = tables["cognition_flow"]["id"].astype(str)
    cohort = persons.loc[persons["id"].astype(str).isin(cohort_ids)].copy()
    if len(cohort) != 852 or int(cohort["sessions"].sum()) != 2220:
        raise AssertionError(
            f"Cohort check failed: {len(cohort)} participants, {int(cohort['sessions'].sum())} sessions"
        )
    cohort["last_age"] = cohort["baseline_age"] + cohort["followup_years"]
    return cohort.sort_values(["baseline_age", "followup_years"]).reset_index(drop=True)


def draw_sampling(ax: plt.Axes, cohort: pd.DataFrame) -> None:
    n = len(cohort)
    y = np.arange(n)
    x0 = cohort["baseline_age"].to_numpy()
    x1 = cohort["last_age"].to_numpy()

    segments = np.stack([np.c_[x0, y], np.c_[x1, y]], axis=1)
    ax.add_collection(LineCollection(segments, colors="#C8C8C8", linewidths=0.3, zorder=1))
    ax.scatter(x0, y, s=0.6, color=GRAY, linewidths=0, zorder=2, rasterized=True)
    ax.scatter(x1, y, s=0.6, color=GRAY, linewidths=0, zorder=2, rasterized=True)

    ax.set_xlim(34, 98)
    ax.set_ylim(-10, n + 10)
    ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_yticks([])
    ax.set_xlabel("Age at session (years)")
    ax.set_ylabel("Participants, ordered by baseline age", labelpad=2)
    open_axes(ax, left=False)

    # Key (empty upper-left): the first dot is baseline age, the segment is follow-up.
    ky = 0.84 * n
    kx0, kx1 = 45.0, 55.0
    ax.plot([kx0, kx1], [ky, ky], color=TEXT, lw=0.9, solid_capstyle="butt")
    ax.scatter([kx0, kx1], [ky, ky], s=7, color=TEXT, zorder=3, linewidths=0)
    bracket_y = ky + 28
    ax.plot([kx0, kx0, kx1, kx1], [bracket_y - 10, bracket_y, bracket_y, bracket_y - 10], color=TEXT, lw=0.5)
    ax.text((kx0 + kx1) / 2, bracket_y + 12, "elapsed follow-up", ha="center", va="bottom", fontsize=SMALL)
    ax.annotate(
        "baseline age",
        xy=(kx0, ky - 8),
        xytext=(kx0 - 1.5, ky - 70),
        ha="left",
        va="top",
        fontsize=SMALL,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT, shrinkA=1, shrinkB=1),
    )

    # Cohort summary (empty lower-right).
    n_sessions = int(cohort["sessions"].sum())
    median_fu = float(cohort["followup_years"].median())
    ax.text(
        97,
        0.02 * n,
        f"{n} participants\n{n_sessions:,} sessions\n"
        f"median follow-up {median_fu:.1f} y\n\n"
        "Four fMRI states: REST,\nCARIT, FaceName,\nvisual–motor",
        ha="right",
        va="bottom",
        fontsize=SMALL,
        linespacing=1.2,
        color=TEXT,
    )


# --------------------------------------------------------------------------
# Panel B: covariance and amplitude standardization (schematic)
# --------------------------------------------------------------------------
BLOCK_SIZES = [8, 7, 9, 6, 6]
HIGHLIGHT_BLOCK = 2


def synthetic_parcels(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = sum(BLOCK_SIZES)
    labels = np.repeat(np.arange(len(BLOCK_SIZES)), BLOCK_SIZES)
    loadings = np.zeros((n, len(BLOCK_SIZES) + 1))
    loadings[np.arange(n), labels] = rng.uniform(0.65, 0.9, n)
    loadings[:, -1] = 0.25
    shared = loadings @ loadings.T
    np.fill_diagonal(shared, 0)
    shared += np.eye(n) * shared.max() * 1.05
    d = np.sqrt(np.diag(shared))
    coupling = shared / np.outer(d, d)
    amplitude = rng.uniform(0.55, 1.45, n)
    amp_product = np.outer(amplitude, amplitude)
    return amp_product * coupling, amp_product


def draw_matrix(ax: plt.Axes, mat: np.ndarray, cmap) -> None:
    n = mat.shape[0]
    shown = mat.copy()
    np.fill_diagonal(shown, np.nan)  # diagonal excluded from the network means
    cmap = cmap.copy()
    cmap.set_bad("white")
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=np.nanpercentile(shown, 99), interpolation="nearest")
    edges = np.r_[0, np.cumsum(BLOCK_SIZES)] - 0.5
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        shade = TEXT if b == HIGHLIGHT_BLOCK else ("#D0D0D0" if b % 2 else "#A8A8A8")
        ax.add_patch(Rectangle((lo, -2.6), hi - lo, 1.4, color=shade, clip_on=False, lw=0))
        ax.add_patch(Rectangle((-2.6, lo), 1.4, hi - lo, color=shade, clip_on=False, lw=0))
    lo, hi = edges[HIGHLIGHT_BLOCK], edges[HIGHLIGHT_BLOCK + 1]
    ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec=TEXT, lw=1.0))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(GRAY)
    ax.set_anchor("W")


def draw_panel_b(ax_cov: plt.Axes, ax_amp: plt.Axes) -> None:
    """Two stacked matrices, each on the same row as the Panel C profile it feeds."""
    covariance, amp_product = synthetic_parcels()
    blue_map = LinearSegmentedColormap.from_list("native", ["#FFFFFF", NATIVE])
    orange_map = LinearSegmentedColormap.from_list("amplitude", ["#FFFFFF", AMPLITUDE])
    draw_matrix(ax_cov, covariance, blue_map)
    draw_matrix(ax_amp, amp_product, orange_map)

    # Name and within-network pair average to the right of each matrix.
    for ax, name, eq, color in [
        (ax_cov, "Covariance $C_{pq}$", r"$C_k=\overline{C_{pq}}$", NATIVE),
        (ax_amp, "Amplitude\nproduct $a_pa_q$", r"$P_k=\overline{a_pa_q}$", AMPLITUDE_TEXT),
    ]:
        ax.text(1.10, 0.66, name, transform=ax.transAxes, ha="left", va="center",
                fontsize=BASE, color=TEXT, linespacing=1.15)
        ax.text(1.10, 0.30, eq, transform=ax.transAxes, ha="left", va="center",
                fontsize=8, color=color)

    # Network k: label the black segment of the top strip.
    edges = np.r_[0, np.cumsum(BLOCK_SIZES)] - 0.5
    mid = (edges[HIGHLIGHT_BLOCK] + edges[HIGHLIGHT_BLOCK + 1]) / 2
    n = sum(BLOCK_SIZES)
    ax_cov.annotate(
        "network $k$",
        xy=(mid, -2.8),
        xytext=(mid + 0.35 * n, -6.5),
        ha="left",
        va="center",
        fontsize=SMALL,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT, shrinkA=1, shrinkB=0),
        annotation_clip=False,
    )

    # Ratio in the gap between rows, level with Panel C's "subtract log P_k".
    ax_cov.text(0.5, -0.27, r"$C^{\mathrm{STD}}_k = C_k \,/\, P_k$", transform=ax_cov.transAxes,
                ha="center", va="center", fontsize=8, color=STANDARDIZED)

    # Footnote under the lower matrix, level with Panel C's network labels.
    ax_amp.text(-0.02, -0.10, "Overbars: means over distinct\nparcel pairs within network $k$;\ndiagonal excluded.",
                transform=ax_amp.transAxes, ha="left", va="top", fontsize=SMALL, color=NOTE,
                linespacing=1.25)


# --------------------------------------------------------------------------
# Panel C: common level and relative profile (schematic)
# --------------------------------------------------------------------------
# Hand-chosen illustrative log values (relative to an arbitrary level). The
# standardized profile is deliberately less dispersed than the native one, and
# both rows are drawn on the same vertical span so that the difference in
# dispersion is shown faithfully rather than set by autoscaling. FP sits a
# little above the common level in both rows: the rows are two
# representations of one set of measurements, not two visits.
NATIVE_DEV = np.array([0.30, 0.18, 0.36, -0.06, -0.20, -0.34, 0.10, 0.24, -0.02, -0.25])
STD_DEV = np.array([-0.06, -0.22, 0.24, 0.04, -0.10, -0.12, 0.08, 0.28, -0.12, -0.10])
NATIVE_L = NATIVE_DEV + 2.2
STD_L = STD_DEV - 0.4
LOG_P = NATIVE_L - STD_L  # network-specific log pair amplitude implied by the toy values


def draw_profile(ax: plt.Axes, values: np.ndarray, color: str, half_span: float) -> float:
    x = np.arange(len(values))
    g = values.mean()
    ax.axhline(g, color=color, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.vlines(x, g, values, colors=[FP if i == FP_INDEX else GRAY for i in x], lw=1.1, zorder=2)
    ax.scatter(x, values, s=16, c=[FP if i == FP_INDEX else color for i in x],
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_ylim(g - half_span, g + half_span)  # same span in both rows; offsets differ
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_yticks([])
    open_axes(ax)
    return g


def label_stem(ax: plt.Axes, values: np.ndarray, g: float, i: int, text: str) -> None:
    y = g + 0.45 * (values[i] - g)
    ax.annotate(text, xy=(i + 0.06, y), xytext=(i + 0.34, y), fontsize=BASE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=TEXT, shrinkA=1, shrinkB=0))


def draw_panel_c(ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    deviations = np.r_[NATIVE_L - NATIVE_L.mean(), STD_L - STD_L.mean()]
    half_span = 1.25 * np.abs(deviations).max()

    g_raw = draw_profile(ax_top, NATIVE_L, NATIVE, half_span)
    g_std = draw_profile(ax_bot, STD_L, STANDARDIZED, half_span)

    ax_top.set_ylabel("Native\n" + r"$\log C_k$", color=NATIVE, labelpad=3, linespacing=1.3)
    ax_bot.set_ylabel("Standardized\n" + r"$\log C^{\mathrm{STD}}_k$", color=STANDARDIZED,
                      labelpad=3, linespacing=1.3)

    for ax, g, name, color in [(ax_top, g_raw, "RAW", NATIVE), (ax_bot, g_std, "STD", STANDARDIZED)]:
        ax.text(len(NETWORKS) - 0.45, g, rf"$G^{{\mathrm{{{name}}}}}$", ha="left", va="center",
                fontsize=BASE, color=color)

    # One example stem per row (LANG native, VIS2 standardized),
    # each chosen where the neighbouring stem leaves room for the label.
    label_stem(ax_top, NATIVE_L, g_raw, 5, r"$R^{\mathrm{RAW}}_k$")
    label_stem(ax_bot, STD_L, g_std, 1, r"$R^{\mathrm{STD}}_k$")

    for ax in (ax_top, ax_bot):
        ax.set_xticks(range(len(NETWORKS)))
    ax_top.set_xticklabels([])
    ax_bot.set_xticklabels(NETWORKS, rotation=45, ha="right", rotation_mode="anchor")
    ax_bot.get_xticklabels()[FP_INDEX].set_color(FP)
    ax_bot.get_xticklabels()[FP_INDEX].set_fontweight("bold")

    # The representation change, in the gap between rows.
    ax_top.annotate(
        "",
        xy=(0.30, -0.40),
        xycoords="axes fraction",
        xytext=(0.30, -0.12),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=AMPLITUDE, mutation_scale=7),
        annotation_clip=False,
    )
    ax_top.text(0.33, -0.27, r"subtract network-specific $\log P_k$", transform=ax_top.transAxes,
                fontsize=BASE, color=AMPLITUDE_TEXT, ha="left", va="center")


# --------------------------------------------------------------------------
def build_figure1(cohort: pd.DataFrame) -> plt.Figure:
    configure_figure1_style()
    fig = plt.figure(figsize=(7.4, 3.35))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[0.92, 0.92, 1.30], height_ratios=[1, 1],
        left=0.035, right=0.955, bottom=0.14, top=0.875, wspace=0.30, hspace=0.50,
    )

    ax_a = fig.add_subplot(grid[:, 0])
    draw_sampling(ax_a, cohort)

    ax_cov = fig.add_subplot(grid[0, 1])
    ax_amp = fig.add_subplot(grid[1, 1])
    draw_panel_b(ax_cov, ax_amp)

    ax_c1 = fig.add_subplot(grid[0, 2])
    ax_c2 = fig.add_subplot(grid[1, 2])
    draw_panel_c(ax_c1, ax_c2)

    panel_header(fig, ax_a.get_position().x0 - 0.03, "A", "Longitudinal sampling", "cohort data")
    panel_header(fig, ax_cov.get_position().x0 - 0.01, "B", "Amplitude standardization", "schematic")
    panel_header(fig, ax_c1.get_position().x0 - 0.075, "C", "Common level and relative profile",
                 "schematic; not longitudinal change")
    return fig

def errorbar_x(ax, frame, estimate, low, high, y, **kwargs) -> None:
    center = frame[estimate].to_numpy(float)
    lo = frame[low].to_numpy(float)
    hi = frame[high].to_numpy(float)
    ax.errorbar(
        center,
        y,
        xerr=np.vstack([center - lo, hi - center]),
        **kwargs,
    )


def reconstruct_profile_levels(
    checkpoint: dict,
    network_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = checkpoint.get("result", checkpoint)
    payload = result.get("payload", result)
    coefficients = np.asarray(payload["coef"], dtype=float)
    basis = np.asarray(payload["basis"], dtype=float)
    names = list(payload["names"])

    if basis.shape[0] != len(network_order):
        raise AssertionError("Profile checkpoint and network order disagree.")
    if coefficients.size != basis.shape[1] * len(names):
        raise AssertionError("Profile checkpoint dimensions are inconsistent.")

    def state_weight(state: str, coded_state: str) -> float:
        if state == coded_state:
            return 1.0
        if state == STATES[-1]:
            return -1.0
        return 0.0

    def design_vector(state: str, slope: bool) -> np.ndarray:
        vector = np.zeros(len(names), dtype=float)
        base = "t5" if slope else "Intercept"
        vector[names.index(base)] = 1.0
        for coded in STATES[:-1]:
            term = f"t5:state_sum[{coded}]" if slope else f"state_sum[{coded}]"
            if term in names:
                vector[names.index(term)] = state_weight(state, coded)
        return vector

    rows = []
    for state in STATES:
        level_vector = design_vector(state, slope=False)
        slope_vector = design_vector(state, slope=True)
        for index, network in enumerate(network_order):
            network_weight = np.zeros(len(network_order), dtype=float)
            network_weight[index] = 1.0
            coordinate_weight = network_weight @ basis
            level_contrast = np.kron(coordinate_weight, level_vector)
            slope_contrast = np.kron(coordinate_weight, slope_vector)
            baseline = float(level_contrast @ coefficients)
            slope_value = float(slope_contrast @ coefficients)
            rows.append(
                {
                    "state": state,
                    "network": network,
                    "fitted_baseline_relative_level": baseline,
                    "relative_profile_slope_per_5y": slope_value,
                    "fitted_relative_level_at_5y": baseline + slope_value,
                }
            )

    state_profiles = pd.DataFrame(rows)
    equal_state = (
        state_profiles.groupby("network", as_index=False)
        .agg(
            fitted_baseline_relative_level=("fitted_baseline_relative_level", "mean"),
            relative_profile_slope_per_5y=("relative_profile_slope_per_5y", "mean"),
            fitted_relative_level_at_5y=("fitted_relative_level_at_5y", "mean"),
        )
    )
    return state_profiles, equal_state


def build_table_1(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    profile_cohort = tables["profile_cohort"]
    primary = one_row(profile_cohort, "cohort", "FULL_PRIMARY")
    eligibility = tables["network_state_eligibility"]
    eligible_networks = set(tables["network_order"])
    eligibility = eligibility.loc[eligibility["network"].isin(eligible_networks)]

    people = tables["person_summary"].copy()
    cognition_flow = tables["cognition_flow"].copy()
    cognition_flow["id"] = cognition_flow["id"].astype(str)
    people["id"] = people["id"].astype(str)
    primary_people = people.loc[people["id"].isin(set(cognition_flow["id"]))].copy()
    if len(primary_people) != int(primary["participants"]):
        raise AssertionError(
            "The participant summaries do not recover the primary imaging cohort."
        )

    rows: list[dict] = []

    def add(section, item, value, denominator="", note=""):
        rows.append(
            {
                "section": section,
                "item": item,
                "value": value,
                "denominator": denominator,
                "note": note,
            }
        )

    n_primary = int(primary["participants"])
    add("Primary imaging cohort", "Participants", n_primary)
    add("Primary imaging cohort", "Sessions", int(primary["sessions"]))
    add("Primary imaging cohort", "Session-state rows", int(primary["session_state_rows"]))
    add("Primary imaging cohort", "Participants with ≥3 sessions", int(primary["participants_3plus"]), n_primary)

    age = pd.to_numeric(primary_people["baseline_age"], errors="coerce").dropna()
    add("Primary imaging cohort", "Baseline age, mean (SD), years", f"{age.mean():.2f} ({age.std(ddof=1):.2f})", len(age))
    add("Primary imaging cohort", "Baseline age, range, years", f"{age.min():.2f}–{age.max():.2f}", len(age))
    for label, count in primary_people["sex"].astype("string").fillna("Missing").value_counts(dropna=False).items():
        add("Primary imaging cohort", f"Sex: {label}", int(count), n_primary, f"{100 * count / n_primary:.1f}%")

    followup = pd.to_numeric(primary_people["followup_years"], errors="coerce").dropna()
    add("Longitudinal coverage", "Follow-up, median [IQR], years", f"{followup.median():.2f} [{followup.quantile(.25):.2f}, {followup.quantile(.75):.2f}]", len(followup))
    add("Longitudinal coverage", "Same-scanner participants", int((~primary_people["scanner_changed"].map(to_bool)).sum()), n_primary)
    add("Longitudinal coverage", "Scanner-changed participants", int(primary_people["scanner_changed"].map(to_bool).sum()), n_primary)

    state_rows = (
        eligibility.groupby("state", as_index=False)["opportunity_rows"]
        .agg(["min", "max"])
        .reset_index()
    )
    if not state_rows["min"].eq(state_rows["max"]).all():
        raise AssertionError("State opportunity counts differ across eligible networks.")
    for _, row in state_rows.iterrows():
        add("State coverage", STATE_LABELS.get(str(row["state"]), str(row["state"])), int(row["min"]))

    structural = tables["structural_cohort"]
    for _, row in structural.iterrows():
        add("Structural subsets", str(row["cohort"]).replace("_", " ").title(), int(row["participants"]), n_primary, f"{int(row['sessions'])} sessions")
    area_cohort = tables["area_cohort"]
    for _, row in area_cohort.iterrows():
        add("Structural subsets", str(row["cohort"]).replace("_", " ").title(), int(row["participants"]), n_primary, f"{int(row['sessions'])} sessions")

    flow_counts = cognition_flow["selection_category"].value_counts()
    add("Cognition subset", "At least two paired FluidIQ visits", int(flow_counts.get("INCLUDED", 0)), n_primary)
    add("Cognition subset", "One paired FluidIQ visit", int(flow_counts.get("ONE_PAIRED_COGNITION_VISIT", 0)), n_primary)
    add("Cognition subset", "No paired FluidIQ visit", int(flow_counts.get("NO_PAIRED_COGNITION", 0)), n_primary)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_root = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_root / "manuscript_outputs"
    )
    if not results_root.is_dir():
        raise NotADirectoryError(results_root)

    main_table_dir = output_dir / "tables"
    supp_table_dir = output_dir / "tables"
    main_figure_dir = output_dir / "figures"
    supp_figure_dir = output_dir / "figures"
    audit_dir = output_dir / "audit"
    for directory in [main_table_dir, supp_table_dir, main_figure_dir, supp_figure_dir, audit_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    configure_plotting()
    sources: list[dict] = []

    d01 = results_root / "01_qc_and_cohort" / "tables"
    d02 = results_root / "02_primary_profile_and_measurement"
    d03 = results_root / "03_fp_specificity_and_robustness"
    d04 = results_root / "04_structural_sensitivity"
    d05 = results_root / "05_cognition"
    d06 = results_root / "06_final_checks"
    d07 = results_root / "07_site_profile_shape_diagnostic"

    profile_manifest = read_json(d02 / "run_manifest.json", sources)
    specificity_manifest = read_json(d03 / "run_manifest.json", sources)
    cognition_manifest = read_json(d05 / "run_manifest.json", sources)
    final_manifest = read_json(d06 / "run_manifest.json", sources)
    network_order = list(specificity_manifest["eligible_networks"])
    if len(network_order) != 10 or network_order != list(profile_manifest["eligible_networks"]):
        raise AssertionError("Analyses 02 and 03 must use the same ten-network order.")

    tables = {
        "network_order": network_order,
        "person_summary": read_csv(d01 / "person_summary.csv", sources),
        "qc": read_csv(d01 / "qc_resource_audit.csv", sources),
        "profile_cohort": read_csv(d02 / "tables" / "cohort_summary.csv", sources),
        "network_state_eligibility": read_csv(d02 / "tables" / "network_state_eligibility.csv", sources),
        "common_tests": read_csv(d02 / "tables" / "common_level_tests.csv", sources),
        "profile_tests": read_csv(d02 / "tables" / "profile_omnibus_tests.csv", sources),
        "measurement_ordinary": read_csv(d02 / "tables" / "gate1_ordinary_common_models.csv", sources),
        "measurement_decomposition": read_csv(d02 / "tables" / "gate1_common_weight_decomposition.csv", sources),
        "measurement_fraction": read_csv(d02 / "tables" / "gate1_residual_fraction_intervals.csv", sources),
        "profile_10x4": read_csv(d03 / "tables" / "amplitude_standardized_profile_10x4.csv", sources),
        "fp_maxwald": read_csv(d03 / "tables" / "fp_vs_network_maxwald.csv", sources),
        "fp_maxt": read_csv(d03 / "tables" / "fp_vs_network_maxt_intervals.csv", sources),
        "reference": read_csv(d03 / "tables" / "fp_reference_driver_sensitivity.csv", sources),
        "fp_slopes": read_csv(d03 / "tables" / "fp_relative_state_slopes.csv", sources),
        "fp_tests": read_csv(d03 / "tables" / "fp_relative_tests.csv", sources),
        "fp_site": read_csv(d03 / "tables" / "fp_relative_site_state_slopes.csv", sources),
        "fp_loo": read_csv(d03 / "tables" / "fp_relative_leave_one_site_out_slopes.csv", sources),
        "network_size": read_csv(d03 / "tables" / "network_size_diagnostic.csv", sources),
        "network_size_correlations": read_csv(d03 / "tables" / "network_size_correlations.csv", sources),
        "structural_cohort": read_csv(d04 / "tables" / "structural_cohort_summary.csv", sources),
        "area_cohort": read_csv(d04 / "tables" / "surface_area_cohort_summary.csv", sources),
        "structural_slopes": read_csv(d04 / "tables" / "structural_slopes.csv", sources),
        "area_slope": read_csv(d04 / "tables" / "surface_area_relative_slope.csv", sources),
        "structural_fmri_slopes": read_csv(d04 / "tables" / "fmri_state_slopes.csv", sources),
        "structural_changes": read_csv(d04 / "tables" / "fmri_coefficient_change_joint.csv", sources),
        "structural_components": read_csv(d04 / "tables" / "structural_fmri_component_coefficients.csv", sources),
        "structural_component_tests": read_csv(d04 / "tables" / "structural_fmri_component_joint_tests.csv", sources),
        "cognition_tests": read_csv(d05 / "tables" / "joint_tests.csv", sources),
        "cognition_coefficients": read_csv(d05 / "tables" / "brain_cognition_couplings.csv", sources),
        "cognition_precision": read_csv(d05 / "tables" / "within_person_precision.csv", sources),
        "cognition_flow": read_csv(d05 / "tables" / "cognition_selection_flow.csv", sources),
        "selection_summary": read_csv(d05 / "tables" / "selection_flow_summary.csv", sources),
        "selection_balance": read_csv(d05 / "tables" / "selection_balance_746_vs_106.csv", sources),
        "visit_balance": read_csv(d05 / "tables" / "one_vs_two_plus_cognition_visit_balance.csv", sources),
        "fluidiq_variance": read_csv(d05 / "tables" / "fluidiq_variance_audit.csv", sources),
        "age_tests": read_csv(d06 / "tables" / "age_moderation_joint_tests.csv", sources),
        "age_coefficients": read_csv(d06 / "tables" / "age_moderation_coefficients.csv", sources),
        "convergence": read_csv(d06 / "tables" / "profile_convergence_summary.csv", sources),
        "partial_covariance": read_csv(d06 / "tables" / "partial_covariance_comparison.csv", sources),
    }
    profile_checkpoint = read_json(
        d02 / "checkpoints" / "profile__amplitude_standardized.json", sources
    )
    state_profiles, equal_state = reconstruct_profile_levels(profile_checkpoint, network_order)

    profile_compare = state_profiles.merge(
        tables["profile_10x4"][["state", "network", "relative_profile_slope_per_5y"]],
        on=["state", "network"],
        suffixes=("_reconstructed", "_reported"),
        validate="one_to_one",
    )
    slope_error = np.max(
        np.abs(
            profile_compare["relative_profile_slope_per_5y_reconstructed"]
            - profile_compare["relative_profile_slope_per_5y_reported"]
        )
    )
    if slope_error > 1e-10:
        raise AssertionError(f"Profile slope reconstruction error: {slope_error:.3e}")

    # Main Table 1
    table_1 = build_table_1(tables)
    table_1.to_csv(main_table_dir / "Table_1_cohort_and_longitudinal_coverage.csv", index=False)

    # Figure 1: cohort sampling and representation schematic. Keep its plotting
    # settings local so Figures 2--4 and the supplementary figures are unchanged.
    with plt.rc_context():
        fig = build_figure1(load_figure1_cohort(tables))
        save_figure(
            fig,
            main_figure_dir,
            "Figure1",
            include_svg=True,
            png_dpi=400,
            vector_dpi=600,
            pad_inches=0.03,
        )

    # Figure 2: common covariance and amplitude scale
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    scenario_labels = {
        "RAW_MATCHED": "Native",
        "PAIR_AMPLITUDE_COMPONENT": "Amplitude component",
        "AMPLITUDE_STANDARDIZED": "Amplitude-standardized",
    }
    scenario_colors = {
        "RAW_MATCHED": "#0072B2",
        "PAIR_AMPLITUDE_COMPONENT": "#E69F00",
        "AMPLITUDE_STANDARDIZED": "#009E73",
    }
    scenario_markers = {
        "RAW_MATCHED": "o",
        "PAIR_AMPLITUDE_COMPONENT": "s",
        "AMPLITUDE_STANDARDIZED": "^",
    }
    zero_gray = "#8A8A8A"
    figure_gray = "#6E6E6E"

    def fig2_clean_axes(ax, keep_left=True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not keep_left:
            ax.spines["left"].set_visible(False)
        ax.tick_params(direction="out")

    def fig2_panel_label(ax, label, x=-0.11):
        ax.text(
            x, 1.035, label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9.5, fontweight="bold",
            clip_on=False,
        )

    fig = plt.figure(figsize=(7.4, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # A. Ordinary native / amplitude / standardized fits.
    for scenario, offset in zip(scenario_labels, [-0.17, 0.0, 0.17]):
        part = ordered_states(
            tables["measurement_ordinary"].loc[
                tables["measurement_ordinary"]["scenario"].eq(scenario)
            ]
        )
        x = np.arange(len(STATES), dtype=float) + offset
        est = part["slope_per_5y"].to_numpy(float)
        lo = part["simultaneous_ci_low"].to_numpy(float)
        hi = part["simultaneous_ci_high"].to_numpy(float)

        ax_a.errorbar(
            x,
            est,
            yerr=np.vstack([est - lo, hi - est]),
            fmt=scenario_markers[scenario],
            color=scenario_colors[scenario],
            markerfacecolor=scenario_colors[scenario],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.0,
            elinewidth=1.15,
            capsize=2.5,
            capthick=1.0,
            label=scenario_labels[scenario],
            zorder=3,
        )

    ax_a.axhline(0, color=zero_gray, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax_a.set_xticks(np.arange(len(STATES)))
    ax_a.set_xticklabels(STATES)
    ax_a.set_ylabel("Common-level slope / 5 y")
    ax_a.yaxis.set_major_locator(MaxNLocator(5))
    ax_a.margins(x=0.08, y=0.12)
    fig2_clean_axes(ax_a)
    fig2_panel_label(ax_a, "A", x=-0.09)

    # B. Fixed-common-weight residual slope fractions.
    fraction = ordered_states(
        tables["measurement_fraction"].loc[
            tables["measurement_fraction"]["state"].isin(STATES)
        ]
    )
    y = np.arange(len(fraction))[::-1]
    q = fraction["q_point"].to_numpy(float)
    lo = fraction["simultaneous_95_ci_low"].to_numpy(float)
    hi = fraction["simultaneous_95_ci_high"].to_numpy(float)

    ax_b.errorbar(
        q,
        y,
        xerr=np.vstack([q - lo, hi - q]),
        fmt="o",
        color="#0072B2",
        markerfacecolor="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=5.0,
        elinewidth=1.15,
        capsize=2.5,
        capthick=1.0,
        zorder=3,
    )
    ax_b.axvline(0.5, color=figure_gray, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax_b.axvline(1.0, color=figure_gray, lw=0.9, ls=(0, (1.5, 2)), zorder=1)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(fraction["state"])
    ax_b.set_ylim(-0.45, len(fraction) - 0.55)

    xmax = max(1.08, float(np.nanmax(hi)) * 1.04)
    ax_b.set_xlim(0, xmax)
    ax_b.set_xlabel(
        r"Residual slope fraction, $q_s = |\beta_{\mathrm{std}}/\beta_{\mathrm{raw}}|$"
    )
    ax_b.xaxis.set_major_locator(MaxNLocator(6))
    fig2_clean_axes(ax_b)
    fig2_panel_label(ax_b, "B", x=-0.13)

    # Shared legend above panel A.
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.9),
        ncol=3,
        frameon=False,
        fontsize=7,
        handletextpad=0.45,
        columnspacing=1.25,
        borderaxespad=0,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.20,
        top=0.79,
        wspace=0.31,
    )
    save_figure(fig, main_figure_dir, "Figure2")

    # Figure 3: network profile, FP comparisons, and convergence
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    network_abbrev = {
        "Visual1": "VIS1",
        "Visual2": "VIS2",
        "Somatomotor": "SM",
        "Cingulo-Opercular": "CO",
        "Dorsal-attention": "DA",
        "Language": "LANG",
        "Frontoparietal": "FP",
        "Auditory": "AUD",
        "Default": "DMN",
        "Posterior-Multimodal": "PMM",
        "Ventral-Multimodal": "VMM",
    }
    fig3_blue = "#0072B2"
    fig3_purple = "#7B3294"
    fig3_gray = "#777777"
    zero_gray = "#8A8A8A"

    def net_short(name):
        return network_abbrev.get(str(name), str(name))

    def fig3_panel_label(ax, label, x=-0.12):
        ax.text(
            x, 1.035, label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9.5, fontweight="bold",
            clip_on=False,
        )

    def fig3_clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    heat = (
        tables["profile_10x4"]
        .pivot(index="network", columns="state", values="relative_profile_slope_per_5y")
        .reindex(index=network_order, columns=STATES)
    )
    if heat.isna().any().any():
        raise AssertionError("The 10 × 4 profile table is incomplete.")

    # Full-width, two-row publication layout.  The former one-row A/B/C
    # arrangement made all three information-dense panels unnecessarily narrow
    # at the journal's final double-column width.  A and B share the top row;
    # the descriptive convergence panel is centered below at ~3/4 page width.
    fig = plt.figure(figsize=(7.4, 6.05))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.10, 0.88],
        hspace=0.48,
    )
    top = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0],
        width_ratios=[1.12, 0.13, 1.00],
        wspace=0.30,
    )
    bottom = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[1],
        width_ratios=[0.13, 0.74, 0.13],
        wspace=0.0,
    )
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 2])
    ax_c = fig.add_subplot(bottom[0, 1])

    # A. Complete frozen ten-network profile.
    matrix = heat.to_numpy(float)
    limit = float(np.max(np.abs(matrix)))
    image = ax_a.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax_a.set_xticks(np.arange(len(STATES)))
    ax_a.set_xticklabels(STATES, rotation=45)
    ax_a.set_yticks(np.arange(len(network_order)))
    ax_a.set_yticklabels([net_short(n) for n in network_order])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if abs(value) >= 0.58 * limit else "#202020"
            ax_a.text(
                j, i, f"{value:.02f}",
                ha="center", va="center",
                fontsize=6.3, color=text_color,
            )

    for spine in ax_a.spines.values():
        spine.set_linewidth(0.7)
    ax_a.tick_params(length=2.5)
    fig3_panel_label(ax_a, "A", x=-0.12)

    cbar = fig.colorbar(image, ax=ax_a, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=6.5, width=0.7, length=2)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Slope / 5 y", fontsize=7)

    # B. FP versus each comparator.
    forest = (
        tables["fp_maxt"]
        .sort_values("equal_state_mean_difference")
        .reset_index(drop=True)
    )
    y = np.arange(len(forest))
    for i, row in forest.iterrows():
        est = float(row["equal_state_mean_difference"])
        lo = float(row["simultaneous_ci_low"])
        hi = float(row["simultaneous_ci_high"])
        excludes_zero = (lo > 0) or (hi < 0)

        ax_b.errorbar(
            est,
            i,
            xerr=[[est - lo], [hi - est]],
            fmt="o",
            color=fig3_blue,
            markerfacecolor=fig3_blue if excludes_zero else "white",
            markeredgecolor=fig3_blue,
            markeredgewidth=1.0,
            markersize=4.4,
            elinewidth=1.05,
            capsize=2.2,
            capthick=0.9,
            zorder=3,
        )

    ax_b.axvline(0, color=zero_gray, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([net_short(x) for x in forest["comparator"]])
    ax_b.set_xlabel("FP − comparator slope / 5 y")
    ax_b.xaxis.set_major_locator(MaxNLocator(5))
    ax_b.margins(x=0.08, y=0.07)
    fig3_clean_axes(ax_b)
    fig3_panel_label(ax_b, "B", x=-0.14)

    # C. Descriptive baseline-profile / slope relation.
    diag = equal_state.copy()
    x_diag = diag["fitted_baseline_relative_level"].to_numpy(float)
    y_diag = diag["relative_profile_slope_per_5y"].to_numpy(float)

    # Fitted linear trend.
    line_x = np.linspace(x_diag.min(), x_diag.max(), 100)
    line_y = np.polyval(np.polyfit(x_diag, y_diag, 1), line_x)
    ax_c.plot(line_x, line_y, color=fig3_gray, lw=1.0, zorder=2)

    non_fp = diag.loc[~diag["network"].eq(FP_NETWORK)]
    ax_c.scatter(
        non_fp["fitted_baseline_relative_level"],
        non_fp["relative_profile_slope_per_5y"],
        s=25,
        color=fig3_gray,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )

    fp_diag = diag.loc[diag["network"].eq(FP_NETWORK)]
    if len(fp_diag) != 1:
        raise AssertionError(f"Expected one equal-state FP row; found {len(fp_diag)}")
    ax_c.scatter(
        fp_diag["fitted_baseline_relative_level"],
        fp_diag["relative_profile_slope_per_5y"],
        s=47,
        marker="s",
        color=fig3_purple,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )

    label_offsets = {
        "FP": (4, 3, "left"),
        "LANG": (4, 4, "left"),
        "SM": (4, 4, "left"),
        "CO": (4, 3, "left"),
        "DMN": (4, 3, "left"),
        "DA": (4, -8, "left"),
        "VIS1": (4, 3, "left"),
        "VIS2": (4, -8, "left"),
        "PMM": (4, 4, "left"),
        "VMM": (4, 4, "left"),
        "AUD": (-4, 4, "right"),
    }
    for _, row in diag.iterrows():
        short = net_short(row["network"])
        dx, dy, ha = label_offsets.get(short, (4, 4, "left"))
        ax_c.annotate(
            short,
            (
                float(row["fitted_baseline_relative_level"]),
                float(row["relative_profile_slope_per_5y"]),
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.3,
            color="#303030",
        )

    ax_c.axhline(0, color=zero_gray, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_c.axvline(0, color=zero_gray, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_c.set_xlabel("Baseline relative profile\n(equal-state mean)")
    ax_c.set_ylabel("Longitudinal slope / 5 y")

    convergence = tables["convergence"]
    corr = one_row(
        convergence.loc[convergence["state"].eq("EQUAL_STATE_AVERAGE")],
        "metric",
        "baseline_level_slope_r",
    )
    ratio = one_row(
        convergence.loc[convergence["state"].eq("EQUAL_STATE_AVERAGE")],
        "metric",
        "sd_ratio_5y_over_baseline",
    )
    ax_c.text(
        0.97,
        0.96,
        f"$r$ = {float(corr['observed']):.3f} "
        f"[{float(corr['bootstrap_q025']):.3f}, {float(corr['bootstrap_q975']):.3f}]\n"
        f"SD ratio = {float(ratio['observed']):.3f} "
        f"[{float(ratio['bootstrap_q025']):.3f}, {float(ratio['bootstrap_q975']):.3f}]",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        zorder=6,
        bbox=dict(
            boxstyle="square,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.92,
        ),
    )

    ax_c.margins(x=0.10, y=0.13)
    ax_c.xaxis.set_major_locator(MaxNLocator(5))
    ax_c.yaxis.set_major_locator(MaxNLocator(5))
    fig3_clean_axes(ax_c)
    fig3_panel_label(ax_c, "C", x=-0.14)

    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.095,
        top=0.965,
    )
    save_figure(fig, main_figure_dir, "Figure3")

    # Figure 4: structural and cognitive context
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig4_blue = "#0072B2"
    fig4_orange = "#E69F00"
    fig4_green = "#009E73"
    zero_gray = "#8A8A8A"

    def fig4_clean_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out")

    def fig4_panel_label(ax, label, x=-0.10):
        ax.text(
            x, 1.035, label,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9.5, fontweight="bold",
            clip_on=False,
        )

    structural_slopes = tables["structural_slopes"]
    thickness = one_row(structural_slopes, "scenario", "THICKNESS_PRIMARY")
    volume = one_row(structural_slopes, "scenario", "VOLUME_ICV_SECONDARY")
    area = tables["area_slope"].iloc[0]

    fig = plt.figure(figsize=(7.4, 4.75))
    outer = fig.add_gridspec(
        2, 2,
        height_ratios=[0.90, 1.05],
        width_ratios=[1.04, 1.26],
    )

    # A. Native-unit structural trajectories.
    sub = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0, 0],
        wspace=0.48,
    )
    structural_items = [
        (thickness, "Thickness", "mm / 5 y"),
        (area, "Surface area", "log units / 5 y"),
        (volume, "ICV-adjusted\nvolume", "log units / 5 y"),
    ]
    struct_axes = []

    for index, (row, title, xlabel) in enumerate(structural_items):
        ax = fig.add_subplot(sub[0, index])
        struct_axes.append(ax)

        estimate = float(row["slope_per_5y"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        ax.errorbar(
            estimate,
            0,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="o",
            color=fig4_blue,
            markerfacecolor=fig4_blue,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.0,
            elinewidth=1.15,
            capsize=2.5,
            capthick=1.0,
            zorder=3,
        )
        ax.axvline(0, color=zero_gray, lw=0.8, ls=(0, (3, 2)), zorder=1)

        xmin = min(0.0, lo)
        xmax = max(0.0, hi)
        span = xmax - xmin
        if span <= 0:
            span = max(abs(estimate), 1.0)
        pad = 0.13 * span
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=6.8)
        ax.set_title(title, fontsize=7.2, fontweight="normal", pad=4)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)

    fig4_panel_label(struct_axes[0], "A", x=-0.17)

    # B. Frozen FP-relative fMRI slopes before/after structural adjustment.
    ax_b = fig.add_subplot(outer[0, 1])
    scenario_order = [
        "THICKNESS_MATCHED_ORIGINAL",
        "THICKNESS_WITHIN_BETWEEN",
        "VOLUME_WITHIN_BETWEEN",
    ]
    scenario_names = {
        "THICKNESS_MATCHED_ORIGINAL": "Original",
        "THICKNESS_WITHIN_BETWEEN": "+ thickness",
        "VOLUME_WITHIN_BETWEEN": "+ volume",
    }
    fig4_scenario_colors = {
        "THICKNESS_MATCHED_ORIGINAL": fig4_blue,
        "THICKNESS_WITHIN_BETWEEN": fig4_orange,
        "VOLUME_WITHIN_BETWEEN": fig4_green,
    }
    fig4_scenario_markers = {
        "THICKNESS_MATCHED_ORIGINAL": "o",
        "THICKNESS_WITHIN_BETWEEN": "s",
        "VOLUME_WITHIN_BETWEEN": "^",
    }

    for scenario, offset in zip(scenario_order, [-0.16, 0.0, 0.16]):
        part = ordered_states(
            tables["structural_fmri_slopes"].loc[
                tables["structural_fmri_slopes"]["scenario"].eq(scenario)
            ]
        )
        if len(part) != 4:
            raise AssertionError(f"Expected four state slopes for {scenario}.")

        x = np.arange(len(STATES), dtype=float) + offset
        est = part["slope_per_5y"].to_numpy(float)
        lo = part["simultaneous_ci_low"].to_numpy(float)
        hi = part["simultaneous_ci_high"].to_numpy(float)
        ax_b.errorbar(
            x,
            est,
            yerr=np.vstack([est - lo, hi - est]),
            fmt=fig4_scenario_markers[scenario],
            color=fig4_scenario_colors[scenario],
            markerfacecolor=fig4_scenario_colors[scenario],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.6,
            elinewidth=1.05,
            capsize=2.2,
            capthick=0.9,
            label=scenario_names[scenario],
            zorder=3,
        )

    ax_b.axhline(0, color=zero_gray, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_b.set_xticks(np.arange(len(STATES)))
    ax_b.set_xticklabels(STATES)
    ax_b.set_ylabel("FP-relative fMRI slope / 5 y")
    ax_b.yaxis.set_major_locator(MaxNLocator(5))
    ax_b.margins(x=0.08, y=0.13)
    fig4_clean_axes(ax_b)
    fig4_panel_label(ax_b, "B", x=-0.12)
    ax_b.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.09),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )

    # C. Frozen within-person FP-relative–FluidIQ coupling, full-width.
    ax_c = fig.add_subplot(outer[1, :])
    cognition = ordered_states(tables["cognition_precision"])
    y = np.arange(len(cognition))[::-1]
    est = cognition[
        "estimate_standardized_using_observed_within_person_sds"
    ].to_numpy(float)
    lo = cognition[
        "ci_low_standardized_using_observed_within_person_sds"
    ].to_numpy(float)
    hi = cognition[
        "ci_high_standardized_using_observed_within_person_sds"
    ].to_numpy(float)

    ax_c.errorbar(
        est,
        y,
        xerr=np.vstack([est - lo, hi - est]),
        fmt="o",
        color=fig4_blue,
        markerfacecolor=fig4_blue,
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=5.0,
        elinewidth=1.15,
        capsize=2.5,
        capthick=1.0,
        zorder=3,
    )
    ax_c.axvline(0, color=zero_gray, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(cognition["state"])
    ax_c.set_ylim(-0.45, len(cognition) - 0.55)

    xmax = 1.08 * max(abs(float(np.nanmin(lo))), abs(float(np.nanmax(hi))))
    ax_c.set_xlim(-xmax, xmax)
    ax_c.set_xlabel(
        "Standardized within-person coefficient (observed within-person SD units)"
    )
    ax_c.xaxis.set_major_locator(MaxNLocator(7))

    within_test = one_row(
        tables["cognition_tests"],
        "test",
        "FOUR_WITHIN_PERSON_COUPLINGS_ZERO",
    )
    ax_c.text(
        1.02,
        0.92,
        rf"$P_{{joint}}$ = {float(within_test['p']):.3f}",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    fig4_clean_axes(ax_c)
    fig4_panel_label(ax_c, "C", x=-0.055)

    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.115,
        top=0.94,
        wspace=0.42,
        hspace=0.55,
    )
    save_figure(fig, main_figure_dir, "Figure4")

    # Reset plotting defaults for supplementary outputs.
    configure_plotting()

    # Supplementary tables
    public_selection = tables["selection_balance"].loc[
        ~tables["selection_balance"]["variable"].astype(str).eq("study")
    ].copy()
    supp_tables = {
        "Table_S03_common_covariance_and_pair_amplitude_models.csv": concat_parts([("ordinary_models", tables["measurement_ordinary"]), ("common_weight_decomposition", tables["measurement_decomposition"]), ("common_level_tests", tables["common_tests"])]),
        "Table_S05_network_profile_and_convergence.csv": concat_parts([("network_profile", tables["profile_10x4"]), ("profile_tests", tables["profile_tests"]), ("convergence", tables["convergence"]), ("network_size", tables["network_size"]), ("network_size_correlations", tables["network_size_correlations"])]),
        "Table_S06_FP_network_comparisons.csv": concat_parts([("four_state_familywise_tests", tables["fp_maxwald"]), ("equal_state_simultaneous_intervals", tables["fp_maxt"]), ("fp_state_slopes", tables["fp_slopes"]), ("fp_tests", tables["fp_tests"])]),
        "Table_S07_age_moderation.csv": concat_parts([("joint_tests", tables["age_tests"]), ("coefficients", tables["age_coefficients"])]),
        "Table_S08_scanner_site_reference_QC_and_partial_covariance.csv": concat_parts([("reference_sensitivity", tables["reference"]), ("site_state_slopes", tables["fp_site"]), ("leave_one_site_out", tables["fp_loo"]), ("partial_covariance", tables["partial_covariance"]), ("qc_resources", tables["qc"])]),
        "Table_S09_structural_trajectories_and_fMRI_changes.csv": concat_parts([("structural_trajectories", tables["structural_slopes"]), ("surface_area", tables["area_slope"]), ("fMRI_coefficient_changes", tables["structural_changes"])]),
        "Table_S10_structural_fMRI_coefficients.csv": concat_parts([("coefficients", tables["structural_components"]), ("joint_tests", tables["structural_component_tests"])]),
        "Table_S11_cognition_precision_context_and_selection.csv": concat_parts([("within_person_precision", tables["cognition_precision"]), ("within_between_coefficients", tables["cognition_coefficients"]), ("joint_tests", tables["cognition_tests"]), ("selection_summary", tables["selection_summary"]), ("selection_balance", public_selection), ("one_vs_two_plus_visits", tables["visit_balance"]), ("fluidiq_variance", tables["fluidiq_variance"])]),
    }
    for filename, frame in supp_tables.items():
        frame.to_csv(supp_table_dir / filename, index=False)

    for filename in [
        "01_site_composition.csv",
        "01b_site_state_coverage.csv",
        "04_site_equal_state_similarity.csv",
        "05_site_equal_state_magnitude.csv",
        "06_site_FP_relative_slopes.csv",
    ]:
        source = d07 / "tables" / filename
        shutil.copy2(source, supp_table_dir / filename)
        sources.append({"file": str(source), "rows": len(pd.read_csv(source))})

    # Supplementary Figure S1
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    y = np.arange(4)[::-1]
    errorbar_x(axes[0], fraction, "q_point", "simultaneous_95_ci_low", "simultaneous_95_ci_high", y, fmt="o", capsize=3, color=COLORS["neutral"])
    axes[0].axvline(0.5, color="black", lw=0.8, ls="--")
    axes[0].axvline(1.0, color="black", lw=0.8, ls=":")
    axes[0].set_yticks(y, [STATE_LABELS[s] for s in fraction["state"]])
    axes[0].set_xlabel("Residual slope fraction")
    axes[0].set_title("A  Simultaneous intervals", loc="left", weight="bold")
    axes[1].bar([STATE_LABELS[s] for s in fraction["state"]], fraction["raw_denominator_sign_crossing_fraction"], color=COLORS["neutral"])
    axes[1].set_ylabel("Bootstrap sign-crossing fraction")
    axes[1].set_title("B  Denominator stability", loc="left", weight="bold")
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S1_residual_fraction_diagnostics")

    # Supplementary Figure S2
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for index, state in enumerate(STATES):
        part = tables["profile_10x4"].loc[tables["profile_10x4"]["state"].eq(state)].set_index("network").reindex(network_order).reset_index()
        errorbar_x(ax, part, "relative_profile_slope_per_5y", "simultaneous_ci_low", "simultaneous_ci_high", np.arange(10) + (index - 1.5) * 0.08, fmt="o", ms=3, capsize=2, label=STATE_LABELS[state])
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(np.arange(10), network_order)
    ax.set_xlabel("Amplitude-standardized relative-profile slope / 5 years")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S2_complete_network_profile")

    # Supplementary Figure S3
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    display_states = [*STATES, "EQUAL_STATE_AVERAGE"]
    display_labels = [*[STATE_LABELS[s] for s in STATES], "Equal-state average"]
    for ax, metric, null, xlabel in [(axes[0], "baseline_level_slope_r", 0.0, "Baseline level–slope correlation"), (axes[1], "sd_ratio_5y_over_baseline", 1.0, "Across-network SD ratio (5 y / baseline)")]:
        part = convergence.loc[convergence["metric"].eq(metric)].set_index("state").reindex(display_states).reset_index()
        y = np.arange(5)[::-1]
        errorbar_x(ax, part, "observed", "bootstrap_q025", "bootstrap_q975", y, fmt="o", capsize=3, color=COLORS["neutral"])
        ax.axvline(null, color="black", lw=0.8, ls="--")
        ax.set_yticks(y, display_labels)
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S3_profile_convergence_uncertainty")

    # Supplementary Figure S4
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    maxwald = tables["fp_maxwald"].sort_values("p_fwer_maxwald", ascending=False)
    y = np.arange(len(maxwald))
    axes[0].scatter(maxwald["p_fwer_maxwald"], y, color=COLORS["fp"])
    axes[0].axvline(0.05, color="black", lw=0.8, ls="--")
    axes[0].set_yticks(y, maxwald["comparator"])
    axes[0].set_xlabel("Familywise adjusted p")
    axes[0].set_title("A  Four-state trajectory tests", loc="left", weight="bold")
    y = np.arange(len(forest))
    errorbar_x(axes[1], forest, "equal_state_mean_difference", "simultaneous_ci_low", "simultaneous_ci_high", y, fmt="o", capsize=3, color=COLORS["fp"])
    axes[1].axvline(0, color="black", lw=0.8, ls="--")
    axes[1].set_yticks(y, forest["comparator"])
    axes[1].set_xlabel("Equal-state FP − comparator slope")
    axes[1].set_title("B  Equal-state simultaneous intervals", loc="left", weight="bold")
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S4_FP_network_comparisons")

    # Supplementary Figure S6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    reference = tables["reference"].loc[~tables["reference"]["reference_choice"].eq("ALL_NINE")].sort_values("mean_four_state_slope")
    y = np.arange(len(reference))
    axes[0].scatter(reference["mean_four_state_slope"], y, color=COLORS["neutral"])
    all_nine = float(tables["reference"].loc[tables["reference"]["reference_choice"].eq("ALL_NINE"), "mean_four_state_slope"].iloc[0])
    axes[0].axvline(all_nine, color="black", lw=0.8, ls="--")
    axes[0].set_yticks(y, reference["omitted_network"])
    axes[0].set_xlabel("Mean four-state FP-relative slope")
    axes[0].set_title("A  Reference-network sensitivity", loc="left", weight="bold")
    for scenario, marker, color in [("AMPLITUDE_STANDARDIZED", "o", COLORS["standardized"]), ("PARTIAL_MATCHED", "s", COLORS["native"])]:
        part = ordered_states(tables["fp_slopes"].loc[tables["fp_slopes"]["scenario"].eq(scenario)])
        axes[1].plot([STATE_LABELS[s] for s in part["state"]], part["slope_per_5y"], marker=marker, color=color, label=scenario.replace("_", " ").title())
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_ylabel("FP-relative slope / 5 years")
    axes[1].set_title("B  Full and partial covariance", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S6_reference_and_covariance_sensitivities")

    # Supplementary Figure S7
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    components = tables["structural_components"].copy()
    label_columns = [column for column in ["scenario", "component_type", "state"] if column in components]
    components["label"] = components[label_columns].astype(str).agg(" | ".join, axis=1)
    y = np.arange(len(components))
    errorbar_x(ax, components, "coefficient_per_1sd", "simultaneous_ci_low", "simultaneous_ci_high", y, fmt="o", capsize=2.5, color=COLORS["native"])
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y, components["label"], fontsize=7)
    ax.set_xlabel("fMRI coefficient per 1 SD structural measure")
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S7_structural_fMRI_coefficients")

    # Cognition and selection
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for component, marker, color in [("WITHIN_PERSON", "o", COLORS["amplitude"]), ("BETWEEN_PERSON", "s", COLORS["native"])]:
        part = ordered_states(tables["cognition_coefficients"].loc[tables["cognition_coefficients"]["component"].eq(component)])
        center = part["coefficient_per_0_1_log_unit"].to_numpy(float)
        low = 0.1 * part["simultaneous_ci_low"].to_numpy(float)
        high = 0.1 * part["simultaneous_ci_high"].to_numpy(float)
        axes[0].errorbar(np.arange(4), center, yerr=np.vstack([center - low, high - center]), fmt=marker, capsize=3, color=color, label=component.replace("_", " ").title())
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xticks(np.arange(4), [STATE_LABELS[s] for s in STATES], rotation=20)
    axes[0].set_ylabel("FluidIQ coefficient per 0.1 log unit")
    axes[0].set_title("A  Within- and between-person coefficients", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    selection_variables = [
        "n_eligible_fmri_visits",
        "fmri_followup_years",
        "baseline_age",
        "first_equal_state_mean_fp_relative",
    ]
    selection = public_selection.loc[
        public_selection["variable"].isin(selection_variables)
        & public_selection["standardized_difference"].notna()
    ].copy()
    selection_labels = {
        "n_eligible_fmri_visits": "Eligible fMRI visits",
        "fmri_followup_years": "Imaging follow-up",
        "baseline_age": "Baseline age",
        "first_equal_state_mean_fp_relative": "First equal-state FP-relative value",
    }
    selection["label"] = selection["variable"].map(selection_labels).fillna(selection["variable"])
    selection = selection.sort_values("standardized_difference")
    y = np.arange(len(selection))
    axes[1].scatter(selection["standardized_difference"], y, color=COLORS["neutral"])
    axes[1].axvline(0, color="black", lw=0.8, ls="--")
    axes[1].set_yticks(y, selection["label"], fontsize=7)
    axes[1].set_xlabel("Included − excluded standardized difference")
    axes[1].set_title("B  Repeated-cognition selection", loc="left", weight="bold")
    fig.tight_layout()
    save_figure(fig, supp_figure_dir, "Figure_S8_cognition_and_selection")

    copied_figures = {
        "Figure_S5_leave_one_site_out_FP_relative": d03 / "figures" / "fp_relative_leave_one_site_out_forest",
        "Figure_S9_site_profile_heterogeneity": d07 / "figures" / "Figure_S_site_profile_heterogeneity",
    }
    for target_stem, source_stem in copied_figures.items():
        for suffix in [".pdf", ".png"]:
            source = source_stem.with_suffix(suffix)
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, main_figure_dir / f"{target_stem}{suffix}")
            sources.append({"file": str(source), "rows": np.nan})

    source_inventory = pd.DataFrame(sources).drop_duplicates("file").sort_values("file")
    source_inventory.to_csv(audit_dir / "source_file_inventory.csv", index=False)

    main_stems = ["Figure1", "Figure2", "Figure3", "Figure4"]
    supp_stems = [
        "Figure_S1_residual_fraction_diagnostics",
        "Figure_S2_complete_network_profile",
        "Figure_S3_profile_convergence_uncertainty",
        "Figure_S4_FP_network_comparisons",
        "Figure_S5_leave_one_site_out_FP_relative",
        "Figure_S6_reference_and_covariance_sensitivities",
        "Figure_S7_structural_fMRI_coefficients",
        "Figure_S8_cognition_and_selection",
        "Figure_S9_site_profile_heterogeneity",
    ]
    main_png = [main_figure_dir / f"{stem}.png" for stem in main_stems]
    main_pdf = [main_figure_dir / f"{stem}.pdf" for stem in main_stems]
    supp_png = [supp_figure_dir / f"{stem}.png" for stem in supp_stems]
    supp_pdf = [supp_figure_dir / f"{stem}.pdf" for stem in supp_stems]
    checks = {
        "manuscript_tables": len(list(main_table_dir.glob("*.csv"))),
        "main_figure_families": len(main_png),
        "supplementary_figure_families": len(supp_png),
        "matching_main_pdfs": all(path.is_file() and path.stat().st_size > 0 for path in [*main_png, *main_pdf]),
        "figure1_svg": (main_figure_dir / "Figure1.svg").is_file() and (main_figure_dir / "Figure1.svg").stat().st_size > 0,
        "matching_supplementary_pdfs": all(path.is_file() and path.stat().st_size > 0 for path in [*supp_png, *supp_pdf]),
        "model_fitting_performed": False,
        "new_inferential_tests_performed": False,
    }
    expected = {
        "manuscript_tables": 14,
        "main_figure_families": 4,
        "supplementary_figure_families": 9,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise AssertionError(f"Expected {value} {key}; found {checks[key]}.")
    if not checks["matching_main_pdfs"] or not checks["matching_supplementary_pdfs"]:
        raise AssertionError("A PNG figure is missing its matching PDF.")
    if not checks["figure1_svg"]:
        raise AssertionError("Figure 1 SVG is missing or empty.")

    manifest = {
        "script": "08_make_manuscript_outputs.py",
        "results_directory": str(results_root),
        "output_directory": str(output_dir),
        "eligible_networks": network_order,
        "imaging_participants": int(profile_manifest["profile_participants"]),
        "cognition_participants": int(cognition_manifest["cognition_participants"]),
        "partial_convergence_supported": bool(final_manifest["partial_convergence_supported"]),
        "outputs": checks,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\nPAPER OUTPUTS COMPLETE")
    print(f"Manuscript source tables: {checks['manuscript_tables']}")
    print(f"Main figure families: {checks['main_figure_families']} PNG + PDF (Figure1 also SVG)")
    print(f"Supplementary figure families: {checks['supplementary_figure_families']} PNG + PDF")
    print("Model fitting: none")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
