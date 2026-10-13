#!/usr/bin/env python3
"""Build Figure 1, Table 1, and source CSVs from analyses 01--07."""

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
from matplotlib.patches import Rectangle


STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
STATE_LABELS = {
    "REST": "Rest",
    "CARIT": "CARIT",
    "FACENAME": "FaceName",
    "VISMOTOR": "Visual–motor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
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


# Figure 1: cohort sampling and illustrative covariance decomposition.
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
    """Select the primary imaging cohort from the participant summaries."""
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

    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    audit_dir = output_dir / "audit"
    for directory in [table_dir, figure_dir, audit_dir]:
        directory.mkdir(parents=True, exist_ok=True)

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
    state_profiles, _ = reconstruct_profile_levels(profile_checkpoint, network_order)

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
    table_1.to_csv(table_dir / "Table_1_cohort_and_longitudinal_coverage.csv", index=False)

    # Figure 1: cohort sampling and representation schematic.
    with plt.rc_context():
        fig = build_figure1(load_figure1_cohort(tables))
        save_figure(
            fig,
            figure_dir,
            "Figure1",
            include_svg=True,
            png_dpi=400,
            vector_dpi=600,
            pad_inches=0.03,
        )

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
        frame.to_csv(table_dir / filename, index=False)

    for filename in [
        "01_site_composition.csv",
        "01b_site_state_coverage.csv",
        "04_site_equal_state_similarity.csv",
        "05_site_equal_state_magnitude.csv",
        "06_site_FP_relative_slopes.csv",
    ]:
        source = d07 / "tables" / filename
        shutil.copy2(source, table_dir / filename)
        sources.append({"file": str(source), "rows": len(pd.read_csv(source))})

    source_inventory = pd.DataFrame(sources).drop_duplicates("file").sort_values("file")
    source_inventory.to_csv(audit_dir / "source_file_inventory.csv", index=False)
    table_paths = [table_dir / "Table_1_cohort_and_longitudinal_coverage.csv"]
    table_paths += [table_dir / name for name in supp_tables]
    table_paths += [table_dir / name for name in [
        "01_site_composition.csv", "01b_site_state_coverage.csv",
        "04_site_equal_state_similarity.csv", "05_site_equal_state_magnitude.csv",
        "06_site_FP_relative_slopes.csv",
    ]]
    figure_paths = [figure_dir / f"Figure1.{ext}" for ext in ["png", "pdf", "svg"]]
    if not all(path.is_file() and path.stat().st_size for path in table_paths + figure_paths):
        raise AssertionError("A source table or Figure 1 output is missing or empty.")
    manifest = {
        "script": Path(__file__).name,
        "results_directory": str(results_root),
        "output_directory": str(output_dir),
        "eligible_networks": network_order,
        "imaging_participants": int(profile_manifest["profile_participants"]),
        "cognition_participants": int(cognition_manifest["cognition_participants"]),
        "partial_convergence_supported": bool(final_manifest["partial_convergence_supported"]),
        "source_tables": [str(path.relative_to(output_dir)) for path in table_paths],
        "figures": [str(path.relative_to(output_dir)) for path in figure_paths],
    }
    (audit_dir / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(table_paths)} source CSVs and Figure 1 to {output_dir}")


if __name__ == "__main__":
    main()
