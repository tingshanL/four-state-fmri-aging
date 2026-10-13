"""Shared colors, typography, and drawing helpers for manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

# One meaning per colour, used the same way in every figure.
COLORS = {
    "native": "#0072B2",        # native covariance
    "signal_amp": "#E69F00",    # released signal (noise-removed) amplitude
    "signal_norm": "#009E73",   # signal-normalized (amplitude-standardized) covariance
    "total": "#56B4E9",         # candidate total amplitude / total-normalized covariance
    "noise": "#777777",         # estimated unstructured-noise amplitude 
    "fp": "#7B3294",            # frontoparietal network
    "contrast": "#333333",      # paired differences
}
AMPLITUDE_TEXT = "#B97A00"      # darker amplitude hue for small text on white
GRAY = "#777777"
ZERO = "#9A9A9A"
TEXT = "#1A1A1A"

# Marker shapes distinguish series in monochrome.
MARKERS = {"native": "o", "signal_amp": "o", "total": "D", "noise": "s",
           "signal_norm": "o", "contrast": "o"}

STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
STATE_LABELS = {"REST": "REST", "CARIT": "CARIT", "FACENAME": "FaceName", "VISMOTOR": "Visual–motor"}

NETWORK_ABBREV = {
    "Visual1": "VIS1", "Visual2": "VIS2", "Somatomotor": "SM", "Cingulo-Opercular": "CO",
    "Dorsal-attention": "DA", "Language": "LANG", "Frontoparietal": "FP", "Auditory": "AUD",
    "Default": "DMN", "Posterior-Multimodal": "PMM",
}

SMALL = 6.8
BASE = 7.0
DOUBLE_COLUMN_IN = 7.4  # NeuroImage double column is 190 mm


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.fallback": "cm",
        "font.size": BASE,
        "axes.labelsize": BASE,
        "axes.titlesize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": TEXT,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": SMALL,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def open_axes(ax, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(direction="out")


def panel_label(fig, ax, label: str, title: str | None = None, dx: float = -0.045, y: float | None = None) -> None:
    """Bold letter plus short plain title, aligned to the axes' top-left."""
    box = ax.get_position()
    y = box.y1 + 0.035 if y is None else y
    fig.text(box.x0 + dx, y, label, fontsize=9.5, fontweight="bold", ha="left", va="baseline")
    if title:
        fig.text(box.x0 + dx + 0.025, y, title, fontsize=7.5, ha="left", va="baseline", color=TEXT)


def save(fig, out_dir, basename) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, opts in {"pdf": {}, "svg": {}, "png": {"dpi": 400}}.items():
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03, **opts)
        paths.append(path)
    return paths
