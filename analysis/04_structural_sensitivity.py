#!/usr/bin/env python3
"""Evaluate structural context for frontoparietal-relative fMRI change."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import helmert, qr
from scipy.optimize import minimize
from scipy.stats import chi2, norm, pearsonr, spearmanr

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def display(value):
    """Compact replacement for analysis display calls."""
    print(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Local multimodal project root containing fMRI and structural archives.",
    )
    parser.add_argument(
        "--subject-table", type=Path, required=True,
        help="Longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Local results root used by scripts 02 through 04.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development run with 199 coefficient-change bootstrap draws.",
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Ignore compatible script-04 checkpoints and refit all models.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=None,
        help="Override coefficient-change bootstrap draws (paper default: 999).",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
SUBJECT_TABLE = ARGS.subject_table.expanduser().resolve()
RESULTS_ROOT = ARGS.results_dir.expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "04_structural_sensitivity"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PROFILE_OUTPUT_DIR = RESULTS_ROOT / "02_primary_profile_and_measurement"
SOURCE_PROFILE_TABLE_DIR = SOURCE_PROFILE_OUTPUT_DIR / "tables"
SOURCE_PROFILE_CHECKPOINT_DIR = SOURCE_PROFILE_OUTPUT_DIR / "checkpoints"
SOURCE_FP_OUTPUT_DIR = RESULTS_ROOT / "03_fp_specificity_and_robustness"
SOURCE_FP_CHECKPOINT_DIR = SOURCE_FP_OUTPUT_DIR / "checkpoints"

if not DATA_DIR.is_dir():
    raise NotADirectoryError(DATA_DIR)
if not SUBJECT_TABLE.is_file():
    raise FileNotFoundError(SUBJECT_TABLE)
for dependency in [SOURCE_PROFILE_OUTPUT_DIR, SOURCE_FP_OUTPUT_DIR]:
    if not dependency.is_dir():
        raise FileNotFoundError(f"Required preceding results are missing: {dependency}")

FAST_MODE = bool(ARGS.fast)
FORCE_REFIT = bool(ARGS.force_refit)
MIN_STATE_PARTICIPANTS = 12 if FAST_MODE else 40
EXPECTED_REPEAT_PARTICIPANTS = 852
EXPECTED_CONFIRMATION_PARTICIPANTS = 852
EXPECTED_ELIGIBLE_NETWORK_N = 10
MIN_PARCELS = 2
MIN_STATE_COVERAGE = .90
MIN_STATE_POSITIVE = .99
SEED = 20260829

STRUCTURAL_MIN_PARCEL_FRACTION = .95
COEFFICIENT_CHANGE_BOOTSTRAPS = int(
    ARGS.bootstrap_b if ARGS.bootstrap_b is not None else (199 if FAST_MODE else 999)
)
SUBSTANTIAL_ATTENUATION_FRACTION = .25
UNCHANGED_ATTENUATION_FRACTION = .20
AREA_MATERIAL_LOG_MARGIN_5Y = float(np.log(1.01))
AREA_MATERIAL_PERCENT_MARGIN_5Y = 1.0

FP_NETWORK = "Frontoparietal"
FOCAL_NETWORKS = ["Frontoparietal", "Default", "Cingulo-Opercular"]
STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
CANONICAL_NETWORKS = [
    "Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
    "Dorsal-attention", "Language", "Frontoparietal", "Auditory",
    "Default", "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective",
]
SENSORY_NETWORKS = ["Visual1", "Visual2", "Somatomotor", "Auditory"]
ASSOCIATION_NETWORKS = [
    network for network in CANONICAL_NETWORKS if network not in SENSORY_NETWORKS
]
PRESPECIFIED_METRICS = [
    "within_default", "within_cingulo_opercular", "within_frontoparietal"
]
METRIC_TO_NETWORK = {
    "within_default": "Default",
    "within_cingulo_opercular": "Cingulo-Opercular",
    "within_frontoparietal": "Frontoparietal",
}

pd.set_option("display.max_columns", 260)
sns.set_theme(style="whitegrid", context="paper")



# Construct the repeat-cohort fMRI data

def sha256_file(path: Path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def outside_output(path: Path):
    p = path.resolve()
    return OUTPUT_DIR not in p.parents and ".codex-upload-" not in p.name


def resolve_unique(patterns, required=True, label="file"):
    candidates = []
    for pattern in patterns:
        candidates.extend(p for p in DATA_DIR.rglob(pattern) if p.is_file() and outside_output(p))
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))
    if not candidates:
        if required:
            raise FileNotFoundError(f"Could not resolve {label}; tried {patterns} under {DATA_DIR}")
        return None
    forbidden = [p for p in candidates if "primary_age_sex_adjusted" in p.name.lower()]
    candidates = [p for p in candidates if p not in forbidden]
    if not candidates:
        if required:
            raise RuntimeError(f"Only forbidden age/sex-adjusted candidates were found for {label}")
        return None
    hashes = OrderedDict()
    for p in candidates:
        hashes.setdefault(sha256_file(p), []).append(p)
    if len(hashes) > 1:
        detail = "\n".join(f"  {h[:12]}  {p}" for h, paths in hashes.items() for p in paths)
        raise RuntimeError(f"Ambiguous nonidentical candidates for {label}:\n{detail}")
    return candidates[0]


def norm_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def pick_column(frame, aliases, required=False, label=None):
    lookup = {norm_name(c): c for c in frame.columns}
    for alias in aliases:
        if norm_name(alias) in lookup:
            return lookup[norm_name(alias)]
    if required:
        raise KeyError(f"Missing {label or aliases[0]}; accepted aliases: {aliases}")
    return None


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def normalize_state(value):
    text = norm_name(value)
    for state in STATES:
        if state.lower() in text:
            return state
    return np.nan


source_rows = []
raw_paths = {}
for state in STATES:
    p = resolve_unique(
        [f"full_covariance__{state.lower()}__prespecified_metrics.parquet",
         f"full_covariance__{state.lower()}__prespecified_metrics.csv"],
        required=False, label=f"raw {state} full-covariance metrics",
    )
    if p is not None:
        raw_paths[state] = p

input_scale = None
if len(raw_paths) == 4:
    metric_wide = None
    for state, path in raw_paths.items():
        table = read_table(path)
        id_col = pick_column(table, ["id_event"], required=True, label="id_event")
        table = table.rename(columns={id_col: "id_event"})
        missing = sorted(set(PRESPECIFIED_METRICS) - set(table.columns))
        if missing:
            raise KeyError(f"{path} lacks specified metrics: {missing}")
        table = table[["id_event", *PRESPECIFIED_METRICS]].copy()
        table["id_event"] = table["id_event"].astype(str).str.strip()
        if table["id_event"].duplicated().any():
            raise AssertionError(f"Duplicate id_event rows in {path}")
        table = table.rename(columns={m: f"raw__{state.lower()}__{m}" for m in PRESPECIFIED_METRICS})
        metric_wide = table if metric_wide is None else metric_wide.merge(
            table, on="id_event", how="outer", validate="one_to_one"
        )
        source_rows.append({"role": f"raw_{state}", "path": str(path), "sha256": sha256_file(path)})
    input_scale = "RAW_UNADJUSTED_FULL_COVARIANCE_PRESPECIFIED_METRICS"
else:
    raw_statewise = resolve_unique(
        ["AABC2_four_state_fMRI__full_covariance__raw_statewise.parquet",
         "AABC2_four_state_fMRI__full_covariance__raw_statewise.csv"],
        required=False, label="raw full-covariance statewise export",
    )
    technical_oof = resolve_unique(
        ["*full_covariance*age_sex_preserved_technical_only__statewise_oof.parquet",
         "*full_covariance*age_sex_preserved_technical_only__statewise_oof.csv"],
        required=False, label="age/sex-preserved technical-only statewise OOF export",
    )
    selected = raw_statewise or technical_oof
    if selected is None:
        raise FileNotFoundError(
            "Need all four raw per-state full-covariance metric caches, the raw statewise export, "
            "or the age/sex-preserved technical-only statewise OOF export."
        )
    if "primary_age_sex_adjusted" in selected.name.lower():
        raise AssertionError("Age/sex-adjusted fMRI representation is prohibited.")
    table = read_table(selected)
    id_col = pick_column(table, ["id_event"], required=True, label="id_event")
    table = table.rename(columns={id_col: "id_event"})
    table["id_event"] = table["id_event"].astype(str).str.strip()
    metric_wide = table[["id_event"]].copy()
    recognized = {}
    for state in STATES:
        for metric in PRESPECIFIED_METRICS:
            aliases = [
                f"raw__{state.lower()}__{metric}", f"resid__{state.lower()}__{metric}",
                f"residual__{state.lower()}__{metric}", f"oof__{state.lower()}__{metric}",
                f"{state.lower()}__{metric}", f"fmri__{state.lower()}__{metric}",
            ]
            col = pick_column(table, aliases)
            if col is None:
                raise KeyError(
                    f"Could not bind {state}/{metric} in {selected}. "
                    "No approximate or age/sex-adjusted fallback is allowed."
                )
            target = f"raw__{state.lower()}__{metric}"
            metric_wide[target] = pd.to_numeric(table[col], errors="coerce")
            recognized[target] = col
    if metric_wide["id_event"].duplicated().any():
        raise AssertionError(f"Duplicate id_event rows in {selected}")
    input_scale = (
        "RAW_UNADJUSTED_FULL_COVARIANCE_STATEWISE"
        if selected == raw_statewise else
        "AGE_SEX_PRESERVED_TECHNICAL_ONLY_STATEWISE_OOF"
    )
    source_rows.append({"role": "statewise_metric_input", "path": str(selected), "sha256": sha256_file(selected)})

for col in metric_wide.columns:
    if col != "id_event":
        metric_wide[col] = pd.to_numeric(metric_wide[col], errors="coerce")

subject_path = SUBJECT_TABLE
subject_resolution = "COMMAND_LINE_ARGUMENT"
subjects_raw = read_table(subject_path)
source_rows.append({
    "role": "longitudinal_subject_event_table",
    "path": str(subject_path),
    "sha256": sha256_file(subject_path),
    "resolution": subject_resolution,
})
source_manifest = pd.DataFrame(source_rows)
display(source_manifest)
print("Selected full-covariance metric scale:", input_scale)

id_event_col = pick_column(subjects_raw, ["id_event", "subject_event", "participant_event"])
raw_id_col = pick_column(subjects_raw, ["id", "subject_id", "participant_id", "src_subject_id"])
raw_event_col = pick_column(subjects_raw, ["eventname", "event_name", "event", "visit", "session"])
if id_event_col is None:
    if raw_id_col is None or raw_event_col is None:
        raise KeyError(
            "The event table needs id_event, or both a participant-ID column and an event/visit column."
        )
    subjects_raw = subjects_raw.copy()
    subjects_raw["__constructed_id_event"] = (
        subjects_raw[raw_id_col].astype(str).str.strip() + "_" +
        subjects_raw[raw_event_col].astype(str).str.strip()
    )
    id_event_col = "__constructed_id_event"
age_col = pick_column(subjects_raw, ["age_open", "age", "event_age", "interview_age"], required=True, label="event age")
sex_col = pick_column(subjects_raw, ["sex", "gender"], required=True, label="sex")
site_col = pick_column(subjects_raw, ["site", "site_name", "site_id", "collection_site", "institution"], required=True, label="site")
scanner_col = pick_column(subjects_raw, [
    "scanner", "scanner_id", "scanner_name", "acquisition_scanner", "mri_scanner",
    "scanner_model", "scanner_serial_number",
])
days_col = pick_column(subjects_raw, ["days_from_V1", "days_from_v1", "daysfromv1"])
id_col = raw_id_col
study_col = pick_column(subjects_raw, ["study", "project", "cohort"])
pedigree_col = pick_column(subjects_raw, ["pedid", "pedigree", "pedigree_id", "family_id"])
generic_qc_col = pick_column(subjects_raw, ["MR_QC_Issue_Codes", "mr_qc_issue_codes"])

keep_map = {id_event_col: "id_event", age_col: "event_age", sex_col: "sex", site_col: "site"}
if scanner_col is not None:
    keep_map[scanner_col] = "scanner"
for original, target in [
    (days_col, "days_from_v1"), (id_col, "id"), (study_col, "study"),
    (pedigree_col, "pedigree_group"), (generic_qc_col, "mr_qc_issue_codes"),
]:
    if original is not None:
        keep_map[original] = target
subjects = subjects_raw[list(keep_map)].rename(columns=keep_map).copy()
subjects["id_event"] = subjects["id_event"].astype(str).str.strip()
subjects["event_age"] = pd.to_numeric(subjects["event_age"], errors="coerce")
if "scanner" not in subjects:
    subjects["scanner"] = "MISSING"
for c in ["sex", "site", "scanner"]:
    subjects[c] = subjects[c].astype("string").str.strip().fillna("MISSING")
if "id" not in subjects:
    subjects["id"] = subjects["id_event"].str.replace(r"_[^_]+$", "", regex=True)
subjects["id"] = subjects["id"].astype(str).str.strip()
if "study" not in subjects:
    subjects["study"] = "MISSING"
if "pedigree_group" not in subjects:
    subjects["pedigree_group"] = "UNRELATED__" + subjects["id"]
subjects["pedigree_group"] = subjects["pedigree_group"].astype("string").str.strip()
subjects.loc[subjects["pedigree_group"].isin(["", "<NA>", "NAN"]), "pedigree_group"] = (
    "UNRELATED__" + subjects.loc[subjects["pedigree_group"].isin(["", "<NA>", "NAN"]), "id"]
)
if "days_from_v1" in subjects:
    subjects["days_from_v1"] = pd.to_numeric(subjects["days_from_v1"], errors="coerce")
else:
    subjects["days_from_v1"] = np.nan

def generic_issue_flag(series):
    text = series.astype("string").str.strip().str.upper()
    no_issue = (
        text.isna()
        | text.isin([
            "", "0", "0.0", "NONE", "NO", "NO CODE",
            "NO ISSUE", "NO ISSUES", "NO QC ISSUE", "NO QC ISSUES",
            "N/A", "NA", "NOT APPLICABLE", "NORMAL", "PASS", "PASSED", "OK",
            "[]", "{}", "<NA>", "NAN",
        ])
        | text.str.fullmatch(r"0(?:\.0+)?", na=False)
        | text.str.fullmatch(r"[\[\]\{\}\(\),;\s]*", na=False)
    )
    return (~no_issue).astype(float)

if "mr_qc_issue_codes" in subjects:
    subjects["generic_mr_qc_flag"] = generic_issue_flag(subjects["mr_qc_issue_codes"])
else:
    subjects["generic_mr_qc_flag"] = np.nan

qc_path = resolve_unique(
    ["AABC2_fMRI_QC_summary.csv", "AABC_Release2_fMRI_QC_summary.csv", "fMRI_QC_summary.csv",
     "AABC2_fMRI_QC_summary.parquet", "fMRI_QC_summary.parquet"],
    required=False, label="optional state-specific fMRI QC summary",
)
qc_long = pd.DataFrame(columns=["id_event", "state", "motion", "tsnr", "usable"])
qc_resource_rows = []
if qc_path is not None:
    q = read_table(qc_path)
    q_id = pick_column(q, ["id_event"], required=True, label="QC id_event")
    q_state = pick_column(q, ["state", "task", "scan", "run", "paradigm"])
    motion_col = pick_column(q, ["motion", "mean_fd", "meanfd", "framewise_displacement", "mean_framewise_displacement"])
    tsnr_col = pick_column(q, ["tsnr", "temporal_snr", "temporalsnr"])
    usable_col = pick_column(q, ["usable", "usability", "fmri_pass", "qc_status", "include"])
    if q_state is not None:
        qc_long = pd.DataFrame({
            "id_event": q[q_id].astype(str).str.strip(),
            "state": q[q_state].map(normalize_state),
            "motion": pd.to_numeric(q[motion_col], errors="coerce") if motion_col else np.nan,
            "tsnr": pd.to_numeric(q[tsnr_col], errors="coerce") if tsnr_col else np.nan,
            "usable_raw": q[usable_col] if usable_col else np.nan,
        }).dropna(subset=["state"])
        if usable_col:
            text = qc_long["usable_raw"].astype("string").str.strip().str.upper()
            numeric = pd.to_numeric(qc_long["usable_raw"], errors="coerce")
            qc_long["usable"] = np.where(
                numeric.notna(), numeric.gt(0).astype(float),
                text.isin(["Y", "YES", "TRUE", "PASS", "PASSED", "USABLE", "INCLUDE"]).astype(float),
            )
        else:
            qc_long["usable"] = np.nan
        qc_long = qc_long.groupby(["id_event", "state"], as_index=False).agg(
            motion=("motion", "mean"), tsnr=("tsnr", "mean"), usable=("usable", "min")
        )
        source_manifest.loc[len(source_manifest)] = {
            "role": "optional_state_specific_fmri_qc", "path": str(qc_path), "sha256": sha256_file(qc_path)
        }
    else:
        qc_resource_rows.append({"resource": str(qc_path), "status": "FOUND_BUT_NO_STATE_COLUMN", "used_as_motion": False})

for kind in ["motion", "tsnr", "usable"]:
    available = kind in qc_long and qc_long[kind].notna().any()
    qc_resource_rows.append({
        "resource": kind, "status": "AVAILABLE" if available else "UNAVAILABLE",
        "coverage": float(qc_long[kind].notna().mean()) if len(qc_long) and kind in qc_long else 0.0,
        "used_as_motion": bool(kind == "motion" and available),
    })
qc_resource_rows.append({
    "resource": "MR_QC_Issue_Codes", "status": "AVAILABLE" if generic_qc_col else "UNAVAILABLE",
    "coverage": float(subjects["generic_mr_qc_flag"].notna().mean()), "used_as_motion": False,
})
qc_resource_audit = pd.DataFrame(qc_resource_rows)
display(qc_resource_audit)

def compact_token(value):
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def resolve_node_manifest():
    candidates = sorted({
        path.resolve() for path in DATA_DIR.rglob("fmri_node_parcel_manifest.csv")
        if path.is_file() and OUTPUT_DIR not in path.parents
    })
    valid = []
    for path in candidates:
        table = pd.read_csv(path, low_memory=False)
        area = pick_column(table, ["Brain Area", "brain_area", "parcel", "area"])
        node = pick_column(table, ["node", "network"])
        index = pick_column(table, ["Real Index", "real_index", "index"])
        if area and node and index:
            canonical = table[[area, node, index]].rename(columns={area: "Brain Area", node: "node", index: "Real Index"})
            canonical["Real Index"] = pd.to_numeric(canonical["Real Index"], errors="coerce")
            valid.append((path, canonical.dropna().sort_values("Real Index").reset_index(drop=True)))
    if not valid:
        raise FileNotFoundError("The selected fmri_node_parcel_manifest.csv is required")
    signatures = {hashlib.sha256(table.to_csv(index=False).encode()).hexdigest() for _, table in valid}
    if len(signatures) > 1:
        raise RuntimeError("Nonidentical node-parcel manifests were found")
    return valid[0]


node_manifest_path, parcel_map = resolve_node_manifest()
parcel_map["node_token"] = parcel_map["node"].map(compact_token)
canonical_token = {network: compact_token(network) for network in CANONICAL_NETWORKS}
parcel_counts = pd.DataFrame([
    {"network": network, "parcel_n": int(parcel_map["node_token"].eq(token).sum())}
    for network, token in canonical_token.items()
])
source_manifest.loc[len(source_manifest)] = {
    "role": "node_parcel_manifest", "path": str(node_manifest_path), "sha256": sha256_file(node_manifest_path)
}


def resolve_node_table(family, state):
    stem = family.lower()
    return resolve_unique(
        [f"{stem}__{state.lower()}__node_connectivity.parquet",
         f"{stem}__{state.lower()}__node_connectivity.csv"],
        required=True, label=f"{family}/{state} node connectivity",
    )


def load_within_network_wide(family):
    combined = None
    audit = []
    for state in STATES:
        path = resolve_node_table(family, state)
        table = read_table(path)
        id_col = pick_column(table, ["id_event"], required=True, label=f"{family}/{state} id_event")
        table = table.rename(columns={id_col: "id_event"})
        expected = {network: f"fc__{slug(network)}__{slug(network)}" for network in CANONICAL_NETWORKS}
        missing = [column for column in expected.values() if column not in table]
        if missing:
            raise KeyError(f"{path} lacks canonical within-network columns: {missing}")
        part = table[["id_event", *expected.values()]].copy()
        part["id_event"] = part["id_event"].astype(str).str.strip()
        if part["id_event"].duplicated().any():
            raise AssertionError(f"Duplicate id_event rows in {path}")
        part = part.rename(columns={column: f"{family.lower()}__{state.lower()}__{slug(network)}"
                                    for network, column in expected.items()})
        combined = part if combined is None else combined.merge(part, on="id_event", how="outer", validate="one_to_one")
        audit.append({"family": family, "state": state, "path": str(path), "sha256": sha256_file(path), "rows": len(part)})
    return combined, pd.DataFrame(audit)


full_node_wide, full_node_sources = load_within_network_wide("full_covariance")
partial_node_wide, partial_node_sources = load_within_network_wide("partial_covariance")


def build_event_timing(wide):
    events = wide[["id_event"]].merge(subjects, on="id_event", how="inner", validate="one_to_one")
    value_cols = [column for column in wide if column.startswith("raw__")]
    availability = wide[["id_event", *value_cols]].copy()
    availability["any_metric"] = availability[value_cols].notna().any(axis=1)
    events = events.merge(availability[["id_event", "any_metric"]], on="id_event", validate="one_to_one")
    events = events.loc[events["any_metric"]].copy()
    events["sort_time"] = pd.to_numeric(events["days_from_v1"], errors="coerce")
    events["sort_time"] = events["sort_time"].where(
        events["sort_time"].notna(), pd.to_numeric(events["event_age"], errors="coerce") * 365.25
    )
    events = events.sort_values(["id", "sort_time", "id_event"])
    events["baseline_sort_time"] = events.groupby("id")["sort_time"].transform("min")
    baseline_age = events.dropna(subset=["event_age"]).groupby("id")["event_age"].first()
    events["baseline_age"] = events["id"].map(baseline_age)
    events["time_years"] = (events["sort_time"] - events["baseline_sort_time"]) / 365.25
    events["t5"] = events["time_years"] / 5.0
    center = float(baseline_age.mean())
    events["age0_c5"] = (events["baseline_age"] - center) / 5.0
    events["age0_c10"] = (events["baseline_age"] - center) / 10.0
    events["visit_number"] = events.groupby("id").cumcount() + 1
    events["site_scanner"] = events["site"].astype(str) + " | " + events["scanner"].astype(str)
    return events, center


df_timing_cols = ["id_event"] + [
    column for column in metric_wide.columns
    if column.startswith("raw__") and (column.endswith("__within_default") or column.endswith("__within_frontoparietal"))
]
events, BASELINE_AGE_CENTER = build_event_timing(metric_wide[df_timing_cols].copy())
CORE_COVARS = [
    "id_event", "id", "pedigree_group", "study", "sex", "site", "scanner", "site_scanner",
    "event_age", "baseline_age", "age0_c5", "age0_c10", "time_years", "t5", "visit_number",
]


def representation_rows(wide, family):
    merged = events.merge(wide, on="id_event", how="left", validate="one_to_one")
    rows = []
    for state in STATES:
        part = merged[CORE_COVARS].copy()
        part["state"] = state
        for network in CANONICAL_NETWORKS:
            column = f"{family.lower()}__{state.lower()}__{slug(network)}"
            part[f"cov__{network}"] = pd.to_numeric(merged[column], errors="coerce")
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


full_rows = representation_rows(full_node_wide, "full_covariance")
partial_rows = representation_rows(partial_node_wide, "partial_covariance")

# Reproduce the selected DMN/FP repeat parent before testing other networks.
parent_ok = (
    np.isfinite(full_rows[["age0_c5", "t5"]].to_numpy(float)).all(axis=1)
    & (full_rows["cov__Default"] > 0)
    & (full_rows["cov__Frontoparietal"] > 0)
)
parent_candidate = full_rows.loc[parent_ok].copy()
parent_session_n = parent_candidate[["id", "id_event"]].drop_duplicates().groupby("id")["id_event"].nunique()
repeat_ids = set(parent_session_n[parent_session_n.ge(2)].index)
parent_rows = parent_candidate.loc[parent_candidate["id"].isin(repeat_ids)].copy()
if not FAST_MODE and len(repeat_ids) != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Repeat cohort mismatch: expected {EXPECTED_REPEAT_PARTICIPANTS}, found {len(repeat_ids)}"
    )

# Eligibility uses only the repeat-cohort observations and is completed before slopes.
eligibility_rows = []
for network in CANONICAL_NETWORKS:
    parcel_n = int(parcel_counts.loc[parcel_counts["network"].eq(network), "parcel_n"].iloc[0])
    for state in STATES:
        values = pd.to_numeric(parent_rows.loc[parent_rows["state"].eq(state), f"cov__{network}"], errors="coerce")
        finite = values.notna()
        eligibility_rows.append({
            "network": network, "state": state, "parcel_n": parcel_n, "opportunity_rows": len(values),
            "finite_rows": int(finite.sum()), "positive_rows": int((values > 0).sum()),
            "coverage": float(finite.mean()) if len(values) else 0.0,
            "positive_among_finite": float((values[finite] > 0).mean()) if finite.any() else 0.0,
        })
network_state_eligibility = pd.DataFrame(eligibility_rows)
network_eligibility = network_state_eligibility.groupby("network", as_index=False).agg(
    parcel_n=("parcel_n", "first"), min_state_coverage=("coverage", "min"),
    min_state_positive=("positive_among_finite", "min"),
)
network_eligibility["eligible"] = (
    network_eligibility["parcel_n"].ge(MIN_PARCELS)
    & network_eligibility["min_state_coverage"].ge(MIN_STATE_COVERAGE)
    & network_eligibility["min_state_positive"].ge(MIN_STATE_POSITIVE)
)
network_eligibility["reason"] = np.select([
    network_eligibility["parcel_n"].lt(MIN_PARCELS),
    network_eligibility["min_state_coverage"].lt(MIN_STATE_COVERAGE),
    network_eligibility["min_state_positive"].lt(MIN_STATE_POSITIVE),
], ["TOO_FEW_PARCELS", "INSUFFICIENT_COVERAGE", "NONPOSITIVE_VALUES"], default="PASS")
ELIGIBLE_NETWORKS = [network for network in CANONICAL_NETWORKS
                     if bool(network_eligibility.set_index("network").loc[network, "eligible"])]
if len(ELIGIBLE_NETWORKS) < 3:
    raise AssertionError(f"Only {len(ELIGIBLE_NETWORKS)} networks passed the eligibility criteria")
if not FAST_MODE and len(ELIGIBLE_NETWORKS) != EXPECTED_ELIGIBLE_NETWORK_N:
    raise AssertionError(
        f"Confirmation requires the existing ten-network set; recovered {len(ELIGIBLE_NETWORKS)}: {ELIGIBLE_NETWORKS}"
    )
for required in ["Default", "Cingulo-Opercular", "Frontoparietal"]:
    if required not in ELIGIBLE_NETWORKS:
        raise AssertionError(f"Required continuity network failed eligibility: {required}")


def retain_two_sessions(frame):
    counts = frame[["id", "id_event"]].drop_duplicates().groupby("id")["id_event"].nunique()
    return frame.loc[frame["id"].isin(counts[counts.ge(2)].index)].copy()


complete_full = np.isfinite(parent_rows[[f"cov__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float)).all(axis=1)
positive_full = (parent_rows[[f"cov__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float) > 0).all(axis=1)
profile_data = retain_two_sessions(parent_rows.loc[complete_full & positive_full].copy())
K = len(ELIGIBLE_NETWORKS)
if not FAST_MODE and profile_data["id"].nunique() != EXPECTED_CONFIRMATION_PARTICIPANTS:
    raise AssertionError(
        f"Confirmation requires 852 repeat participants; recovered {profile_data['id'].nunique()}"
    )


def derive_profile(frame, covariance_prefix="cov__", output_suffix=""):
    out = frame.copy()
    l_columns = []
    for network in ELIGIBLE_NETWORKS:
        target = f"L{output_suffix}__{network}"
        values = pd.to_numeric(out[f"{covariance_prefix}{network}"], errors="coerce").astype(np.float64)
        out[target] = np.log(values.where(values > 0))
        l_columns.append(target)
    out[f"G{output_suffix}"] = out[l_columns].to_numpy(dtype=np.float64).mean(axis=1)
    for network, column in zip(ELIGIBLE_NETWORKS, l_columns):
        out[f"R{output_suffix}__{network}"] = (
            out[column].to_numpy(dtype=np.float64) - out[f"G{output_suffix}"].to_numpy(dtype=np.float64)
        )
    return out


profile_data = derive_profile(profile_data)
contrast_error = (
    (profile_data["R__Default"] - profile_data["R__Frontoparietal"])
    - (profile_data["L__Default"] - profile_data["L__Frontoparietal"])
).to_numpy(dtype=np.float64)
r_columns = [f"R__{network}" for network in ELIGIBLE_NETWORKS]
profile_sum_error = profile_data[r_columns].to_numpy(dtype=np.float64).sum(axis=1)
profile_identity_audit = pd.DataFrame([{
    "identity": "R_Default - R_Frontoparietal = L_Default - L_Frontoparietal",
    "max_absolute_error": float(np.nanmax(np.abs(contrast_error))),
    "max_profile_sum_error": float(np.nanmax(np.abs(profile_sum_error))),
}])
audit_errors = profile_identity_audit[["max_absolute_error", "max_profile_sum_error"]].to_numpy(float)
if not np.isfinite(audit_errors).all() or audit_errors.max() > 1e-10:
    display(profile_identity_audit)
    raise AssertionError("Common-level/profile algebra failed after float64 reconstruction")

# Freeze optional partition before slopes; never redefine it after eligibility.
association_sensory_available = set(ASSOCIATION_NETWORKS + SENSORY_NETWORKS).issubset(ELIGIBLE_NETWORKS)
planned_contrast_definitions = [
    {"contrast": "FP_MINUS_MEAN_DMN_CO", "positive": {"Frontoparietal": 1.0},
     "negative": {"Default": .5, "Cingulo-Opercular": .5}},
    {"contrast": "DMN_MINUS_CO", "positive": {"Default": 1.0},
     "negative": {"Cingulo-Opercular": 1.0}},
]
if association_sensory_available:
    planned_contrast_definitions.append({
        "contrast": "ASSOCIATION_MINUS_SENSORY",
        "positive": {network: 1 / len(ASSOCIATION_NETWORKS) for network in ASSOCIATION_NETWORKS},
        "negative": {network: 1 / len(SENSORY_NETWORKS) for network in SENSORY_NETWORKS},
    })

display(parcel_counts)
display(network_state_eligibility)
display(network_eligibility)
display(profile_identity_audit)
print("Eligible networks selected before slopes:", ELIGIBLE_NETWORKS)
print("Association-sensory planned contrast available:", association_sensory_available)

def identify_id_column(columns):
    exact = [column for column in columns if compact_token(column) == "IDEVENT"]
    if len(exact) == 1:
        return exact[0]
    candidates = [column for column in columns if compact_token(column) in {
        "IDEVENT", "SUBJECTEVENT", "SUBJECTID", "SUBJECT", "ID", "PARTICIPANTID", "X", "SESSION"
    }]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Could not identify one ID column; header begins {list(columns)[:12]}")


def classify_state_from_name(value):
    token = compact_token(value)
    if "VISMOTOR" in token: return "VISMOTOR"
    if "FACENAME" in token: return "FACENAME"
    if "CARIT" in token: return "CARIT"
    if "REST" in token and ("FMRI" in token or "RFMRI" in token): return "REST"
    return None


def resolve_amplitude_zip(state):
    hits = [path.resolve() for path in DATA_DIR.rglob("*fMRIAmplitudes.zip")
            if classify_state_from_name(path.name) == state and OUTPUT_DIR not in path.parents]
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one {state} fMRIAmplitudes ZIP; found {len(hits)}")
    return hits[0]


def amplitude_member(zip_path):
    with ZipFile(zip_path) as archive:
        hits = [name for name in archive.namelist() if name.lower().endswith(".csv")
                and compact_token(Path(name).stem).endswith("FULLAMPLITUDES")]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one FullAmplitudes.csv in {zip_path.name}; found {len(hits)}")
    return hits[0]


def load_pair_amplitude(state):
    zip_path = resolve_amplitude_zip(state)
    member = amplitude_member(zip_path)
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        header = list(pd.read_csv(stream, nrows=0).columns)
    id_col = identify_id_column(header)
    lookup = {compact_token(column): column for column in header}
    network_areas = {
        network: parcel_map.loc[parcel_map["node_token"].eq(compact_token(network)), "Brain Area"].astype(str).tolist()
        for network in ELIGIBLE_NETWORKS
    }
    required = sorted(set(area for areas in network_areas.values() for area in areas))
    missing = [area for area in required if compact_token(area) not in lookup]
    if missing:
        raise KeyError(f"{member} lacks {len(missing)} selected parcels; first {missing[:5]}")
    columns = [id_col, *[lookup[compact_token(area)] for area in required]]
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        table = pd.read_csv(stream, usecols=columns, low_memory=False)
    rows = {"id_event": table[id_col].astype(str).str.strip()}
    for network, areas in network_areas.items():
        values = table[[lookup[compact_token(area)] for area in areas]].apply(pd.to_numeric, errors="coerce")
        parcel_n = values.notna().sum(axis=1)
        coverage = parcel_n.ge(max(2, math.ceil(.75 * len(areas))))
        total = values.sum(axis=1, skipna=True)
        total_sq = values.pow(2).sum(axis=1, skipna=True)
        rows[f"P__{network}"] = ((total.pow(2) - total_sq) / (parcel_n * (parcel_n - 1))).where(coverage)
    source_manifest.loc[len(source_manifest)] = {
        "role": f"{state}_FullAmplitudes", "path": str(zip_path),
        "sha256": "NOT_COMPUTED_LARGE_ARCHIVE", "size_bytes": zip_path.stat().st_size,
    }
    result = pd.DataFrame(rows)
    result["state"] = state
    return result


amplitude_wide = pd.concat([load_pair_amplitude(state) for state in STATES], ignore_index=True)
amplitude_data = profile_data.merge(amplitude_wide, on=["id_event", "state"], how="left", validate="one_to_one")
amplitude_complete = np.isfinite(amplitude_data[[f"P__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float)).all(axis=1)
amplitude_positive = (amplitude_data[[f"P__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float) > 0).all(axis=1)
amplitude_data = retain_two_sessions(amplitude_data.loc[amplitude_complete & amplitude_positive].copy())
for network in ELIGIBLE_NETWORKS:
    amplitude_data[f"cov_std__{network}"] = np.exp(amplitude_data[f"L__{network}"]) / amplitude_data[f"P__{network}"]
amplitude_standardized = derive_profile(amplitude_data, covariance_prefix="cov_std__", output_suffix="_STD")

# Partial positivity/coverage is audited on the primary-profile opportunities.
partial_on_primary = partial_rows.merge(
    profile_data[["id", "id_event", "state"]], on=["id", "id_event", "state"], how="inner", validate="one_to_one"
)
partial_audit_rows = []
for state in STATES:
    group = partial_on_primary.loc[partial_on_primary["state"].eq(state)]
    for network in ELIGIBLE_NETWORKS:
        values = pd.to_numeric(group[f"cov__{network}"], errors="coerce")
        partial_audit_rows.append({
            "state": state, "network": network, "opportunity_rows": len(values),
            "finite_rows": int(values.notna().sum()), "positive_rows": int((values > 0).sum()),
            "nonpositive_rows": int((values <= 0).sum()),
            "coverage": float(values.notna().mean()) if len(values) else 0.0,
        })
partial_positivity_coverage = pd.DataFrame(partial_audit_rows)
partial_complete = np.isfinite(partial_on_primary[[f"cov__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float)).all(axis=1)
partial_positive = (partial_on_primary[[f"cov__{network}" for network in ELIGIBLE_NETWORKS]].to_numpy(float) > 0).all(axis=1)
partial_complete_rows = partial_on_primary.loc[partial_complete & partial_positive].copy()
partial_ids = partial_complete_rows[["id", "id_event"]].drop_duplicates().groupby("id")["id_event"].nunique()
partial_complete_rows = partial_complete_rows.loc[partial_complete_rows["id"].isin(partial_ids[partial_ids.ge(2)].index)]
partial_keys = partial_complete_rows[["id", "id_event", "state"]]
full_partial_matched = profile_data.merge(partial_keys, on=["id", "id_event", "state"], how="inner", validate="one_to_one")
partial_matched = derive_profile(partial_complete_rows)

scanner_n = profile_data[["id", "id_event", "scanner"]].drop_duplicates().groupby("id")["scanner"].nunique(dropna=False)
same_scanner_data = retain_two_sessions(profile_data.loc[profile_data["id"].map(scanner_n.eq(1)).fillna(False)])


def analysis_frame(frame, profile_suffix=""):
    out = frame.copy()
    out["outcome_G"] = out[f"G{profile_suffix}"]
    for network in ELIGIBLE_NETWORKS:
        out[f"profile__{network}"] = out[f"R{profile_suffix}__{network}"]
    return out


FRAME_REGISTRY = {
    "FULL_PRIMARY": analysis_frame(profile_data),
    "RAW_AMPLITUDE_MATCHED": analysis_frame(amplitude_data),
    "AMPLITUDE_STANDARDIZED": analysis_frame(amplitude_standardized, "_STD"),
    "FULL_PARTIAL_MATCHED": analysis_frame(full_partial_matched),
    "PARTIAL_MATCHED": analysis_frame(partial_matched),
    "SAME_SCANNER": analysis_frame(same_scanner_data),
}
for site in sorted(profile_data["site"].astype(str).unique()):
    key = "LOO_" + re.sub(r"[^A-Za-z0-9]+", "_", site).strip("_").upper()
    FRAME_REGISTRY[key] = analysis_frame(retain_two_sessions(profile_data.loc[profile_data["site"].astype(str).ne(site)]))

SCENARIOS = [
    {"scenario": "FULL_PRIMARY", "frame": "FULL_PRIMARY", "role": "PRIMARY_FULL_COVARIANCE", "site_left_out": None},
    {"scenario": "RAW_AMPLITUDE_MATCHED", "frame": "RAW_AMPLITUDE_MATCHED", "role": "RAW_ON_AMPLITUDE_MATCHED_ROWS", "site_left_out": None},
    {"scenario": "AMPLITUDE_STANDARDIZED", "frame": "AMPLITUDE_STANDARDIZED", "role": "FIXED_AMPLITUDE_STANDARDIZATION", "site_left_out": None},
    {"scenario": "FULL_PARTIAL_MATCHED", "frame": "FULL_PARTIAL_MATCHED", "role": "FULL_ON_PARTIAL_MATCHED_ROWS", "site_left_out": None},
    {"scenario": "PARTIAL_MATCHED", "frame": "PARTIAL_MATCHED", "role": "PARTIAL_COVARIANCE_SENSITIVITY", "site_left_out": None},
    {"scenario": "SAME_SCANNER", "frame": "SAME_SCANNER", "role": "SAME_SCANNER_SENSITIVITY", "site_left_out": None},
]
for key in [key for key in FRAME_REGISTRY if key.startswith("LOO_")]:
    site = next(site for site in sorted(profile_data["site"].astype(str).unique())
                if key == "LOO_" + re.sub(r"[^A-Za-z0-9]+", "_", site).strip("_").upper())
    SCENARIOS.append({"scenario": key, "frame": key, "role": "LEAVE_ONE_SITE_OUT", "site_left_out": site})


def cohort_row(label, frame):
    visits = frame[["id", "id_event"]].drop_duplicates()
    counts = visits.groupby("id")["id_event"].nunique() if len(visits) else pd.Series(dtype=float)
    return {"cohort": label, "participants": int(frame["id"].nunique()), "sessions": int(frame["id_event"].nunique()),
            "session_state_rows": len(frame), "participants_2plus": int((counts >= 2).sum()),
            "participants_3plus": int((counts >= 3).sum())}

cohort_summary = pd.DataFrame([cohort_row(spec["scenario"], FRAME_REGISTRY[spec["frame"]]) for spec in SCENARIOS])
amplitude_coverage = pd.DataFrame([{
    "network": network, "rows": len(profile_data),
    "positive_pair_amplitude_rows": int((pd.to_numeric(profile_data.merge(
        amplitude_wide, on=["id_event", "state"], how="left")[f"P__{network}"], errors="coerce") > 0).sum()),
} for network in ELIGIBLE_NETWORKS])
amplitude_coverage["positive_coverage"] = amplitude_coverage["positive_pair_amplitude_rows"] / amplitude_coverage["rows"]

display(cohort_summary)
display(amplitude_coverage)
display(partial_positivity_coverage)



# Statistical model utilities

def holm_adjust(values):
    p = np.asarray(values, float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[idx]))
        out[idx] = running
    return out


class DirectStateDesign:
    def __init__(self, age_moderation=False, amplitude_ratio=None):
        self.age_moderation = bool(age_moderation)
        self.amplitude_ratio = amplitude_ratio

    @staticmethod
    def _sum(series, levels, prefix):
        text = series.astype(str)
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(np.where(text == level, 1.0, np.where(text == levels[-1], -1.0, 0.0)))
            names.append(f"{prefix}[{level}]")
        return arrays, names

    @staticmethod
    def _dummy(series, levels, prefix):
        text = series.astype("string").fillna("MISSING")
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(text.eq(level).to_numpy(float))
            names.append(f"{prefix}[{level}]")
        return arrays, names

    @staticmethod
    def _stable_rank(matrix):
        if not np.isfinite(matrix).all():
            return -1
        scale = np.sqrt(np.mean(matrix ** 2, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        _, r = qr(matrix / scale, mode="economic", pivoting=False, check_finite=False)
        diagonal = np.abs(np.diag(r))
        if not len(diagonal):
            return 0
        tol = max(matrix.shape) * np.finfo(float).eps * max(diagonal.max(), 1.0)
        return int(np.sum(diagonal > tol))

    @staticmethod
    def _coded_weight(level, levels, coded):
        if level == coded:
            return 1.0
        if level == levels[-1]:
            return -1.0
        return 0.0

    def fit(self, frame):
        self.sex_levels = sorted(frame["sex"].astype("string").fillna("MISSING").unique())
        self.study_levels = sorted(frame["study"].astype("string").fillna("MISSING").unique())
        self.acq_levels = sorted(frame["site_scanner"].astype("string").fillna("MISSING").unique())
        self.site_levels = sorted(frame["site"].astype("string").fillna("MISSING").unique())
        raw, names = self._raw(frame)
        if not np.isfinite(raw).all():
            bad = [names[j] for j in np.flatnonzero(~np.isfinite(raw).all(axis=0))]
            raise ValueError(f"Nonfinite direct-design columns: {bad}")

        mandatory = ["Intercept", "age0_c5", "t5"]
        mandatory += [f"state_sum[{state}]" for state in STATES[:-1]]
        mandatory += [f"age0_c5:state_sum[{state}]" for state in STATES[:-1]]
        mandatory += [f"t5:state_sum[{state}]" for state in STATES[:-1]]
        for site in self.site_levels[:-1]:
            mandatory.append(f"t5:site_sum[{site}]")
            mandatory += [f"t5:site_sum[{site}]:state_sum[{state}]" for state in STATES[:-1]]
        if self.age_moderation:
            mandatory.append("age0_c10:t5")
            mandatory += [f"age0_c10:t5:state_sum[{state}]" for state in STATES[:-1]]
        if self.amplitude_ratio is not None:
            for component in [f"{self.amplitude_ratio}_between", f"{self.amplitude_ratio}_within"]:
                mandatory.append(component)
                mandatory += [f"{component}:state_sum[{state}]" for state in STATES[:-1]]
        missing = [name for name in mandatory if name not in names]
        if missing:
            raise AssertionError(f"Missing direct estimand terms: {missing}")

        mandatory_idx = [names.index(name) for name in mandatory]
        keep, rank = [], 0
        for j in mandatory_idx + [k for k in range(raw.shape[1]) if k not in mandatory_idx]:
            new_rank = self._stable_rank(raw[:, keep + [j]])
            if new_rank > rank:
                keep.append(j)
                rank = new_rank
            elif j in mandatory_idx:
                raise AssertionError(f"Direct estimand is aliased: {names[j]}")
        self.keep = sorted(keep)
        self.names = [names[j] for j in self.keep]
        self.dropped_names = [names[j] for j in range(len(names)) if j not in self.keep]
        return self

    def _raw(self, frame):
        n = len(frame)
        age5 = pd.to_numeric(frame["age0_c5"], errors="coerce").to_numpy(float)
        age10 = pd.to_numeric(frame["age0_c10"], errors="coerce").to_numpy(float)
        t5 = pd.to_numeric(frame["t5"], errors="coerce").to_numpy(float)
        state_arrays, _ = self._sum(frame["state"], STATES, "state_sum")
        arrays = [np.ones(n), age5, t5]
        names = ["Intercept", "age0_c5", "t5"]
        for state, q in zip(STATES[:-1], state_arrays):
            arrays.append(q); names.append(f"state_sum[{state}]")
        for state, q in zip(STATES[:-1], state_arrays):
            arrays.append(age5 * q); names.append(f"age0_c5:state_sum[{state}]")
        for state, q in zip(STATES[:-1], state_arrays):
            arrays.append(t5 * q); names.append(f"t5:state_sum[{state}]")
        for column, levels, prefix in [
            ("sex", self.sex_levels, "sex"),
            ("study", self.study_levels, "study"),
            ("site_scanner", self.acq_levels, "site_scanner"),
        ]:
            a, b = self._dummy(frame[column], levels, prefix)
            arrays.extend(a); names.extend(b)
        if len(self.site_levels) >= 2:
            site_arrays, _ = self._sum(frame["site"], self.site_levels, "site_sum")
            for site, sv in zip(self.site_levels[:-1], site_arrays):
                arrays.append(t5 * sv); names.append(f"t5:site_sum[{site}]")
                for state, q in zip(STATES[:-1], state_arrays):
                    arrays.append(t5 * sv * q)
                    names.append(f"t5:site_sum[{site}]:state_sum[{state}]")
        if self.age_moderation:
            arrays.append(age10 * t5); names.append("age0_c10:t5")
            for state, q in zip(STATES[:-1], state_arrays):
                arrays.append(age10 * t5 * q)
                names.append(f"age0_c10:t5:state_sum[{state}]")
        if self.amplitude_ratio is not None:
            for component in [f"{self.amplitude_ratio}_between", f"{self.amplitude_ratio}_within"]:
                value = pd.to_numeric(frame[component], errors="coerce").to_numpy(float)
                arrays.append(value); names.append(component)
                for state, q in zip(STATES[:-1], state_arrays):
                    arrays.append(value * q); names.append(f"{component}:state_sum[{state}]")
        return np.column_stack(arrays), names

    def transform(self, frame):
        raw, _ = self._raw(frame)
        return raw[:, self.keep]

    def _put(self, vector, name, value):
        if name in self.names:
            vector[self.names.index(name)] = value

    def state_slope_vector(self, state, age0_c10=0.0):
        c = np.zeros(len(self.names))
        self._put(c, "t5", 1.0)
        for coded in STATES[:-1]:
            weight = self._coded_weight(state, STATES, coded)
            self._put(c, f"t5:state_sum[{coded}]", weight)
        if self.age_moderation:
            self._put(c, "age0_c10:t5", age0_c10)
            for coded in STATES[:-1]:
                weight = self._coded_weight(state, STATES, coded)
                self._put(c, f"age0_c10:t5:state_sum[{coded}]", age0_c10 * weight)
        return c

    def state_slope_matrix(self, age0_c10=0.0):
        return np.vstack([self.state_slope_vector(state, age0_c10) for state in STATES])

    def age_moderation_matrix(self):
        if not self.age_moderation:
            return np.zeros((0, len(self.names)))
        rows = []
        for state in STATES:
            c = np.zeros(len(self.names))
            self._put(c, "age0_c10:t5", 1.0)
            for coded in STATES[:-1]:
                self._put(c, f"age0_c10:t5:state_sum[{coded}]", self._coded_weight(state, STATES, coded))
            rows.append(c)
        return np.vstack(rows)

    def site_time_matrix(self):
        names = [f"t5:site_sum[{site}]" for site in self.site_levels[:-1]]
        return self._selector(names)

    def site_time_state_matrix(self):
        names = [
            f"t5:site_sum[{site}]:state_sum[{state}]"
            for site in self.site_levels[:-1] for state in STATES[:-1]
        ]
        return self._selector(names)

    def _selector(self, names):
        rows = []
        for name in names:
            if name in self.names:
                row = np.zeros(len(self.names)); row[self.names.index(name)] = 1.0; rows.append(row)
        return np.vstack(rows) if rows else np.zeros((0, len(self.names)))

def fit_direct_profile_covariance(x, y, participant, state, pedigree):
    x = np.asarray(x, float); y = np.asarray(y, float)
    participant = np.asarray(participant).astype(str)
    state = np.asarray(state).astype(str); pedigree = np.asarray(pedigree).astype(str)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[valid], y[valid]
    participant, state, pedigree = participant[valid], state[valid], pedigree[valid]
    if len(y) <= x.shape[1] + 12: return None
    column_scale = np.sqrt(np.mean(x ** 2, axis=0))
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 1e-10), column_scale, 1.0)
    xs = x / column_scale
    blocks = []
    for pid in np.unique(participant):
        idx = np.flatnonzero(participant == pid)
        z_state = pd.get_dummies(pd.Series(state[idx]), dtype=float).to_numpy()
        kernels = [np.ones((len(idx), len(idx))), z_state @ z_state.T]
        blocks.append((idx, kernels, str(pedigree[idx[0]])))

    def profile(theta):
        tau = np.expm1(theta)
        xtx = np.zeros((xs.shape[1], xs.shape[1])); xty = np.zeros(xs.shape[1])
        yty, logdet_v = 0.0, 0.0
        for idx, kernels, _ in blocks:
            a = np.eye(len(idx))
            for value, kernel in zip(tau, kernels): a += value * kernel
            sign, ld = np.linalg.slogdet(a)
            if sign <= 0: return {"objective": np.inf}
            try:
                ax = np.linalg.solve(a, xs[idx]); ay = np.linalg.solve(a, y[idx])
            except np.linalg.LinAlgError:
                return {"objective": np.inf}
            xtx += xs[idx].T @ ax; xty += xs[idx].T @ ay
            yty += float(y[idx] @ ay); logdet_v += ld
        sign, logdet_xtx = np.linalg.slogdet(xtx)
        if sign <= 0: return {"objective": np.inf}
        beta = np.linalg.pinv(xtx, rcond=1e-11) @ xty
        q = float(yty - 2 * beta @ xty + beta @ xtx @ beta)
        df = len(y) - len(beta)
        if q <= 0 or df <= 0: return {"objective": np.inf}
        sigma_e2 = q / df
        return {"objective": .5 * (logdet_v + df * np.log(sigma_e2) + logdet_xtx),
                "coef": beta, "xtx": xtx, "sigma_e2": sigma_e2}

    starts = [np.log1p([.15, .15]), np.zeros(2), np.log1p([1.0, .05])]
    bounds = [(0, np.log1p(1e5))] * 2
    candidates = []
    for start in starts:
        opt = minimize(lambda theta: profile(theta)["objective"], start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 180, "ftol": 1e-9})
        if np.isfinite(opt.fun): candidates.append((opt.x, profile(opt.x), opt))
    if not candidates: return None
    theta, fitted, selected_opt = min(candidates, key=lambda item: item[1]["objective"])
    tau = np.expm1(theta); beta_s = fitted["coef"]
    residual = y - xs @ beta_s; bread_inv = np.linalg.pinv(fitted["xtx"], rcond=1e-11)
    pedigree_scores = {}
    for idx, kernels, pg in blocks:
        a = np.eye(len(idx))
        for value, kernel in zip(tau, kernels): a += value * kernel
        score = xs[idx].T @ np.linalg.solve(a, residual[idx])
        pedigree_scores[pg] = pedigree_scores.get(pg, np.zeros_like(score)) + score
    meat = sum(np.outer(v, v) for v in pedigree_scores.values())
    cov_s = bread_inv @ meat @ bread_inv
    g = len(pedigree_scores)
    correction = (g / max(g - 1, 1)) * ((len(y) - 1) / max(len(y) - len(beta_s), 1))
    cov_s *= correction
    influence_s = {pg: math.sqrt(correction) * (bread_inv @ score) for pg, score in pedigree_scores.items()}
    beta = beta_s / column_scale
    robust_cov = cov_s / np.outer(column_scale, column_scale)
    influences = {pg: value / column_scale for pg, value in influence_s.items()}
    return {
        "coef": beta, "robust_cov": robust_cov, "influences": influences,
        "sigma_e2": fitted["sigma_e2"], "tau_participant": float(tau[0]),
        "tau_participant_state": float(tau[1]), "objective": fitted["objective"],
        "row_n": len(y), "participant_n": len(np.unique(participant)), "pedigree_n": g,
        "rank": int(np.linalg.matrix_rank(fitted["xtx"])),
        "optimizer_success": bool(selected_opt.success), "optimizer_nfev": int(getattr(selected_opt, "nfev", 0)),
    }


def scalar_contrast(beta, cov, vector):
    estimate = float(vector @ beta); variance = float(vector @ cov @ vector)
    se = math.sqrt(max(0., variance)); z = estimate / se if se > 0 else np.nan
    return estimate, se, z, float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan


def wald_matrix(beta, cov, matrix):
    matrix = np.asarray(matrix, float)
    b = matrix @ np.asarray(beta); v = matrix @ np.asarray(cov) @ matrix.T
    rank = int(np.linalg.matrix_rank(v))
    if rank == 0: return np.nan, 0, np.nan
    stat = float(b @ np.linalg.pinv(v, rcond=1e-11) @ b)
    return stat, rank, float(chi2.sf(stat, rank))


def jsonable(value):
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.bool_): return bool(value)
    raise TypeError(type(value).__name__)


def fit_multivariate_profile(x, y_profile, participant, state, pedigree):
    x = np.asarray(x, float)
    y_profile = np.asarray(y_profile, float)
    participant = np.asarray(participant).astype(str)
    state = np.asarray(state).astype(str)
    pedigree = np.asarray(pedigree).astype(str)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y_profile).all(axis=1)
    x, y_profile = x[valid], y_profile[valid]
    participant, state, pedigree = participant[valid], state[valid], pedigree[valid]
    if len(y_profile) <= x.shape[1] + 12:
        return None

    q_basis = helmert(y_profile.shape[1], full=False).T
    z = y_profile @ q_basis
    q = z.shape[1]
    column_scale = np.sqrt(np.mean(x ** 2, axis=0))
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 1e-10), column_scale, 1.0)
    xs = x / column_scale
    blocks = []
    for pid in np.unique(participant):
        idx = np.flatnonzero(participant == pid)
        state_dummy = pd.get_dummies(pd.Series(state[idx]), dtype=float).to_numpy()
        blocks.append((idx, [np.ones((len(idx), len(idx))), state_dummy @ state_dummy.T], str(pedigree[idx[0]])))

    def profile(theta):
        tau = np.expm1(theta)
        xtx = np.zeros((xs.shape[1], xs.shape[1]))
        xtz = np.zeros((xs.shape[1], q))
        ztz = np.zeros((q, q))
        logdet_v = 0.0
        for idx, kernels, _ in blocks:
            a = np.eye(len(idx))
            for value, kernel in zip(tau, kernels):
                a += value * kernel
            sign, ld = np.linalg.slogdet(a)
            if sign <= 0:
                return {"objective": np.inf}
            try:
                ax = np.linalg.solve(a, xs[idx])
                az = np.linalg.solve(a, z[idx])
            except np.linalg.LinAlgError:
                return {"objective": np.inf}
            xtx += xs[idx].T @ ax
            xtz += xs[idx].T @ az
            ztz += z[idx].T @ az
            logdet_v += ld
        sign_x, logdet_x = np.linalg.slogdet(xtx)
        if sign_x <= 0:
            return {"objective": np.inf}
        beta = np.linalg.pinv(xtx, rcond=1e-11) @ xtz
        rss = ztz - xtz.T @ beta - beta.T @ xtz + beta.T @ xtx @ beta
        rss = (rss + rss.T) / 2
        df = len(z) - len(beta)
        if df <= q:
            return {"objective": np.inf}
        sigma = rss / df
        sign_s, logdet_s = np.linalg.slogdet(sigma)
        if sign_s <= 0 or not np.isfinite(logdet_s):
            return {"objective": np.inf}
        objective = .5 * (q * logdet_v + df * logdet_s + q * logdet_x)
        return {"objective": objective, "beta": beta, "xtx": xtx, "sigma": sigma}

    starts = [np.log1p([.15, .15]), np.zeros(2), np.log1p([1.0, .05])]
    bounds = [(0, np.log1p(1e5))] * 2
    candidates = []
    for start in starts:
        opt = minimize(lambda theta: profile(theta)["objective"], start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 180, "ftol": 1e-9})
        fitted = profile(opt.x)
        if np.isfinite(fitted.get("objective", np.inf)):
            candidates.append((opt.x, fitted, opt))
    if not candidates:
        return None
    theta, fitted, selected = min(candidates, key=lambda item: item[1]["objective"])
    tau = np.expm1(theta)
    beta_s = fitted["beta"]
    residual = z - xs @ beta_s
    bread = np.linalg.pinv(fitted["xtx"], rcond=1e-11)
    pedigree_scores = {}
    for idx, kernels, pg in blocks:
        a = np.eye(len(idx))
        for value, kernel in zip(tau, kernels):
            a += value * kernel
        score = xs[idx].T @ np.linalg.solve(a, residual[idx])
        pedigree_scores[pg] = pedigree_scores.get(pg, np.zeros_like(score)) + score
    g = len(pedigree_scores)
    correction = (g / max(g - 1, 1)) * ((len(z) - 1) / max(len(z) - beta_s.shape[0], 1))
    influence = {pg: math.sqrt(correction) * (bread @ score) for pg, score in pedigree_scores.items()}
    beta = beta_s / column_scale[:, None]
    influence = {pg: value / column_scale[:, None] for pg, value in influence.items()}
    beta_vec = beta.ravel(order="F")
    robust_cov = sum(np.outer(value.ravel(order="F"), value.ravel(order="F")) for value in influence.values())
    return {
        "coef": beta_vec, "robust_cov": robust_cov, "basis": q_basis,
        "tau_participant": float(tau[0]), "tau_participant_state": float(tau[1]),
        "sigma_coordinate": fitted["sigma"], "objective": fitted["objective"],
        "row_n": len(z), "participant_n": len(np.unique(participant)), "pedigree_n": g,
        "design_rank": int(np.linalg.matrix_rank(fitted["xtx"])),
        "optimizer_success": bool(selected.success), "optimizer_nfev": int(getattr(selected, "nfev", 0)),
    }


def block_expand(matrix, coordinate_n):
    matrix = np.asarray(matrix, float)
    if matrix.shape[0] == 0:
        return np.zeros((0, matrix.shape[1] * coordinate_n))
    rows = []
    for coordinate in range(coordinate_n):
        unit = np.zeros(coordinate_n); unit[coordinate] = 1.0
        rows.extend(np.kron(unit, row) for row in matrix)
    return np.vstack(rows)



# Reconcile the primary fMRI outcome with script 03


if FP_NETWORK not in ELIGIBLE_NETWORKS or len(ELIGIBLE_NETWORKS) != 10:
    raise AssertionError(
        f"Expected ten eligible networks including FP; found {ELIGIBLE_NETWORKS}"
    )


def add_fp_relative(frame, suffix="_STD"):
    out = frame.copy()
    fp = pd.to_numeric(
        out[f"L{suffix}__{FP_NETWORK}"], errors="coerce"
    ).astype(np.float64)
    references = [
        f"L{suffix}__{network}"
        for network in ELIGIBLE_NETWORKS if network != FP_NETWORK
    ]
    out["Y_FP_REL"] = fp - out[references].to_numpy(dtype=np.float64).mean(axis=1)
    out["Y_FP_REL_FROM_PROFILE"] = (
        len(ELIGIBLE_NETWORKS) / (len(ELIGIBLE_NETWORKS) - 1.0)
    ) * pd.to_numeric(
        out[f"R{suffix}__{FP_NETWORK}"], errors="coerce"
    ).astype(np.float64)
    return out


amplitude_fp_long = add_fp_relative(FRAME_REGISTRY["AMPLITUDE_STANDARDIZED"])
identity_error = amplitude_fp_long["Y_FP_REL"] - amplitude_fp_long["Y_FP_REL_FROM_PROFILE"]
fmri_identity = pd.DataFrame([{
    "identity": "Y_FP_REL_STD = L_STD_FP - mean(other nine L_STD) = (10/9) R_STD_FP",
    "participants": int(amplitude_fp_long["id"].nunique()),
    "sessions": int(amplitude_fp_long["id_event"].nunique()),
    "session_state_rows": int(len(amplitude_fp_long)),
    "max_absolute_error": float(np.nanmax(np.abs(identity_error))),
    "tolerance": 1e-10,
    "pass": bool(np.nanmax(np.abs(identity_error)) <= 1e-10),
}])
if not fmri_identity["pass"].all():
    raise AssertionError("FP-relative fMRI identity failed")
if not FAST_MODE and amplitude_fp_long["id"].nunique() != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Expected {EXPECTED_REPEAT_PARTICIPANTS} repeat participants; "
        f"found {amplitude_fp_long['id'].nunique()}"
    )


def preceding_scalar_fingerprint(frame, spec):
    columns = [
        "id", "id_event", "state", "pedigree_group", "age0_c5", "t5",
        "site", "site_scanner", "sex", "study", "Y_FP_REL",
    ]
    payload = pd.util.hash_pandas_object(
        frame[columns].reset_index(drop=True), index=True
    ).to_numpy().tobytes()
    return hashlib.sha256(
        payload + json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()


source_manifest_path = SOURCE_FP_OUTPUT_DIR / "run_manifest.json"
source_checkpoint_path = SOURCE_FP_CHECKPOINT_DIR / "scalar__amplitude_standardized.json"
if not source_manifest_path.is_file() or not source_checkpoint_path.is_file():
    raise FileNotFoundError(
        "Script 03 outputs are required. Expected "
        f"{source_manifest_path} and {source_checkpoint_path}."
    )
source_fp_manifest = json.loads(source_manifest_path.read_text())
source_fp_checkpoint = json.loads(source_checkpoint_path.read_text())
preceding_spec = {
    "scenario": "AMPLITUDE_STANDARDIZED",
    "representation": "AMPLITUDE_STANDARDIZED",
    "role": "PRINCIPAL_FIXED_AMPLITUDE_STANDARDIZED_COVARIANCE",
    "site_left_out": None,
    "export_site_state": True,
}
current_fingerprint = preceding_scalar_fingerprint(amplitude_fp_long, preceding_spec)
fmri_source_reconciliation = pd.DataFrame([{
    "source_script": source_fp_manifest.get("script"),
    "saved_participants": source_fp_manifest.get("participants"),
    "saved_network_n": len(source_fp_manifest.get("eligible_networks", [])),
    "saved_specificity_status": source_fp_manifest.get("specificity_status"),
    "fingerprint_match": source_fp_checkpoint.get("fingerprint") == current_fingerprint,
    "eligible_network_order_match": (
        source_fp_manifest.get("eligible_networks") == ELIGIBLE_NETWORKS
    ),
}])
if not fmri_source_reconciliation[
    ["fingerprint_match", "eligible_network_order_match"]
].all(axis=None):
    print(fmri_source_reconciliation)
    raise AssertionError(
        "Amplitude-standardized fMRI rows do not match script 03 outputs"
    )
if not FAST_MODE and int(source_fp_manifest.get("participants", -1)) != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError("Script 03 participant count does not match the expected cohort")

display(fmri_identity)
display(fmri_source_reconciliation)



# Construct network-level structural measures

STRUCTURAL_ZIP = resolve_unique(["AABC_Release2_StructuralIDPs.zip"], label="AABC2 structural IDP ZIP")
IDP_INFO_ZIP = resolve_unique(["AABC_Release2_IDP_info.zip"], label="AABC2 IDP-info ZIP")


def zip_member_info(zip_path, basename):
    with ZipFile(zip_path) as archive:
        hits = [item for item in archive.infolist()
                if not item.filename.startswith("__MACOSX/") and Path(item.filename).name == basename]
    if len(hits) != 1:
        raise FileNotFoundError(f"{zip_path.name}: expected one {basename}, found {len(hits)}")
    return hits[0]


def read_zip_csv(zip_path, basename):
    member = zip_member_info(zip_path, basename)
    with ZipFile(zip_path) as archive, archive.open(member.filename) as stream:
        frame = pd.read_csv(stream, low_memory=False)
    return frame, member


roi_map, roi_member = read_zip_csv(IDP_INFO_ZIP, "AreaNamesAndIndicesAndNetworks.csv")
thickness_raw, thickness_member = read_zip_csv(STRUCTURAL_ZIP, "Cortical_Areal_Thicknesses.csv")
area_raw, area_member = read_zip_csv(STRUCTURAL_ZIP, "Cortical_Areal_Surface_Areas.csv")
volume_raw, volume_member = read_zip_csv(STRUCTURAL_ZIP, "Cortical_Areal_Volumes.csv")
aseg_raw, aseg_member = read_zip_csv(STRUCTURAL_ZIP, "asegstats.csv")

structural_source_audit = pd.DataFrame([
    {"role": "structural_zip", "path": str(STRUCTURAL_ZIP), "member": member.filename,
     "member_crc32": f"{member.CRC:08x}", "member_bytes": member.file_size}
    for member in [thickness_member, area_member, volume_member, aseg_member]
] + [{
    "role": "roi_network_map", "path": str(IDP_INFO_ZIP), "member": roi_member.filename,
    "member_crc32": f"{roi_member.CRC:08x}", "member_bytes": roi_member.file_size,
}])


def load_structural_table(frame, strip_mr=False):
    id_column = identify_id_column(frame.columns)
    ids = frame[id_column].astype(str).str.strip()
    if strip_mr:
        ids = ids.str.replace(r"_MR$", "", regex=True)
    numeric = frame.drop(columns=[id_column]).apply(pd.to_numeric, errors="coerce")
    numeric.index = ids
    numeric.index.name = "id_event"
    if numeric.index.duplicated().any():
        raise AssertionError(f"Duplicate structural id_event rows: {numeric.index[numeric.index.duplicated()].tolist()[:5]}")
    return numeric


thickness = load_structural_table(thickness_raw)
surface_area = load_structural_table(area_raw)
volume = load_structural_table(volume_raw)
aseg = load_structural_table(aseg_raw, strip_mr=True)

area_column = pick_column(roi_map, ["Brain Area", "brain_area", "parcel", "area"], required=True)
label_column = pick_column(roi_map, ["Ji Network Label", "ji_network_label"], required=True)
index_column = pick_column(roi_map, ["Real Index", "real_index", "index"], required=True)
official_cortex = roi_map[[area_column, label_column, index_column]].rename(columns={
    area_column: "Brain Area", label_column: "Ji Network Label", index_column: "Real Index",
}).copy()
official_cortex["Ji Network Label"] = pd.to_numeric(official_cortex["Ji Network Label"], errors="coerce")
official_cortex["Real Index"] = pd.to_numeric(official_cortex["Real Index"], errors="coerce")
official_cortex = official_cortex.loc[official_cortex["Ji Network Label"].notna()].sort_values("Real Index")
if not FAST_MODE and len(official_cortex) != 360:
    raise AssertionError(f"Expected 360 cortical parcels in the official map; found {len(official_cortex)}")

label_to_network = {
    1: "Visual1", 2: "Visual2", 3: "Somatomotor", 4: "Cingulo-Opercular",
    5: "Dorsal-attention", 6: "Language", 7: "Frontoparietal", 8: "Auditory",
    9: "Default", 10: "Posterior-Multimodal", 11: "Ventral-Multimodal", 12: "Orbito-Affective",
}
official_cortex["official_network"] = official_cortex["Ji Network Label"].map(label_to_network)
official_cortex["area_token"] = official_cortex["Brain Area"].map(compact_token)
canonical_network_tokens = {compact_token(network) for network in label_to_network.values()}
selected_mapping = parcel_map.loc[
    parcel_map["node"].map(compact_token).isin(canonical_network_tokens),
    ["Brain Area", "node", "Real Index"],
].copy()
selected_mapping["area_token"] = selected_mapping["Brain Area"].map(compact_token)
mapping_compare = selected_mapping.merge(
    official_cortex[["area_token", "official_network"]], on="area_token", how="left", validate="one_to_one"
)
mapping_compare["mapping_match"] = mapping_compare["node"].map(compact_token).eq(
    mapping_compare["official_network"].map(compact_token)
)
network_mapping_audit = pd.DataFrame([{
    "selected_parcels": int(len(selected_mapping)),
    "official_cortical_parcels": int(len(official_cortex)),
    "official_matches": int(mapping_compare["official_network"].notna().sum()),
    "network_mismatches": int((~mapping_compare["mapping_match"]).sum()),
    "same_mapping_pass": bool(mapping_compare["mapping_match"].all()),
}])
if not network_mapping_audit.loc[0, "same_mapping_pass"]:
    display(mapping_compare.loc[~mapping_compare["mapping_match"]].head(20))
    raise AssertionError("Official structural and repeat-cohort fMRI parcel/network mappings differ")


def column_lookup(frame):
    lookup = {}
    for column in frame.columns:
        token = compact_token(column)
        if token in lookup:
            raise AssertionError(f"Duplicate normalized structural columns: {lookup[token]} and {column}")
        lookup[token] = column
    return lookup


thickness_lookup = column_lookup(thickness)
area_lookup = column_lookup(surface_area)
volume_lookup = column_lookup(volume)
network_areas = {
    network: parcel_map.loc[parcel_map["node_token"].eq(compact_token(network)), "Brain Area"].astype(str).tolist()
    for network in ELIGIBLE_NETWORKS
}
missing_structural = []
for network, areas in network_areas.items():
    for area_name in areas:
        token = compact_token(area_name)
        for modality, lookup in [("thickness", thickness_lookup), ("surface_area", area_lookup), ("volume", volume_lookup)]:
            if token not in lookup:
                missing_structural.append({"network": network, "parcel": area_name, "modality": modality})
if missing_structural:
    display(pd.DataFrame(missing_structural).head(30))
    raise KeyError(f"Missing {len(missing_structural)} selected structural parcel fields")

icv_column = next((column for column in aseg if compact_token(column) == compact_token("EstimatedTotalIntraCranialVol")), None)
if icv_column is None:
    raise KeyError("asegstats.csv lacks EstimatedTotalIntraCranialVol")
icv = pd.to_numeric(aseg[icv_column], errors="coerce").where(lambda value: value > 0)

all_ids = sorted(set(thickness.index) | set(surface_area.index) | set(volume.index) | set(aseg.index))
structural_features = pd.DataFrame(index=pd.Index(all_ids, name="id_event"))
coverage_rows = []
for network, areas in network_areas.items():
    t_columns = [thickness_lookup[compact_token(area_name)] for area_name in areas]
    a_columns = [area_lookup[compact_token(area_name)] for area_name in areas]
    v_columns = [volume_lookup[compact_token(area_name)] for area_name in areas]
    t = thickness[t_columns].copy(); t.columns = areas
    a = surface_area[a_columns].copy(); a.columns = areas
    v = volume[v_columns].copy(); v.columns = areas
    t = t.where(t > 0); a = a.where(a > 0); v = v.where(v > 0)
    valid_thickness = t.notna() & a.notna()
    fraction = valid_thickness.sum(axis=1) / len(areas)
    thickness_mean = (
        (t.where(valid_thickness) * a.where(valid_thickness)).sum(axis=1, min_count=1)
        / a.where(valid_thickness).sum(axis=1, min_count=1)
    ).where(fraction >= STRUCTURAL_MIN_PARCEL_FRACTION)
    volume_sum = v.sum(axis=1, min_count=len(areas))
    structural_features[f"thickness__{network}"] = thickness_mean.reindex(structural_features.index)
    structural_features[f"volume_sum__{network}"] = volume_sum.reindex(structural_features.index)
    structural_features[f"volume_log_ratio_icv__{network}"] = np.log(
        (volume_sum / icv).where((volume_sum > 0) & (icv > 0))
    ).reindex(structural_features.index)
    coverage_rows.extend([
        {"network": network, "modality": "THICKNESS_AREA_WEIGHTED", "expected_parcels": len(areas),
         "minimum_fraction": STRUCTURAL_MIN_PARCEL_FRACTION, "eligible_sessions": int(thickness_mean.notna().sum()),
         "median_valid_fraction": float(fraction.median())},
        {"network": network, "modality": "VOLUME_EXACT_SUM", "expected_parcels": len(areas),
         "minimum_fraction": 1.0, "eligible_sessions": int(volume_sum.notna().sum()),
         "median_valid_fraction": float(v.notna().mean(axis=1).median())},
    ])

structural_features["log_icv"] = np.log(icv).reindex(structural_features.index)
reference_networks = [network for network in ELIGIBLE_NETWORKS if network != FP_NETWORK]
structural_features["Y_THICK_REL"] = (
    structural_features[f"thickness__{FP_NETWORK}"]
    - structural_features[[f"thickness__{network}" for network in reference_networks]].mean(axis=1, skipna=False)
)
structural_features["Y_VOLUME_REL"] = (
    structural_features[f"volume_log_ratio_icv__{FP_NETWORK}"]
    - structural_features[[f"volume_log_ratio_icv__{network}" for network in reference_networks]].mean(axis=1, skipna=False)
)
unscaled_volume_relative = (
    np.log(structural_features[f"volume_sum__{FP_NETWORK}"])
    - np.log(structural_features[[f"volume_sum__{network}" for network in reference_networks]]).mean(axis=1, skipna=False)
)
icv_cancellation_error = structural_features["Y_VOLUME_REL"] - unscaled_volume_relative

other_areas = [area_name for network in reference_networks for area_name in network_areas[network]]
t_other = thickness[[thickness_lookup[compact_token(area_name)] for area_name in other_areas]].copy()
t_other.columns = other_areas
a_other = surface_area[[area_lookup[compact_token(area_name)] for area_name in other_areas]].copy()
a_other.columns = other_areas
valid_other = t_other.gt(0) & a_other.gt(0)
pooled_other_thickness = (
    (t_other.where(valid_other) * a_other.where(valid_other)).sum(axis=1, min_count=1)
    / a_other.where(valid_other).sum(axis=1, min_count=1)
).where(valid_other.mean(axis=1) >= STRUCTURAL_MIN_PARCEL_FRACTION)
structural_features["Y_THICK_REL_PARCEL_POOLED_DIAGNOSTIC"] = (
    structural_features[f"thickness__{FP_NETWORK}"] - pooled_other_thickness.reindex(structural_features.index)
)

parcel_counts_selected = {network: len(areas) for network, areas in network_areas.items()}
reference_weight_audit = pd.DataFrame([
    {"network": network, "parcel_n": parcel_counts_selected[network],
     "primary_reference_weight": 0.0 if network == FP_NETWORK else 1 / len(reference_networks),
     "parcel_count_reference_weight_diagnostic": (
         0.0 if network == FP_NETWORK else parcel_counts_selected[network] / sum(parcel_counts_selected[x] for x in reference_networks)
     ),
     "primary_reference_is_equal_network_weighted": True}
    for network in ELIGIBLE_NETWORKS
])
parcel_coverage_audit = pd.DataFrame(coverage_rows)
reference_construction_audit = pd.DataFrame([{
    "eligible_network_n": len(ELIGIBLE_NETWORKS), "reference_network_n": len(reference_networks),
    "fp_parcel_n": parcel_counts_selected[FP_NETWORK],
    "reference_parcel_n_min": min(parcel_counts_selected[x] for x in reference_networks),
    "reference_parcel_n_max": max(parcel_counts_selected[x] for x in reference_networks),
    "unequal_parcel_counts_used_as_primary_weights": False,
    "thickness_equal_network_vs_parcel_pooled_correlation": float(structural_features[
        ["Y_THICK_REL", "Y_THICK_REL_PARCEL_POOLED_DIAGNOSTIC"]
    ].corr().iloc[0, 1]),
    "volume_icv_log_ratio_cancellation_max_error": float(np.nanmax(np.abs(icv_cancellation_error))),
}])

event_meta = amplitude_fp_long[[
    "id_event", "id", "pedigree_group", "study", "sex", "site", "scanner", "site_scanner",
    "event_age", "baseline_age", "age0_c5", "age0_c10", "time_years", "t5", "visit_number",
]].drop_duplicates("id_event")
structural_visits = event_meta.merge(
    structural_features.reset_index(), on="id_event", how="left", validate="one_to_one"
)

display(structural_source_audit)
display(network_mapping_audit)
display(parcel_coverage_audit)
display(reference_weight_audit)
display(reference_construction_audit)



# Direct longitudinal structural models

def retain_two_structural_visits(frame, outcome):
    required = [outcome, "age0_c5", "t5"]
    out = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(axis=1)].copy()
    counts = out.groupby("id")["id_event"].nunique()
    return out.loc[out["id"].isin(counts[counts.ge(2)].index)].copy()


thickness_visits = retain_two_structural_visits(structural_visits, "Y_THICK_REL")
volume_visits = retain_two_structural_visits(structural_visits, "Y_VOLUME_REL")
for frame in [thickness_visits, volume_visits]:
    frame["log_icv_between"] = frame.groupby("id")["log_icv"].transform("mean")
volume_icv_sd = float(volume_visits["log_icv_between"].std(ddof=0))
if not np.isfinite(volume_icv_sd) or volume_icv_sd <= 0:
    raise AssertionError("Participant-mean log ICV has no usable variation")
volume_visits["log_icv_between_z"] = (
    volume_visits["log_icv_between"] - volume_visits["log_icv_between"].mean()
) / volume_icv_sd

structural_cohort_summary = pd.DataFrame([
    {"cohort": "REPEAT_FMRI_COHORT", "participants": amplitude_fp_long["id"].nunique(),
     "sessions": amplitude_fp_long["id_event"].nunique()},
    {"cohort": "THICKNESS_TWO_PLUS_MATCHED", "participants": thickness_visits["id"].nunique(),
     "sessions": thickness_visits["id_event"].nunique()},
    {"cohort": "VOLUME_TWO_PLUS_MATCHED", "participants": volume_visits["id"].nunique(),
     "sessions": volume_visits["id_event"].nunique()},
])


class StructuralLongitudinalDesign:
    @staticmethod
    def _dummy(series, levels, prefix):
        text = series.astype("string").fillna("MISSING")
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(text.eq(level).to_numpy(float)); names.append(f"{prefix}[{level}]")
        return arrays, names

    @staticmethod
    def _sum(series, levels, prefix):
        text = series.astype(str)
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(np.where(text == level, 1.0, np.where(text == levels[-1], -1.0, 0.0)))
            names.append(f"{prefix}[{level}]")
        return arrays, names

    @staticmethod
    def _stable_rank(matrix):
        matrix = np.asarray(matrix, float)
        if not np.isfinite(matrix).all(): return -1
        scale = np.sqrt(np.mean(matrix ** 2, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        _, r = qr(matrix / scale, mode="economic", pivoting=False, check_finite=False)
        diagonal = np.abs(np.diag(r))
        tol = max(matrix.shape) * np.finfo(float).eps * max(diagonal.max(initial=0), 1.0)
        return int(np.sum(diagonal > tol))

    @staticmethod
    def _coded_weight(level, levels, coded):
        if level == coded: return 1.0
        if level == levels[-1]: return -1.0
        return 0.0

    def __init__(self, extra_columns=()):
        self.extra_columns = list(extra_columns)

    def _raw(self, frame):
        n = len(frame)
        age = pd.to_numeric(frame["age0_c5"], errors="coerce").to_numpy(float)
        time = pd.to_numeric(frame["t5"], errors="coerce").to_numpy(float)
        arrays, names = [np.ones(n), age, time], ["Intercept", "age0_c5", "t5"]
        for column, levels, prefix in [
            ("sex", self.sex_levels, "sex"), ("study", self.study_levels, "study"),
            ("site_scanner", self.acq_levels, "site_scanner"),
        ]:
            a, b = self._dummy(frame[column], levels, prefix); arrays.extend(a); names.extend(b)
        if len(self.site_levels) >= 2:
            site_arrays, _ = self._sum(frame["site"], self.site_levels, "site_sum")
            for site, value in zip(self.site_levels[:-1], site_arrays):
                arrays.append(time * value); names.append(f"t5:site_sum[{site}]")
        for column in self.extra_columns:
            arrays.append(pd.to_numeric(frame[column], errors="coerce").to_numpy(float)); names.append(column)
        return np.column_stack(arrays), names

    def fit(self, frame):
        self.sex_levels = sorted(frame["sex"].astype("string").fillna("MISSING").unique())
        self.study_levels = sorted(frame["study"].astype("string").fillna("MISSING").unique())
        self.acq_levels = sorted(frame["site_scanner"].astype("string").fillna("MISSING").unique())
        self.site_levels = sorted(frame["site"].astype("string").fillna("MISSING").unique())
        raw, names = self._raw(frame)
        mandatory = ["Intercept", "age0_c5", "t5", *self.extra_columns]
        mandatory += [f"t5:site_sum[{site}]" for site in self.site_levels[:-1]]
        mandatory_idx = [names.index(name) for name in mandatory]
        keep, rank = [], 0
        for j in mandatory_idx + [k for k in range(raw.shape[1]) if k not in mandatory_idx]:
            new_rank = self._stable_rank(raw[:, keep + [j]])
            if new_rank > rank:
                keep.append(j); rank = new_rank
        self.keep = keep; self.names = [names[j] for j in keep]
        missing = [name for name in mandatory if name not in self.names]
        if missing: raise AssertionError(f"Structural model lost mandatory terms: {missing}")
        return self

    def transform(self, frame):
        raw, _ = self._raw(frame); return raw[:, self.keep]

    def _put(self, vector, name, value):
        if name in self.names: vector[self.names.index(name)] = value

    def time_vector(self):
        vector = np.zeros(len(self.names)); self._put(vector, "t5", 1.0); return vector

    def site_time_matrix(self):
        rows = []
        for site in self.site_levels[:-1]:
            vector = np.zeros(len(self.names)); self._put(vector, f"t5:site_sum[{site}]", 1.0); rows.append(vector)
        return np.vstack(rows) if rows else np.zeros((0, len(self.names)))

    def site_slope_vector(self, site):
        vector = self.time_vector()
        for coded in self.site_levels[:-1]:
            self._put(vector, f"t5:site_sum[{coded}]", self._coded_weight(site, self.site_levels, coded))
        return vector


def fit_random_intercept_pedigree_sandwich(x, y, participant, pedigree):
    x = np.asarray(x, float); y = np.asarray(y, float)
    participant = np.asarray(participant).astype(str); pedigree = np.asarray(pedigree).astype(str)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y, participant, pedigree = x[valid], y[valid], participant[valid], pedigree[valid]
    if len(y) <= x.shape[1] + 12: return None
    column_scale = np.sqrt(np.mean(x ** 2, axis=0))
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 1e-10), column_scale, 1.0)
    xs = x / column_scale
    blocks = []
    for pid in np.unique(participant):
        index = np.flatnonzero(participant == pid)
        blocks.append((index, np.ones((len(index), len(index))), str(pedigree[index[0]])))

    def profile(theta):
        tau = float(np.expm1(theta[0])); xtx = np.zeros((xs.shape[1], xs.shape[1])); xty = np.zeros(xs.shape[1])
        yty = 0.0; logdet_v = 0.0
        for index, kernel, _ in blocks:
            matrix = np.eye(len(index)) + tau * kernel
            sign, logdet = np.linalg.slogdet(matrix)
            if sign <= 0: return {"objective": np.inf}
            try:
                ax = np.linalg.solve(matrix, xs[index]); ay = np.linalg.solve(matrix, y[index])
            except np.linalg.LinAlgError:
                return {"objective": np.inf}
            xtx += xs[index].T @ ax; xty += xs[index].T @ ay
            yty += float(y[index] @ ay); logdet_v += logdet
        sign, logdet_x = np.linalg.slogdet(xtx)
        if sign <= 0: return {"objective": np.inf}
        beta = np.linalg.pinv(xtx, rcond=1e-11) @ xty
        rss = float(yty - 2 * beta @ xty + beta @ xtx @ beta); df = len(y) - len(beta)
        if rss <= 0 or df <= 0: return {"objective": np.inf}
        sigma_e2 = rss / df
        return {"objective": .5 * (logdet_v + df * np.log(sigma_e2) + logdet_x),
                "coef": beta, "xtx": xtx, "sigma_e2": sigma_e2}

    candidates = []
    for start in [0.0, np.log1p(.2), np.log1p(1.0), np.log1p(5.0)]:
        optimization = minimize(lambda theta: profile(theta)["objective"], np.array([start]), method="L-BFGS-B",
                                bounds=[(0, np.log1p(1e5))], options={"maxiter": 200, "ftol": 1e-10})
        fitted = profile(optimization.x)
        if np.isfinite(fitted.get("objective", np.inf)): candidates.append((optimization.x, fitted, optimization))
    if not candidates: return None
    theta, fitted, selected = min(candidates, key=lambda item: item[1]["objective"])
    tau = float(np.expm1(theta[0])); beta_scaled = fitted["coef"]
    residual = y - xs @ beta_scaled; bread = np.linalg.pinv(fitted["xtx"], rcond=1e-11)
    pedigree_scores = {}
    for index, kernel, family in blocks:
        matrix = np.eye(len(index)) + tau * kernel
        score = xs[index].T @ np.linalg.solve(matrix, residual[index])
        pedigree_scores[family] = pedigree_scores.get(family, np.zeros_like(score)) + score
    family_n = len(pedigree_scores)
    correction = (family_n / max(family_n - 1, 1)) * ((len(y) - 1) / max(len(y) - len(beta_scaled), 1))
    meat = sum(np.outer(score, score) for score in pedigree_scores.values())
    covariance_scaled = correction * bread @ meat @ bread
    beta = beta_scaled / column_scale; covariance = covariance_scaled / np.outer(column_scale, column_scale)
    return {
        "coef": beta, "robust_cov": covariance, "sigma_e2": fitted["sigma_e2"], "tau_participant": tau,
        "objective": fitted["objective"], "row_n": len(y), "participant_n": len(np.unique(participant)),
        "pedigree_n": family_n, "rank": int(np.linalg.matrix_rank(fitted["xtx"])),
        "optimizer_success": bool(selected.success), "optimizer_nfev": int(getattr(selected, "nfev", 0)),
    }


def fit_structural_model(frame, outcome, scenario, extra_columns=()):
    fingerprint_columns = [
        "id", "id_event", "pedigree_group", "age0_c5", "t5", "sex", "study", "site", "site_scanner",
        outcome, *extra_columns,
    ]
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame[fingerprint_columns].reset_index(drop=True), index=True).to_numpy().tobytes()
        + scenario.encode()
    ).hexdigest()
    checkpoint = CHECKPOINT_DIR / f"structural__{scenario.lower()}.json"
    if checkpoint.exists() and not FORCE_REFIT:
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") == fingerprint: return saved["result"]
    design = StructuralLongitudinalDesign(extra_columns).fit(frame)
    x = design.transform(frame)
    fit = fit_random_intercept_pedigree_sandwich(x, frame[outcome], frame["id"], frame["pedigree_group"])
    if fit is None: raise RuntimeError(f"Structural model {scenario} failed")
    beta, covariance = fit["coef"], fit["robust_cov"]
    time_vector = design.time_vector()
    slope, slope_se, slope_z, slope_p = scalar_contrast(beta, covariance, time_vector)
    site_stat, site_df, site_p = wald_matrix(beta, covariance, design.site_time_matrix())
    q_site = norm.ppf(1 - .05 / (2 * max(1, len(design.site_levels))))
    site_rows = []
    for site in design.site_levels:
        vector = design.site_slope_vector(site)
        estimate, se, z, p = scalar_contrast(beta, covariance, vector)
        site_rows.append({"scenario": scenario, "site": site, "slope_per_5y": estimate, "se": se,
                          "z": z, "p": p, "simultaneous_ci_low": estimate - q_site * se,
                          "simultaneous_ci_high": estimate + q_site * se})
    result = {
        "summary": {"scenario": scenario, "outcome": outcome, "fit_status": "PASS", "rows": fit["row_n"],
                    "participants": fit["participant_n"], "pedigrees": fit["pedigree_n"],
                    "design_columns": len(design.names), "design_rank": fit["rank"],
                    "condition_number": float(np.linalg.cond(x)), "optimizer_success": fit["optimizer_success"],
                    "optimizer_nfev": fit["optimizer_nfev"], "objective": fit["objective"],
                    "sigma_e2": fit["sigma_e2"], "tau_participant": fit["tau_participant"],
                    "robust_cov_min_eigenvalue": float(np.linalg.eigvalsh((covariance + covariance.T) / 2).min())},
        "tests": [{"scenario": scenario, "test": "time_zero", "chi2": slope_z ** 2, "df": 1, "p": slope_p},
                  {"scenario": scenario, "test": "site_x_time", "chi2": site_stat, "df": site_df, "p": site_p}],
        "slope": [{"scenario": scenario, "outcome": outcome, "slope_per_5y": slope, "se": slope_se,
                   "z": slope_z, "p": slope_p, "ci_low": slope - norm.ppf(.975) * slope_se,
                   "ci_high": slope + norm.ppf(.975) * slope_se}],
        "site": site_rows,
    }
    checkpoint.write_text(json.dumps({"fingerprint": fingerprint, "result": result}, default=jsonable, indent=2) + "\n")
    return result


structural_results = [
    fit_structural_model(thickness_visits, "Y_THICK_REL", "THICKNESS_PRIMARY"),
    fit_structural_model(volume_visits, "Y_VOLUME_REL", "VOLUME_ICV_SECONDARY", ["log_icv_between_z"]),
]
structural_model_summary = pd.DataFrame([result["summary"] for result in structural_results])
structural_tests = pd.DataFrame([row for result in structural_results for row in result["tests"]])
structural_slopes = pd.DataFrame([row for result in structural_results for row in result["slope"]])
structural_site_slopes = pd.DataFrame([row for result in structural_results for row in result["site"]])

display(structural_cohort_summary)
display(structural_model_summary)
display(structural_tests)
display(structural_slopes)
display(structural_site_slopes)



# Structure-adjusted fMRI sensitivity models

def structural_components(frame, outcome, prefix):
    out = frame[["id", "id_event", outcome]].copy()
    out[f"{prefix}_between_raw"] = out.groupby("id")[outcome].transform("mean")
    out[f"{prefix}_within_raw"] = out[outcome] - out[f"{prefix}_between_raw"]
    participant_means = out[["id", f"{prefix}_between_raw"]].drop_duplicates("id")
    between_mean = float(participant_means[f"{prefix}_between_raw"].mean())
    between_sd = float(participant_means[f"{prefix}_between_raw"].std(ddof=0))
    within_sd = float(out[f"{prefix}_within_raw"].std(ddof=0))
    if not np.isfinite(between_sd) or between_sd <= 0 or not np.isfinite(within_sd) or within_sd <= 0:
        raise AssertionError(f"{prefix} structural components have no usable variation")
    out[f"{prefix}_between_z"] = (out[f"{prefix}_between_raw"] - between_mean) / between_sd
    out[f"{prefix}_within_z"] = out[f"{prefix}_within_raw"] / within_sd
    audit = {
        "structure": prefix, "outcome": outcome, "participants": out["id"].nunique(), "visits": len(out),
        "between_center": between_mean, "between_sd": between_sd, "within_sd": within_sd,
        "max_reconstruction_error": float(np.max(np.abs(
            out[outcome] - out[f"{prefix}_between_raw"] - out[f"{prefix}_within_raw"]
        ))),
        "max_participant_within_mean_error": float(np.max(np.abs(
            out.groupby("id")[f"{prefix}_within_raw"].mean()
        ))),
    }
    return out, audit


thickness_components, thickness_component_audit = structural_components(
    thickness_visits, "Y_THICK_REL", "thickness"
)
volume_components, volume_component_audit = structural_components(volume_visits, "Y_VOLUME_REL", "volume")
structural_component_audit = pd.DataFrame([thickness_component_audit, volume_component_audit])
if structural_component_audit[["max_reconstruction_error", "max_participant_within_mean_error"]].to_numpy().max() > 1e-10:
    raise AssertionError("Structural within-/between-person decomposition failed")


def merge_structure_with_fmri(components, prefix):
    columns = ["id_event", f"{prefix}_between_raw", f"{prefix}_within_raw",
               f"{prefix}_between_z", f"{prefix}_within_z"]
    out = amplitude_fp_long.merge(components[columns], on="id_event", how="inner", validate="many_to_one")
    required = ["Y_FP_REL", f"{prefix}_between_z", f"{prefix}_within_z", "age0_c5", "t5"]
    out = out.loc[np.isfinite(out[required].to_numpy(float)).all(axis=1)].copy()
    return retain_two_sessions(out).sort_values(["id", "time_years", "id_event", "state"]).reset_index(drop=True)


thickness_fmri = merge_structure_with_fmri(thickness_components, "thickness")
volume_fmri = merge_structure_with_fmri(volume_components, "volume")
fmri_matched_cohort = pd.DataFrame([
    {"sample": "THICKNESS_MATCHED", "participants": thickness_fmri["id"].nunique(),
     "sessions": thickness_fmri["id_event"].nunique(), "state_rows": len(thickness_fmri)},
    {"sample": "VOLUME_MATCHED", "participants": volume_fmri["id"].nunique(),
     "sessions": volume_fmri["id_event"].nunique(), "state_rows": len(volume_fmri)},
])


class StructuralAdjustedStateDesign(DirectStateDesign):
    def __init__(self, structural_components=()):
        super().__init__(age_moderation=False, amplitude_ratio=None)
        self.structural_components = list(structural_components)

    def _raw(self, frame):
        base, names = super()._raw(frame)
        if not self.structural_components:
            return base, names
        state_arrays, _ = self._sum(frame["state"], STATES, "state_sum")
        extras, extra_names = [], []
        for component in self.structural_components:
            value = pd.to_numeric(frame[component], errors="coerce").to_numpy(float)
            extras.append(value); extra_names.append(component)
            for state, state_value in zip(STATES[:-1], state_arrays):
                extras.append(value * state_value); extra_names.append(f"{component}:state_sum[{state}]")
        return np.column_stack([base, *extras]), [*names, *extra_names]

    def fit(self, frame):
        super().fit(frame)
        required = []
        for component in self.structural_components:
            required.append(component)
            required += [f"{component}:state_sum[{state}]" for state in STATES[:-1]]
        missing = [name for name in required if name not in self.names]
        if missing:
            raise AssertionError(f"Structurally adjusted fMRI design lost required terms: {missing}")
        return self


def fmri_fingerprint(frame, scenario, components):
    columns = [
        "id", "id_event", "state", "pedigree_group", "age0_c5", "t5", "site", "site_scanner",
        "sex", "study", "Y_FP_REL", *components,
    ]
    payload = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=True).to_numpy().tobytes()
    marker = {"scenario": scenario, "components": list(components)}
    return hashlib.sha256(payload + json.dumps(marker, sort_keys=True).encode()).hexdigest()


def fit_fmri_structural_scenario(frame, scenario, sample, components=(), role="PRIMARY_COMPARISON", site_left_out=None):
    components = list(components)
    state_n = frame.groupby("state")["id"].nunique().reindex(STATES, fill_value=0)
    if state_n.min() < MIN_STATE_PARTICIPANTS:
        raise AssertionError(f"{scenario} lacks state coverage: {state_n.to_dict()}")
    fingerprint = fmri_fingerprint(frame, scenario, components)
    checkpoint = CHECKPOINT_DIR / f"fmri_struct__{scenario.lower()}.json"
    if checkpoint.exists() and not FORCE_REFIT:
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") == fingerprint: return saved["result"]
    design = StructuralAdjustedStateDesign(components).fit(frame)
    x = design.transform(frame)
    fit = fit_direct_profile_covariance(x, frame["Y_FP_REL"], frame["id"], frame["state"], frame["pedigree_group"])
    if fit is None: raise RuntimeError(f"Structurally adjusted fMRI fit failed: {scenario}")
    beta, covariance = fit["coef"], fit["robust_cov"]
    state_matrix = design.state_slope_matrix()
    test_rows = []
    for test, matrix in [
        ("all_four_state_slopes_zero", state_matrix),
        ("state_heterogeneity", state_matrix[:3] - state_matrix[3]),
        ("site_x_time", design.site_time_matrix()),
        ("site_x_time_x_state", design.site_time_state_matrix()),
    ]:
        statistic, df, p = wald_matrix(beta, covariance, matrix)
        test_rows.append({"scenario": scenario, "sample": sample, "test": test, "chi2": statistic, "df": df, "p": p})
    estimates = state_matrix @ beta
    state_covariance = state_matrix @ covariance @ state_matrix.T
    standard_errors = np.sqrt(np.maximum(0, np.diag(state_covariance)))
    q4 = norm.ppf(1 - .05 / 8)
    state_rows = [{
        "scenario": scenario, "sample": sample, "role": role, "site_left_out": site_left_out, "state": state,
        "slope_per_5y": float(estimates[index]), "se": float(standard_errors[index]),
        "z": float(estimates[index] / standard_errors[index]) if standard_errors[index] > 0 else np.nan,
        "p_unadjusted": float(2 * norm.sf(abs(estimates[index] / standard_errors[index])))
        if standard_errors[index] > 0 else np.nan,
        "simultaneous_ci_low": float(estimates[index] - q4 * standard_errors[index]),
        "simultaneous_ci_high": float(estimates[index] + q4 * standard_errors[index]),
    } for index, state in enumerate(STATES)]
    result = {
        "summary": {"scenario": scenario, "sample": sample, "role": role, "site_left_out": site_left_out,
                    "components": json.dumps(components), "fit_status": "PASS", "rows": fit["row_n"],
                    "participants": fit["participant_n"], "pedigrees": fit["pedigree_n"],
                    "design_columns": len(design.names), "design_rank": fit["rank"],
                    "condition_number": float(np.linalg.cond(x)), "optimizer_success": fit["optimizer_success"],
                    "optimizer_nfev": fit["optimizer_nfev"], "objective": fit["objective"],
                    "sigma_e2": fit["sigma_e2"], "tau_participant": fit["tau_participant"],
                    "tau_participant_state": fit["tau_participant_state"],
                    "robust_cov_min_eigenvalue": float(np.linalg.eigvalsh((covariance + covariance.T) / 2).min()),
                    "participant_slopes_extracted": False},
        "tests": test_rows, "state": state_rows,
        "payload": {"coef": beta, "cov": covariance, "names": design.names,
                    "state_matrix": state_matrix, "influences": fit["influences"]},
    }
    checkpoint.write_text(json.dumps({"fingerprint": fingerprint, "result": result}, default=jsonable, indent=2) + "\n")
    return result


fmri_results = []
fmri_results.append(fit_fmri_structural_scenario(
    thickness_fmri, "THICKNESS_MATCHED_ORIGINAL", "THICKNESS_MATCHED", [], "MATCHED_ORIGINAL"
))
fmri_results.append(fit_fmri_structural_scenario(
    thickness_fmri, "THICKNESS_PARTICIPANT_MEAN", "THICKNESS_MATCHED",
    ["thickness_between_z"], "BETWEEN_STRUCTURE_ADJUSTED"
))
fmri_results.append(fit_fmri_structural_scenario(
    thickness_fmri, "THICKNESS_WITHIN_BETWEEN", "THICKNESS_MATCHED",
    ["thickness_between_z", "thickness_within_z"], "FINAL_THICKNESS_ADJUSTED"
))
fmri_results.append(fit_fmri_structural_scenario(
    volume_fmri, "VOLUME_MATCHED_ORIGINAL", "VOLUME_MATCHED", [], "MATCHED_ORIGINAL"
))
fmri_results.append(fit_fmri_structural_scenario(
    volume_fmri, "VOLUME_WITHIN_BETWEEN", "VOLUME_MATCHED",
    ["volume_between_z", "volume_within_z"], "SECONDARY_VOLUME_ADJUSTED"
))

scanner_n = thickness_fmri[["id", "id_event", "scanner"]].drop_duplicates().groupby("id")["scanner"].nunique(dropna=False)
same_scanner_thickness = retain_two_sessions(
    thickness_fmri.loc[thickness_fmri["id"].map(scanner_n.eq(1)).fillna(False)]
)
fmri_results.append(fit_fmri_structural_scenario(
    same_scanner_thickness, "THICKNESS_FINAL_SAME_SCANNER", "THICKNESS_SAME_SCANNER",
    ["thickness_between_z", "thickness_within_z"], "FINAL_THICKNESS_SENSITIVITY"
))
for site in sorted(thickness_fmri["site"].astype(str).unique()):
    site_key = re.sub(r"[^A-Za-z0-9]+", "_", site).strip("_").upper()
    loo = retain_two_sessions(thickness_fmri.loc[thickness_fmri["site"].astype(str).ne(site)])
    fmri_results.append(fit_fmri_structural_scenario(
        loo, f"THICKNESS_FINAL_LOO_{site_key}", f"THICKNESS_LOO_{site_key}",
        ["thickness_between_z", "thickness_within_z"], "FINAL_THICKNESS_SENSITIVITY", site,
    ))

fmri_model_summary = pd.DataFrame([result["summary"] for result in fmri_results])
fmri_tests = pd.DataFrame([row for result in fmri_results for row in result["tests"]])
fmri_state_slopes = pd.DataFrame([row for result in fmri_results for row in result["state"]])


def result_by_scenario(scenario):
    matches = [result for result in fmri_results if result["summary"]["scenario"] == scenario]
    if len(matches) != 1: raise KeyError(scenario)
    return matches[0]


def paired_coefficient_change(base_scenario, adjusted_scenario, comparison):
    base = result_by_scenario(base_scenario); adjusted = result_by_scenario(adjusted_scenario)
    if base["summary"]["rows"] != adjusted["summary"]["rows"] or base["summary"]["participants"] != adjusted["summary"]["participants"]:
        raise AssertionError(f"Paired comparison {comparison} does not use identical rows")
    base_payload, adjusted_payload = base["payload"], adjusted["payload"]
    base_matrix = np.asarray(base_payload["state_matrix"], float)
    adjusted_matrix = np.asarray(adjusted_payload["state_matrix"], float)
    base_estimates = base_matrix @ np.asarray(base_payload["coef"], float)
    adjusted_estimates = adjusted_matrix @ np.asarray(adjusted_payload["coef"], float)
    change = adjusted_estimates - base_estimates
    pedigrees = sorted(set(base_payload["influences"]) | set(adjusted_payload["influences"]))
    influence_rows = []
    for pedigree in pedigrees:
        base_influence = np.asarray(base_payload["influences"].get(pedigree, np.zeros(len(base_payload["coef"]))), float)
        adjusted_influence = np.asarray(
            adjusted_payload["influences"].get(pedigree, np.zeros(len(adjusted_payload["coef"]))), float
        )
        influence_rows.append(adjusted_matrix @ adjusted_influence - base_matrix @ base_influence)
    influence_rows = np.asarray(influence_rows)
    change_covariance = influence_rows.T @ influence_rows
    statistic, df, p = wald_matrix(change, change_covariance, np.eye(4))
    rng = np.random.default_rng(SEED + sum(ord(character) for character in comparison))
    multipliers = rng.standard_normal((COEFFICIENT_CHANGE_BOOTSTRAPS, len(pedigrees)))
    bootstrap = change + multipliers @ influence_rows
    state_rows = []
    se = np.sqrt(np.maximum(0, np.diag(change_covariance)))
    for index, state in enumerate(STATES):
        state_rows.append({
            "comparison": comparison, "base_scenario": base_scenario, "adjusted_scenario": adjusted_scenario,
            "state": state, "original_slope_per_5y": float(base_estimates[index]),
            "adjusted_slope_per_5y": float(adjusted_estimates[index]),
            "change_adjusted_minus_original": float(change[index]), "change_se": float(se[index]),
            "bootstrap_ci_low": float(np.quantile(bootstrap[:, index], .025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap[:, index], .975)),
            "pedigree_bootstrap_draws": COEFFICIENT_CHANGE_BOOTSTRAPS,
        })
    base_norm = float(np.linalg.norm(base_estimates)); adjusted_norm = float(np.linalg.norm(adjusted_estimates))
    joint = {
        "comparison": comparison, "base_scenario": base_scenario, "adjusted_scenario": adjusted_scenario,
        "chi2": statistic, "df": df, "p": p, "original_vector_l2": base_norm,
        "adjusted_vector_l2": adjusted_norm,
        "attenuation_fraction": float(1 - adjusted_norm / base_norm) if base_norm > 0 else np.nan,
        "pedigrees": len(pedigrees), "pedigree_bootstrap_draws": COEFFICIENT_CHANGE_BOOTSTRAPS,
    }
    return state_rows, joint


change_specs = [
    ("THICKNESS_MATCHED_ORIGINAL", "THICKNESS_PARTICIPANT_MEAN", "THICKNESS_MEAN_MINUS_ORIGINAL"),
    ("THICKNESS_MATCHED_ORIGINAL", "THICKNESS_WITHIN_BETWEEN", "THICKNESS_FULL_MINUS_ORIGINAL"),
    ("THICKNESS_PARTICIPANT_MEAN", "THICKNESS_WITHIN_BETWEEN", "THICKNESS_WITHIN_INCREMENT"),
    ("VOLUME_MATCHED_ORIGINAL", "VOLUME_WITHIN_BETWEEN", "VOLUME_FULL_MINUS_ORIGINAL"),
]
change_state_rows, change_joint_rows = [], []
for base_scenario, adjusted_scenario, comparison in change_specs:
    state_rows, joint_row = paired_coefficient_change(base_scenario, adjusted_scenario, comparison)
    change_state_rows.extend(state_rows); change_joint_rows.append(joint_row)
fmri_coefficient_changes = pd.DataFrame(change_state_rows)
fmri_coefficient_change_joint = pd.DataFrame(change_joint_rows)

display(structural_component_audit)
display(fmri_matched_cohort)
display(fmri_model_summary)
display(fmri_tests)
display(fmri_state_slopes)
display(fmri_coefficient_changes)
display(fmri_coefficient_change_joint)



# Summarize the primary sensitivity results

def test_value(table, scenario, test, column="p"):
    hit = table.loc[table["scenario"].eq(scenario) & table["test"].eq(test), column]
    if len(hit) != 1:
        raise KeyError(f"Expected one {scenario}/{test} row; found {len(hit)}")
    return float(hit.iloc[0])


thickness_structural_p = test_value(structural_tests, "THICKNESS_PRIMARY", "time_zero")
volume_structural_p = test_value(structural_tests, "VOLUME_ICV_SECONDARY", "time_zero")
thickness_final_p = test_value(fmri_tests, "THICKNESS_WITHIN_BETWEEN", "all_four_state_slopes_zero")
thickness_site_time_p = test_value(fmri_tests, "THICKNESS_WITHIN_BETWEEN", "site_x_time")
thickness_site_state_p = test_value(fmri_tests, "THICKNESS_WITHIN_BETWEEN", "site_x_time_x_state")
same_scanner_p = test_value(fmri_tests, "THICKNESS_FINAL_SAME_SCANNER", "all_four_state_slopes_zero")
change_row = fmri_coefficient_change_joint.loc[
    fmri_coefficient_change_joint["comparison"].eq("THICKNESS_FULL_MINUS_ORIGINAL")
].iloc[0]
volume_change_row = fmri_coefficient_change_joint.loc[
    fmri_coefficient_change_joint["comparison"].eq("VOLUME_FULL_MINUS_ORIGINAL")
].iloc[0]

original_slopes = fmri_state_slopes.loc[
    fmri_state_slopes["scenario"].eq("THICKNESS_MATCHED_ORIGINAL")
].set_index("state")["slope_per_5y"].reindex(STATES)
final_slopes = fmri_state_slopes.loc[
    fmri_state_slopes["scenario"].eq("THICKNESS_WITHIN_BETWEEN")
].set_index("state")["slope_per_5y"].reindex(STATES)
sensitivity_slopes = fmri_state_slopes.loc[
    fmri_state_slopes["role"].eq("FINAL_THICKNESS_SENSITIVITY")
].pivot(index="scenario", columns="state", values="slope_per_5y").reindex(columns=STATES)
reference_sign = np.sign(final_slopes.to_numpy(float))
directional_rows = sensitivity_slopes.apply(
    lambda row: bool(np.all(np.sign(row.to_numpy(float)) == reference_sign)), axis=1
) if len(sensitivity_slopes) else pd.Series(dtype=bool)
loo_tests = fmri_tests.loc[
    fmri_tests["scenario"].str.startswith("THICKNESS_FINAL_LOO_")
    & fmri_tests["test"].eq("all_four_state_slopes_zero")
].copy()

attenuation_fraction = float(change_row["attenuation_fraction"])
paired_change_p = float(change_row["p"])
structural_thickness_detected = thickness_structural_p < .05
functional_retained = bool(thickness_final_p < .05 and np.all(np.sign(original_slopes) == np.sign(final_slopes)))
functionally_unchanged = bool(
    paired_change_p >= .05 and np.isfinite(attenuation_fraction)
    and abs(attenuation_fraction) <= UNCHANGED_ATTENUATION_FRACTION
)
substantial_attenuation = bool(
    paired_change_p < .05 and np.isfinite(attenuation_fraction)
    and attenuation_fraction >= SUBSTANTIAL_ATTENUATION_FRACTION
)
site_magnitude_heterogeneous = bool(thickness_site_time_p < .05 or thickness_site_state_p < .05)
sensitivity_direction_stable = bool(directional_rows.all()) if len(directional_rows) else False

if (not sensitivity_direction_stable) or (
    np.sign(float(volume_change_row["attenuation_fraction"])) != np.sign(attenuation_fraction)
    and max(abs(float(volume_change_row["attenuation_fraction"])), abs(attenuation_fraction)) >= SUBSTANTIAL_ATTENUATION_FRACTION
):
    final_claim = "STRUCTURE_IS_A_QUALIFICATION_RESULTS_UNSTABLE_ACROSS_SITES_OR_REPRESENTATIONS"
elif (not structural_thickness_detected) and functionally_unchanged:
    final_claim = "NO_FP_RELATIVE_STRUCTURAL_TRAJECTORY_FMRI_ESTIMATES_UNCHANGED"
elif structural_thickness_detected and functional_retained and not substantial_attenuation:
    final_claim = "PARALLEL_FUNCTIONAL_AND_STRUCTURAL_AGING_FMRI_NOT_REDUCIBLE_TO_STRUCTURE"
elif structural_thickness_detected and substantial_attenuation:
    final_claim = "FMRI_PROFILE_SUBSTANTIALLY_ATTENUATED_AFTER_STRUCTURE_ADJUSTMENT"
else:
    final_claim = "STRUCTURAL_SENSITIVITY_INCONCLUSIVE"

decision_table = pd.DataFrame([
    {"criterion": "FP-relative thickness trajectory", "estimate_or_flag": structural_thickness_detected,
     "p": thickness_structural_p, "interpretation": "Detected" if structural_thickness_detected else "Not detected; not equivalence"},
    {"criterion": "FP-relative volume trajectory (secondary)", "estimate_or_flag": volume_structural_p < .05,
     "p": volume_structural_p, "interpretation": "ICV-adjusted volume sensitivity"},
    {"criterion": "Final thickness-adjusted fMRI four-state vector", "estimate_or_flag": functional_retained,
     "p": thickness_final_p, "interpretation": "Retained" if functional_retained else "Not retained"},
    {"criterion": "Paired thickness adjustment changes coefficient vector", "estimate_or_flag": paired_change_p < .05,
     "p": paired_change_p, "interpretation": f"attenuation_fraction={attenuation_fraction:.4f}"},
    {"criterion": "Direct site moderation in final model", "estimate_or_flag": site_magnitude_heterogeneous,
     "p": min(thickness_site_time_p, thickness_site_state_p),
     "interpretation": "Magnitude heterogeneity" if site_magnitude_heterogeneous else "Not detected"},
    {"criterion": "Same-scanner joint evidence", "estimate_or_flag": same_scanner_p < .05,
     "p": same_scanner_p, "interpretation": "Sensitivity only"},
    {"criterion": "All final sensitivities preserve state directions", "estimate_or_flag": sensitivity_direction_stable,
     "p": np.nan, "interpretation": f"{int(directional_rows.sum())}/{len(directional_rows)} scenarios"},
    {"criterion": "Final bounded claim", "estimate_or_flag": final_claim, "p": np.nan,
     "interpretation": "No causal-mediation language; structure is a matched sensitivity"},
])

site_robustness_summary = pd.DataFrame([{
    "primary_final_site_x_time_p": thickness_site_time_p,
    "primary_final_site_x_time_x_state_p": thickness_site_state_p,
    "same_scanner_all_four_p": same_scanner_p,
    "loo_fit_n": int(len(loo_tests)),
    "loo_all_four_p_max": float(loo_tests["p"].max()) if len(loo_tests) else np.nan,
    "all_sensitivity_state_directions_preserved": sensitivity_direction_stable,
    "site_invariance_claimed": False,
}])

display(decision_table)
display(site_robustness_summary)



# Initial structural and adjusted-fMRI figures

# Figure 1: direct structural trajectories.
figure, axis = plt.subplots(figsize=(7.5, 4.2))
plot = structural_slopes.copy()
plot["label"] = plot["scenario"].map({
    "THICKNESS_PRIMARY": "Thickness (primary)",
    "VOLUME_ICV_SECONDARY": "Volume / ICV (secondary)",
})
positions = np.arange(len(plot))
axis.errorbar(
    plot["slope_per_5y"], positions,
    xerr=np.vstack([plot["slope_per_5y"] - plot["ci_low"], plot["ci_high"] - plot["slope_per_5y"]]),
    fmt="o", capsize=3, color="#2f4b7c",
)
axis.axvline(0, color="black", linewidth=1)
axis.set_yticks(positions, plot["label"])
axis.set_xlabel("FP-relative structural change per 5 years")
axis.set_title("Network-matched structural trajectories")
figure.tight_layout()
figure.savefig(FIG_DIR / "structural_fp_relative_trajectories.png", dpi=220, bbox_inches="tight")
plt.close(figure)

# Figure 2: original, participant-mean-adjusted and full thickness-adjusted fMRI estimates.
selected = fmri_state_slopes.loc[fmri_state_slopes["scenario"].isin([
    "THICKNESS_MATCHED_ORIGINAL", "THICKNESS_PARTICIPANT_MEAN", "THICKNESS_WITHIN_BETWEEN",
])].copy()
labels = {
    "THICKNESS_MATCHED_ORIGINAL": "Original (matched)",
    "THICKNESS_PARTICIPANT_MEAN": "Participant-mean thickness",
    "THICKNESS_WITHIN_BETWEEN": "Mean + within-person thickness",
}
figure, axis = plt.subplots(figsize=(9, 5))
offsets = {name: offset for name, offset in zip(labels, [-.18, 0, .18])}
for scenario, group in selected.groupby("scenario", sort=False):
    group = group.set_index("state").reindex(STATES)
    x = np.arange(len(STATES)) + offsets[scenario]
    axis.errorbar(
        x, group["slope_per_5y"],
        yerr=np.vstack([group["slope_per_5y"] - group["simultaneous_ci_low"],
                        group["simultaneous_ci_high"] - group["slope_per_5y"]]),
        fmt="o", capsize=3, label=labels[scenario],
    )
axis.axhline(0, color="black", linewidth=1)
axis.set_xticks(np.arange(len(STATES)), STATES)
axis.set_ylabel("Amplitude-standardized FP-relative fMRI slope per 5 years")
axis.set_title("Coefficient stability after network-matched thickness adjustment")
axis.legend(frameon=False)
figure.tight_layout()
figure.savefig(FIG_DIR / "fmri_original_vs_structural_adjusted.png", dpi=220, bbox_inches="tight")
plt.close(figure)

# Figure 3: final adjusted primary, same-scanner and leave-one-site-out slopes.
robust = fmri_state_slopes.loc[
    fmri_state_slopes["scenario"].eq("THICKNESS_WITHIN_BETWEEN")
    | fmri_state_slopes["role"].eq("FINAL_THICKNESS_SENSITIVITY")
].copy()
robust["label"] = np.where(
    robust["scenario"].eq("THICKNESS_WITHIN_BETWEEN"), "Primary final",
    np.where(robust["scenario"].eq("THICKNESS_FINAL_SAME_SCANNER"), "Same scanner",
             "Leave out " + robust["site_left_out"].fillna("unknown").astype(str)),
)
figure, axes = plt.subplots(1, len(STATES), figsize=(14, max(4.5, .42 * robust["label"].nunique() + 2)), sharex=True)
for axis, state in zip(axes, STATES):
    state_frame = robust.loc[robust["state"].eq(state)].copy().sort_values("label")
    positions = np.arange(len(state_frame))
    axis.errorbar(
        state_frame["slope_per_5y"], positions,
        xerr=np.vstack([state_frame["slope_per_5y"] - state_frame["simultaneous_ci_low"],
                        state_frame["simultaneous_ci_high"] - state_frame["slope_per_5y"]]),
        fmt="o", capsize=2, color="#7a5195",
    )
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_title(state)
    axis.set_yticks(positions)
    axis.set_yticklabels(state_frame["label"] if axis is axes[0] else [])
axes[0].set_xlabel("Final adjusted slope per 5 years")
figure.suptitle("Final thickness-adjusted site/scanner sensitivity", y=1.01)
figure.tight_layout()
figure.savefig(FIG_DIR / "final_adjusted_site_sensitivity.png", dpi=220, bbox_inches="tight")
plt.close(figure)



# Robust-covariance diagnostics

def structural_covariance_diagnostic(frame, outcome, scenario, extra_columns=(), saved_slope=None):
    design = StructuralLongitudinalDesign(extra_columns).fit(frame)
    x = design.transform(frame)
    fit = fit_random_intercept_pedigree_sandwich(
        x, frame[outcome], frame["id"], frame["pedigree_group"]
    )
    if fit is None:
        raise RuntimeError(f"Diagnostic refit failed: {scenario}")
    covariance = np.asarray(fit["robust_cov"], float)
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigen_tolerance = max(
        np.finfo(float).eps * max(covariance.shape) * max(float(np.max(np.abs(eigenvalues))), 1.0),
        1e-14,
    )
    singular_values = np.linalg.svd(x, compute_uv=False)
    design_tolerance = np.finfo(float).eps * max(x.shape) * max(float(singular_values.max()), 1.0)
    time_vector = design.time_vector()
    estimate, se, z, p = scalar_contrast(fit["coef"], covariance, time_vector)
    summary = {
        "scenario": scenario,
        "outcome": outcome,
        "rows": int(fit["row_n"]),
        "participants": int(fit["participant_n"]),
        "pedigrees": int(fit["pedigree_n"]),
        "design_columns": int(x.shape[1]),
        "design_rank_qr": int(fit["rank"]),
        "design_rank_svd": int(np.sum(singular_values > design_tolerance)),
        "design_min_singular_value": float(singular_values.min()),
        "design_max_singular_value": float(singular_values.max()),
        "design_condition_number": float(np.linalg.cond(x)),
        "robust_cov_rank": int(np.sum(eigenvalues > eigen_tolerance)),
        "robust_cov_dimension": int(covariance.shape[0]),
        "robust_cov_min_eigenvalue": float(eigenvalues.min()),
        "robust_cov_max_eigenvalue": float(eigenvalues.max()),
        "robust_cov_eigen_tolerance": float(eigen_tolerance),
        "robust_cov_negative_eigenvalues": int(np.sum(eigenvalues < -eigen_tolerance)),
        "robust_cov_near_zero_eigenvalues": int(np.sum(np.abs(eigenvalues) <= eigen_tolerance)),
        "robust_cov_positive_condition_number": float(
            eigenvalues[eigenvalues > eigen_tolerance].max() / eigenvalues[eigenvalues > eigen_tolerance].min()
        ) if np.any(eigenvalues > eigen_tolerance) else np.inf,
        "time_slope_per_5y_diagnostic": float(estimate),
        "time_slope_se_diagnostic": float(se),
        "time_slope_p_diagnostic": float(p),
        "saved_slope_per_5y": float(saved_slope) if saved_slope is not None else np.nan,
        "absolute_slope_reproduction_error": (
            float(abs(estimate - saved_slope)) if saved_slope is not None else np.nan
        ),
        "optimizer_success": bool(fit["optimizer_success"]),
    }
    spectrum = pd.DataFrame({
        "scenario": scenario,
        "eigen_index_ascending": np.arange(len(eigenvalues)),
        "eigenvalue": eigenvalues,
        "eigen_tolerance": eigen_tolerance,
        "classification": np.where(
            eigenvalues < -eigen_tolerance, "NEGATIVE",
            np.where(eigenvalues > eigen_tolerance, "POSITIVE", "NEAR_ZERO"),
        ),
    })
    return summary, spectrum


saved_structural_slope = structural_slopes.set_index("scenario")["slope_per_5y"]
diagnostic_specs = [
    (thickness_visits, "Y_THICK_REL", "THICKNESS_PRIMARY", [], saved_structural_slope["THICKNESS_PRIMARY"]),
    (volume_visits, "Y_VOLUME_REL", "VOLUME_ICV_SECONDARY", ["log_icv_between_z"],
     saved_structural_slope["VOLUME_ICV_SECONDARY"]),
]
structural_covariance_diagnostic_rows = []
structural_covariance_spectrum_frames = []
for frame, outcome, scenario, extras, saved in diagnostic_specs:
    row, spectrum = structural_covariance_diagnostic(frame, outcome, scenario, extras, saved)
    structural_covariance_diagnostic_rows.append(row)
    structural_covariance_spectrum_frames.append(spectrum)

structural_covariance_diagnostics = pd.DataFrame(structural_covariance_diagnostic_rows)
structural_covariance_eigenvalues = pd.concat(structural_covariance_spectrum_frames, ignore_index=True)
if structural_covariance_diagnostics["absolute_slope_reproduction_error"].max() > 1e-8:
    raise AssertionError("Structural diagnostic refit did not reproduce the saved slopes")

display(structural_covariance_diagnostics)
display(structural_covariance_eigenvalues.groupby(["scenario", "classification"]).size().rename("n").reset_index())



# Structural-component coefficients

def component_state_matrix(payload, component):
    names = list(payload["names"])
    if component not in names:
        raise KeyError(f"{component} is absent from saved design")
    matrix = []
    for state in STATES:
        vector = np.zeros(len(names))
        vector[names.index(component)] = 1.0
        for coded_state in STATES[:-1]:
            term = f"{component}:state_sum[{coded_state}]"
            if term not in names:
                raise KeyError(f"Saved design lacks {term}")
            if state == coded_state:
                vector[names.index(term)] = 1.0
            elif state == STATES[-1]:
                vector[names.index(term)] = -1.0
        matrix.append(vector)
    return np.vstack(matrix)


def export_saved_component(result, component, component_type):
    payload = result["payload"]
    beta = np.asarray(payload["coef"], float)
    covariance = np.asarray(payload["cov"], float)
    matrix = component_state_matrix(payload, component)
    estimates = matrix @ beta
    component_covariance = matrix @ covariance @ matrix.T
    standard_errors = np.sqrt(np.maximum(0, np.diag(component_covariance)))
    q4 = norm.ppf(1 - .05 / 8)
    coefficient_rows = []
    for index, state in enumerate(STATES):
        z = estimates[index] / standard_errors[index] if standard_errors[index] > 0 else np.nan
        coefficient_rows.append({
            "scenario": result["summary"]["scenario"],
            "structure_component": component,
            "component_type": component_type,
            "state": state,
            "coefficient_per_1sd": float(estimates[index]),
            "se": float(standard_errors[index]),
            "z": float(z),
            "p_unadjusted": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "simultaneous_ci_low": float(estimates[index] - q4 * standard_errors[index]),
            "simultaneous_ci_high": float(estimates[index] + q4 * standard_errors[index]),
            "model_refitted_for_export": False,
        })
    test_rows = []
    for test, contrast in [
        ("all_four_component_coefficients_zero", matrix),
        ("component_state_heterogeneity", matrix[:3] - matrix[3]),
    ]:
        statistic, df, p = wald_matrix(beta, covariance, contrast)
        test_rows.append({
            "scenario": result["summary"]["scenario"],
            "structure_component": component,
            "component_type": component_type,
            "test": test,
            "chi2": statistic,
            "df": df,
            "p": p,
            "model_refitted_for_export": False,
        })
    return coefficient_rows, test_rows


saved_component_specs = [
    ("THICKNESS_PARTICIPANT_MEAN", "thickness_between_z", "BETWEEN_PERSON"),
    ("THICKNESS_WITHIN_BETWEEN", "thickness_between_z", "BETWEEN_PERSON"),
    ("THICKNESS_WITHIN_BETWEEN", "thickness_within_z", "WITHIN_PERSON"),
    ("VOLUME_WITHIN_BETWEEN", "volume_between_z", "BETWEEN_PERSON"),
    ("VOLUME_WITHIN_BETWEEN", "volume_within_z", "WITHIN_PERSON"),
]
structural_fmri_coefficient_rows, structural_fmri_test_rows = [], []
for scenario, component, component_type in saved_component_specs:
    result = next(item for item in fmri_results if item["summary"]["scenario"] == scenario)
    coefficients, tests = export_saved_component(result, component, component_type)
    structural_fmri_coefficient_rows.extend(coefficients)
    structural_fmri_test_rows.extend(tests)

structural_fmri_component_coefficients = pd.DataFrame(structural_fmri_coefficient_rows)
structural_fmri_component_joint_tests = pd.DataFrame(structural_fmri_test_rows)
display(structural_fmri_component_coefficients)
display(structural_fmri_component_joint_tests)



# Surface-area analysis

area_network_features = pd.DataFrame(index=structural_features.index)
area_coverage_rows = []
for network in ELIGIBLE_NETWORKS:
    areas = network_areas[network]
    columns = [area_lookup[compact_token(area_name)] for area_name in areas]
    values = surface_area[columns].copy()
    values.columns = areas
    values = values.where(values > 0)
    area_sum = values.sum(axis=1, min_count=len(areas))
    area_network_features[f"surface_area_sum__{network}"] = area_sum.reindex(area_network_features.index)
    area_network_features[f"log_surface_area__{network}"] = np.log(area_sum.where(area_sum > 0)).reindex(
        area_network_features.index
    )
    area_coverage_rows.append({
        "network": network,
        "expected_parcels": len(areas),
        "sessions_all_parcels_positive": int(area_sum.notna().sum()),
        "median_parcel_coverage": float(values.notna().mean(axis=1).median()),
        "primary_network_weight_in_reference": 0.0 if network == FP_NETWORK else 1 / 9,
    })

area_network_features["Y_AREA_REL"] = (
    area_network_features[f"log_surface_area__{FP_NETWORK}"]
    - area_network_features[[f"log_surface_area__{network}" for network in reference_networks]].mean(
        axis=1, skipna=False
    )
)
area_identity_check = (
    area_network_features["Y_AREA_REL"]
    - np.log(area_network_features[f"surface_area_sum__{FP_NETWORK}"])
    + np.log(area_network_features[[f"surface_area_sum__{network}" for network in reference_networks]]).mean(
        axis=1, skipna=False
    )
)
area_definition_audit = pd.DataFrame([{
    "definition": "log(FP summed area) - equal-weight mean log(summed area) of other nine",
    "max_identity_error": float(np.nanmax(np.abs(area_identity_check))),
    "reference_network_n": 9,
    "reference_is_equal_network_weighted": True,
    "parcel_count_weighting_used": False,
}])
if area_definition_audit.loc[0, "max_identity_error"] > 1e-12:
    raise AssertionError("FP-relative surface-area identity failed")

area_structural_visits = event_meta.merge(
    area_network_features.reset_index(), on="id_event", how="left", validate="one_to_one"
)
area_visits = retain_two_structural_visits(area_structural_visits, "Y_AREA_REL")
area_result = fit_structural_model(area_visits, "Y_AREA_REL", "SURFACE_AREA_RELATIVE")
area_relative_model_summary = pd.DataFrame([area_result["summary"]])
area_relative_tests = pd.DataFrame(area_result["tests"])
area_relative_slope = pd.DataFrame(area_result["slope"])
area_relative_site_slopes = pd.DataFrame(area_result["site"])
area_coverage_audit = pd.DataFrame(area_coverage_rows)
area_cohort_summary = pd.DataFrame([{
    "cohort": "SURFACE_AREA_TWO_PLUS_MATCHED",
    "participants": int(area_visits["id"].nunique()),
    "sessions": int(area_visits["id_event"].nunique()),
}])

area_slope_value = float(area_relative_slope.loc[0, "slope_per_5y"])
area_ci_low = float(area_relative_slope.loc[0, "ci_low"])
area_ci_high = float(area_relative_slope.loc[0, "ci_high"])
area_p = float(area_relative_slope.loc[0, "p"])
area_percent_5y = float(100 * np.expm1(area_slope_value))
area_ci_excludes_zero = bool(area_ci_low > 0 or area_ci_high < 0)
AREA_MATERIAL = bool(
    area_ci_excludes_zero and abs(area_slope_value) >= AREA_MATERIAL_LOG_MARGIN_5Y
)
area_branch_decision = pd.DataFrame([{
    "slope_log_ratio_per_5y": area_slope_value,
    "ci_low": area_ci_low,
    "ci_high": area_ci_high,
    "p": area_p,
    "percent_change_per_5y": area_percent_5y,
    "material_margin_log_per_5y": AREA_MATERIAL_LOG_MARGIN_5Y,
    "material_margin_percent_per_5y": AREA_MATERIAL_PERCENT_MARGIN_5Y,
    "ci_excludes_zero": area_ci_excludes_zero,
    "material_area_change": AREA_MATERIAL,
    "branch": "RUN_ONE_AREA_ADJUSTMENT_THEN_STOP" if AREA_MATERIAL else "NO_AREA_ADJUSTMENT",
    "reporting": (
        "Material FP-relative area change detected; run one surface-area adjustment and stop."
        if AREA_MATERIAL else
        "FP-relative area change not materially detected by the materiality criterion; evidence against a simple geometry explanation, not formal equivalence."
    ),
}])

# Add the area covariance spectrum to the structural diagnostics.
area_diag, area_spectrum = structural_covariance_diagnostic(
    area_visits, "Y_AREA_REL", "SURFACE_AREA_RELATIVE", [], area_slope_value
)
structural_covariance_diagnostics = pd.concat([
    structural_covariance_diagnostics,
    pd.DataFrame([area_diag]),
], ignore_index=True)
structural_covariance_eigenvalues = pd.concat([
    structural_covariance_eigenvalues,
    area_spectrum,
], ignore_index=True)

display(area_definition_audit)
display(area_coverage_audit)
display(area_cohort_summary)
display(area_relative_model_summary)
display(area_relative_slope)
display(area_relative_tests)
display(area_branch_decision)



# Conditional surface-area adjustment

def paired_change_from_saved_results(base, adjusted, comparison):
    if base["summary"]["rows"] != adjusted["summary"]["rows"]:
        raise AssertionError(f"{comparison}: original and adjusted rows differ")
    base_payload, adjusted_payload = base["payload"], adjusted["payload"]
    base_matrix = np.asarray(base_payload["state_matrix"], float)
    adjusted_matrix = np.asarray(adjusted_payload["state_matrix"], float)
    base_estimates = base_matrix @ np.asarray(base_payload["coef"], float)
    adjusted_estimates = adjusted_matrix @ np.asarray(adjusted_payload["coef"], float)
    change = adjusted_estimates - base_estimates
    pedigrees = sorted(set(base_payload["influences"]) | set(adjusted_payload["influences"]))
    influence_rows = []
    for pedigree in pedigrees:
        base_influence = np.asarray(
            base_payload["influences"].get(pedigree, np.zeros(len(base_payload["coef"]))), float
        )
        adjusted_influence = np.asarray(
            adjusted_payload["influences"].get(pedigree, np.zeros(len(adjusted_payload["coef"]))), float
        )
        influence_rows.append(adjusted_matrix @ adjusted_influence - base_matrix @ base_influence)
    influence_rows = np.asarray(influence_rows)
    change_covariance = influence_rows.T @ influence_rows
    statistic, df, p = wald_matrix(change, change_covariance, np.eye(4))
    rng = np.random.default_rng(SEED + 411)
    multipliers = rng.standard_normal((COEFFICIENT_CHANGE_BOOTSTRAPS, len(pedigrees)))
    bootstrap = change + multipliers @ influence_rows
    rows = []
    for index, state in enumerate(STATES):
        rows.append({
            "comparison": comparison,
            "state": state,
            "original_slope_per_5y": float(base_estimates[index]),
            "adjusted_slope_per_5y": float(adjusted_estimates[index]),
            "change_adjusted_minus_original": float(change[index]),
            "bootstrap_ci_low": float(np.quantile(bootstrap[:, index], .025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap[:, index], .975)),
            "pedigree_bootstrap_draws": COEFFICIENT_CHANGE_BOOTSTRAPS,
        })
    base_norm = float(np.linalg.norm(base_estimates))
    adjusted_norm = float(np.linalg.norm(adjusted_estimates))
    joint = pd.DataFrame([{
        "comparison": comparison,
        "chi2": statistic,
        "df": df,
        "p": p,
        "original_vector_l2": base_norm,
        "adjusted_vector_l2": adjusted_norm,
        "attenuation_fraction": float(1 - adjusted_norm / base_norm) if base_norm > 0 else np.nan,
        "pedigrees": len(pedigrees),
        "pedigree_bootstrap_draws": COEFFICIENT_CHANGE_BOOTSTRAPS,
    }])
    return pd.DataFrame(rows), joint


if AREA_MATERIAL:
    area_components, area_component_row = structural_components(area_visits, "Y_AREA_REL", "area")
    area_fmri = merge_structure_with_fmri(area_components, "area")
    area_fmri_matched_cohort = pd.DataFrame([{
        "sample": "AREA_MATCHED",
        "participants": int(area_fmri["id"].nunique()),
        "sessions": int(area_fmri["id_event"].nunique()),
        "state_rows": int(len(area_fmri)),
    }])
    area_fmri_original = fit_fmri_structural_scenario(
        area_fmri, "AREA_MATCHED_ORIGINAL", "AREA_MATCHED", [], "MATCHED_ORIGINAL"
    )
    area_fmri_adjusted = fit_fmri_structural_scenario(
        area_fmri, "AREA_WITHIN_BETWEEN", "AREA_MATCHED",
        ["area_between_z", "area_within_z"], "FINAL_AREA_ADJUSTED"
    )
    area_fmri_model_summary = pd.DataFrame([
        area_fmri_original["summary"], area_fmri_adjusted["summary"]
    ])
    area_fmri_tests = pd.DataFrame([
        *area_fmri_original["tests"], *area_fmri_adjusted["tests"]
    ])
    area_fmri_state_slopes = pd.DataFrame([
        *area_fmri_original["state"], *area_fmri_adjusted["state"]
    ])
    area_fmri_coefficient_changes, area_fmri_coefficient_change_joint = paired_change_from_saved_results(
        area_fmri_original, area_fmri_adjusted, "AREA_FULL_MINUS_ORIGINAL"
    )
    area_component_coefficients_rows, area_component_tests_rows = [], []
    for component, component_type in [
        ("area_between_z", "BETWEEN_PERSON"), ("area_within_z", "WITHIN_PERSON")
    ]:
        coefficients, tests = export_saved_component(area_fmri_adjusted, component, component_type)
        area_component_coefficients_rows.extend(coefficients)
        area_component_tests_rows.extend(tests)
    area_fmri_component_coefficients = pd.DataFrame(area_component_coefficients_rows)
    area_fmri_component_joint_tests = pd.DataFrame(area_component_tests_rows)
    area_component_audit = pd.DataFrame([area_component_row])
else:
    area_fmri_matched_cohort = pd.DataFrame(columns=["sample", "participants", "sessions", "state_rows"])
    area_fmri_model_summary = pd.DataFrame(columns=["scenario", "fit_status"])
    area_fmri_tests = pd.DataFrame(columns=["scenario", "test", "chi2", "df", "p"])
    area_fmri_state_slopes = pd.DataFrame(columns=["scenario", "state", "slope_per_5y", "se"])
    area_fmri_coefficient_changes = pd.DataFrame(columns=[
        "comparison", "state", "original_slope_per_5y", "adjusted_slope_per_5y",
        "change_adjusted_minus_original", "bootstrap_ci_low", "bootstrap_ci_high",
    ])
    area_fmri_coefficient_change_joint = pd.DataFrame(columns=[
        "comparison", "chi2", "df", "p", "attenuation_fraction",
    ])
    area_fmri_component_coefficients = pd.DataFrame(columns=[
        "scenario", "structure_component", "component_type", "state", "coefficient_per_1sd",
    ])
    area_fmri_component_joint_tests = pd.DataFrame(columns=[
        "scenario", "structure_component", "component_type", "test", "chi2", "df", "p",
    ])
    area_component_audit = pd.DataFrame([{
        "structure": "area", "decomposition_performed": False,
        "reason": "Surface-area trajectory did not meet the materiality criterion",
    }])

display(area_fmri_model_summary)
display(area_fmri_tests)
display(area_fmri_coefficient_changes)
display(area_fmri_coefficient_change_joint)
display(area_fmri_component_coefficients)
display(area_fmri_component_joint_tests)



# Absolute ten-network structural context

absolute_features = structural_features.join(area_network_features, how="left")
absolute_structural_visits = event_meta.merge(
    absolute_features.reset_index(), on="id_event", how="left", validate="one_to_one"
)


def absolute_structural_fit(network, modality):
    if modality == "THICKNESS":
        outcome = f"thickness__{network}"
        extras = []
    elif modality == "LOG_AREA":
        outcome = f"log_surface_area__{network}"
        extras = []
    elif modality == "LOG_VOLUME_ICV":
        outcome = f"volume_log_ratio_icv__{network}"
        extras = ["log_icv_between_z"]
    else:
        raise KeyError(modality)
    frame = retain_two_structural_visits(absolute_structural_visits, outcome)
    if extras:
        frame["log_icv_between"] = frame.groupby("id")["log_icv"].transform("mean")
        participant_values = frame[["id", "log_icv_between"]].drop_duplicates("id")["log_icv_between"]
        scale = float(participant_values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise AssertionError(f"{network}: participant-mean log ICV has no variation")
        center = float(participant_values.mean())
        frame["log_icv_between_z"] = (frame["log_icv_between"] - center) / scale
    scenario_token = re.sub(r"[^A-Za-z0-9]+", "_", network).strip("_").upper()
    result = fit_structural_model(
        frame, outcome, f"ABS_{modality}_{scenario_token}", extras
    )
    summary = dict(result["summary"], network=network, modality=modality)
    slope = dict(result["slope"][0], network=network, modality=modality, context_type="ABSOLUTE_NETWORK")
    tests = [dict(row, network=network, modality=modality) for row in result["tests"]]
    return summary, slope, tests


absolute_summary_rows, absolute_slope_rows, absolute_test_rows = [], [], []
for modality in ["THICKNESS", "LOG_AREA", "LOG_VOLUME_ICV"]:
    for network in ELIGIBLE_NETWORKS:
        summary, slope, tests = absolute_structural_fit(network, modality)
        absolute_summary_rows.append(summary)
        absolute_slope_rows.append(slope)
        absolute_test_rows.extend(tests)

structural_absolute_model_summary = pd.DataFrame(absolute_summary_rows)
structural_absolute_slopes = pd.DataFrame(absolute_slope_rows)
structural_absolute_tests = pd.DataFrame(absolute_test_rows)

relative_rows = []
for modality, slope_table, scenario in [
    ("THICKNESS", structural_slopes, "THICKNESS_PRIMARY"),
    ("LOG_AREA", area_relative_slope, "SURFACE_AREA_RELATIVE"),
    ("LOG_VOLUME_ICV", structural_slopes, "VOLUME_ICV_SECONDARY"),
]:
    row = slope_table.loc[slope_table["scenario"].eq(scenario)].iloc[0].to_dict()
    row.update({
        "network": "FP minus equal-weight mean(other 9)",
        "modality": modality,
        "context_type": "FP_MINUS_REFERENCE",
    })
    relative_rows.append(row)
structural_context_slopes = pd.concat([
    structural_absolute_slopes,
    pd.DataFrame(relative_rows),
], ignore_index=True, sort=False)

display(structural_absolute_model_summary)
display(structural_context_slopes[[
    "modality", "network", "context_type", "slope_per_5y", "se", "ci_low", "ci_high", "p"
]])



# Final structural figures

modality_labels = {
    "THICKNESS": "Thickness difference (mm per 5 years)",
    "LOG_AREA": "Log surface-area ratio change per 5 years",
    "LOG_VOLUME_ICV": "Log volume/eTIV ratio change per 5 years",
}
modality_titles = {
    "THICKNESS": "FP-relative thickness",
    "LOG_AREA": "FP-relative surface area",
    "LOG_VOLUME_ICV": "FP-relative cortical volume",
}

# Corrected relative-trajectory figure: never place modalities on a common numerical axis.
relative_plot = pd.DataFrame(relative_rows)
figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for axis, modality in zip(axes, ["THICKNESS", "LOG_AREA", "LOG_VOLUME_ICV"]):
    row = relative_plot.loc[relative_plot["modality"].eq(modality)].iloc[0]
    axis.errorbar(
        [row["slope_per_5y"]], [0],
        xerr=[[row["slope_per_5y"] - row["ci_low"]], [row["ci_high"] - row["slope_per_5y"]]],
        fmt="D", capsize=4, color="#2f4b7c",
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks([])
    axis.set_title(modality_titles[modality])
    axis.set_xlabel(modality_labels[modality])
figure.suptitle("FP minus equal-weight mean of the other nine eligible networks", y=1.03)
figure.tight_layout()
figure.savefig(FIG_DIR / "structural_fp_relative_trajectories.png", dpi=220, bbox_inches="tight")
plt.close(figure)

# Supplementary ten-network context with a separate numeric axis for each modality.
display_order = [*ELIGIBLE_NETWORKS, "FP minus equal-weight mean(other 9)"]
figure, axes = plt.subplots(1, 3, figsize=(17, 7.2))
for axis, modality in zip(axes, ["THICKNESS", "LOG_AREA", "LOG_VOLUME_ICV"]):
    frame = structural_context_slopes.loc[structural_context_slopes["modality"].eq(modality)].copy()
    frame["network"] = pd.Categorical(frame["network"], categories=display_order, ordered=True)
    frame = frame.sort_values("network").reset_index(drop=True)
    positions = np.arange(len(frame))
    colors = [
        "#d45087" if str(label) == FP_NETWORK else
        "#2f4b7c" if str(label).startswith("FP minus") else "#8c8c8c"
        for label in frame["network"]
    ]
    for position, (_, row), color in zip(positions, frame.iterrows(), colors):
        marker = "D" if row["context_type"] == "FP_MINUS_REFERENCE" else "o"
        axis.errorbar(
            row["slope_per_5y"], position,
            xerr=[[row["slope_per_5y"] - row["ci_low"]], [row["ci_high"] - row["slope_per_5y"]]],
            fmt=marker, capsize=2.5, color=color,
        )
    axis.axvline(0, color="black", linewidth=.9)
    axis.set_yticks(positions)
    axis.set_yticklabels(frame["network"] if axis is axes[0] else [])
    axis.set_xlabel(modality_labels[modality])
    axis.set_title(modality_titles[modality].replace("FP-relative ", "Absolute "))
axes[0].invert_yaxis()
axes[1].invert_yaxis()
axes[2].invert_yaxis()
figure.suptitle("Absolute structural change across the ten eligible networks", y=1.01)
figure.tight_layout()
figure.savefig(FIG_DIR / "ten_network_absolute_structural_context.png", dpi=220, bbox_inches="tight")
plt.close(figure)

structural_figure_correction_audit = pd.DataFrame([{
    "overwritten_figure": "structural_fp_relative_trajectories.png",
    "replacement_panels": 3,
    "shared_unlabeled_numeric_axis": False,
    "thickness_axis": modality_labels["THICKNESS"],
    "area_axis": modality_labels["LOG_AREA"],
    "volume_axis": modality_labels["LOG_VOLUME_ICV"],
    "supplementary_context_figure": "ten_network_absolute_structural_context.png",
}])
display(structural_figure_correction_audit)


# Export tables and a compact run manifest.
outputs = OrderedDict([
    ("fmri_identity", fmri_identity),
    ("fmri_source_reconciliation", fmri_source_reconciliation),
    ("structural_source_audit", structural_source_audit),
    ("network_mapping_audit", network_mapping_audit),
    ("parcel_coverage_audit", parcel_coverage_audit),
    ("reference_weight_audit", reference_weight_audit),
    ("reference_construction_audit", reference_construction_audit),
    ("structural_cohort_summary", structural_cohort_summary),
    ("structural_model_summary", structural_model_summary),
    ("structural_tests", structural_tests),
    ("structural_slopes", structural_slopes),
    ("structural_site_slopes", structural_site_slopes),
    ("structural_component_audit", structural_component_audit),
    ("fmri_matched_cohort", fmri_matched_cohort),
    ("fmri_model_summary", fmri_model_summary),
    ("fmri_tests", fmri_tests),
    ("fmri_state_slopes", fmri_state_slopes),
    ("fmri_coefficient_changes", fmri_coefficient_changes),
    ("fmri_coefficient_change_joint", fmri_coefficient_change_joint),
    ("site_robustness_summary", site_robustness_summary),
    ("structural_sensitivity_summary", decision_table),
    ("structural_covariance_rank_eigen_diagnostics", structural_covariance_diagnostics),
    ("structural_covariance_eigenvalues", structural_covariance_eigenvalues),
    ("structural_fmri_component_coefficients", structural_fmri_component_coefficients),
    ("structural_fmri_component_joint_tests", structural_fmri_component_joint_tests),
    ("surface_area_definition_audit", area_definition_audit),
    ("surface_area_coverage_audit", area_coverage_audit),
    ("surface_area_cohort_summary", area_cohort_summary),
    ("surface_area_relative_model_summary", area_relative_model_summary),
    ("surface_area_relative_slope", area_relative_slope),
    ("surface_area_relative_tests", area_relative_tests),
    ("surface_area_relative_site_slopes", area_relative_site_slopes),
    ("surface_area_materiality_summary", area_branch_decision),
    ("surface_area_component_audit", area_component_audit),
    ("surface_area_fmri_matched_cohort", area_fmri_matched_cohort),
    ("surface_area_fmri_model_summary", area_fmri_model_summary),
    ("surface_area_fmri_tests", area_fmri_tests),
    ("surface_area_fmri_state_slopes", area_fmri_state_slopes),
    ("surface_area_fmri_coefficient_changes", area_fmri_coefficient_changes),
    ("surface_area_fmri_coefficient_change_joint", area_fmri_coefficient_change_joint),
    ("surface_area_fmri_component_coefficients", area_fmri_component_coefficients),
    ("surface_area_fmri_component_joint_tests", area_fmri_component_joint_tests),
    ("structural_absolute_model_summary", structural_absolute_model_summary),
    ("structural_absolute_slopes", structural_absolute_slopes),
    ("structural_absolute_tests", structural_absolute_tests),
    ("structural_context_slopes", structural_context_slopes),
    ("structural_figure_audit", structural_figure_correction_audit),
])
for filename, table in outputs.items():
    table.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

thickness_row = structural_slopes.loc[
    structural_slopes["scenario"].eq("THICKNESS_PRIMARY")
].iloc[0]
volume_row = structural_slopes.loc[
    structural_slopes["scenario"].eq("VOLUME_ICV_SECONDARY")
].iloc[0]
area_row = area_relative_slope.iloc[0]

run_manifest = {
    "script": "04_structural_sensitivity.py",
    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "participants": int(amplitude_fp_long["id"].nunique()),
    "eligible_networks": ELIGIBLE_NETWORKS,
    "fp_relative_definition": "FP minus equal-weight mean of the other nine networks",
    "thickness_definition": "surface-area-weighted network mean thickness",
    "volume_definition": "log network volume divided by estimated intracranial volume",
    "surface_area_definition": "log summed network parcel area",
    "thickness_slope_per_5y": float(thickness_row["slope_per_5y"]),
    "volume_slope_per_5y": float(volume_row["slope_per_5y"]),
    "surface_area_slope_per_5y": float(area_row["slope_per_5y"]),
    "surface_area_material": bool(AREA_MATERIAL),
    "surface_area_adjustment_performed": bool(AREA_MATERIAL),
    "coefficient_change_bootstrap_draws": COEFFICIENT_CHANGE_BOOTSTRAPS,
    "participant_slopes_extracted": False,
    "causal_mediation_claimed": False,
    "tables": [f"{name}.csv" for name in outputs],
    "figures": sorted(path.name for path in FIG_DIR.glob("*.png")),
}
(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2) + "\n"
)

print("\nSTRUCTURAL SENSITIVITY ANALYSIS COMPLETE")
print(f"Participants: {amplitude_fp_long['id'].nunique():,}")
print(f"Eligible networks: {len(ELIGIBLE_NETWORKS)}")
print(
    "FP-relative thickness: "
    f"{float(thickness_row['slope_per_5y']):+.6f} per 5 years "
    f"(p={float(thickness_row['p']):.6g})"
)
print(
    "FP-relative volume/eTIV: "
    f"{float(volume_row['slope_per_5y']):+.6f} per 5 years "
    f"(p={float(volume_row['p']):.6g})"
)
print(
    "FP-relative surface area: "
    f"{float(area_row['slope_per_5y']):+.6f} per 5 years "
    f"(p={float(area_row['p']):.6g})"
)
print(f"Surface-area adjustment performed: {bool(AREA_MATERIAL)}")
print(f"Outputs: {OUTPUT_DIR}")
