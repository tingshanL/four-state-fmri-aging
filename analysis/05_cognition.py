#!/usr/bin/env python3
"""Analyze cognition coupling and selection in the repeat cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.stats import chi2, norm

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def display(value):
    """Compact replacement for interactive table display."""
    print(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Local multimodal project root containing fMRI and non-imaging files.",
    )
    parser.add_argument(
        "--subject-table", type=Path, required=True,
        help="Longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Local results root used by scripts 03 and 05.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development run with 999 variance-bootstrap draws.",
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Ignore a compatible cognition-model checkpoint and refit the model.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=None,
        help="Override variance-bootstrap draws (paper run default: 4999).",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
SUBJECT_TABLE = ARGS.subject_table.expanduser().resolve()
RESULTS_ROOT = ARGS.results_dir.expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "05_cognition"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_FP_OUTPUT_DIR = RESULTS_ROOT / "03_fp_specificity_and_robustness"
SOURCE_FP_CHECKPOINT_DIR = SOURCE_FP_OUTPUT_DIR / "checkpoints"

if not DATA_DIR.is_dir():
    raise NotADirectoryError(DATA_DIR)
if not SUBJECT_TABLE.is_file():
    raise FileNotFoundError(SUBJECT_TABLE)
if not SOURCE_FP_OUTPUT_DIR.is_dir():
    raise FileNotFoundError(f"Script 03 results are missing: {SOURCE_FP_OUTPUT_DIR}")

FAST_MODE = bool(ARGS.fast)
FORCE_REFIT = bool(ARGS.force_refit)
MIN_STATE_PARTICIPANTS = 12 if FAST_MODE else 40
EXPECTED_REPEAT_PARTICIPANTS = 852
EXPECTED_COGNITION_PARTICIPANTS = 746
EXPECTED_COGNITION_VISITS = 1881
EXPECTED_ELIGIBLE_NETWORK_N = 10
MIN_PARCELS = 2
MIN_STATE_COVERAGE = .90
MIN_STATE_POSITIVE = .99
BOOT_B_VARIANCE = int(
    ARGS.bootstrap_b if ARGS.bootstrap_b is not None else (999 if FAST_MODE else 4999)
)
BOOT_SEED_VARIANCE = 20260907

COGNITION_COLUMN = "FluidIQ_Tr35_60y"
COGNITION_LABEL = "Fluid Cognition Composite"
COGNITION_DIRECTION = "HIGHER_IS_BETTER"
COUPLING_STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
STATES = COUPLING_STATES.copy()
FP_NETWORK = "Frontoparietal"
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
        raise FileNotFoundError("The fmri_node_parcel_manifest.csv is required")
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

# Reproduce the DMN/FP repeat cohort before testing other networks.
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
if not FAST_MODE and profile_data["id"].nunique() != EXPECTED_REPEAT_PARTICIPANTS:
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
        "IDEVENT", "SUBJECTEVENT", "SUBJECTID", "SUBJECT", "ID", "PARTICIPANTID", "X"
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

# Only the amplitude-standardized representation is required here.

def analysis_frame(frame, profile_suffix=""):
    out = frame.copy()
    out["outcome_G"] = out[f"G{profile_suffix}"]
    for network in ELIGIBLE_NETWORKS:
        out[f"profile__{network}"] = out[f"R{profile_suffix}__{network}"]
    return out

FRAME_REGISTRY = {
    "AMPLITUDE_STANDARDIZED": analysis_frame(amplitude_standardized, "_STD"),
}

amplitude_coverage = pd.DataFrame([{
    "network": network,
    "rows": len(profile_data),
    "positive_pair_amplitude_rows": int((
        pd.to_numeric(
            profile_data.merge(
                amplitude_wide,
                on=["id_event", "state"],
                how="left",
            )[f"P__{network}"],
            errors="coerce",
        ) > 0
    ).sum()),
} for network in ELIGIBLE_NETWORKS])
amplitude_coverage["positive_coverage"] = (
    amplitude_coverage["positive_pair_amplitude_rows"]
    / amplitude_coverage["rows"]
)

display(amplitude_coverage)



# Reconcile the fMRI outcome with script 03


if FP_NETWORK not in ELIGIBLE_NETWORKS or K != EXPECTED_ELIGIBLE_NETWORK_N:
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
    out["Y_FP_REL_FROM_PROFILE"] = (K / (K - 1.0)) * pd.to_numeric(
        out[f"R{suffix}__{FP_NETWORK}"], errors="coerce"
    ).astype(np.float64)
    return out


amplitude_fp_long = add_fp_relative(FRAME_REGISTRY["AMPLITUDE_STANDARDIZED"])
identity_error = (
    amplitude_fp_long["Y_FP_REL"] - amplitude_fp_long["Y_FP_REL_FROM_PROFILE"]
).to_numpy(dtype=np.float64)
imaging_identity = pd.DataFrame([{
    "identity": "Y_FP_REL_STD = L_STD_FP - mean(other nine L_STD) = (10/9) R_STD_FP",
    "participants": int(amplitude_fp_long["id"].nunique()),
    "sessions": int(amplitude_fp_long["id_event"].nunique()),
    "session_state_rows": int(len(amplitude_fp_long)),
    "max_absolute_error": float(np.nanmax(np.abs(identity_error))),
    "tolerance": 1e-10,
    "pass": bool(np.nanmax(np.abs(identity_error)) <= 1e-10),
}])
if not imaging_identity["pass"].all():
    raise AssertionError("Amplitude-standardized FP-relative identity failed")
if not FAST_MODE and amplitude_fp_long["id"].nunique() != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Expected {EXPECTED_REPEAT_PARTICIPANTS} imaging participants; "
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
imaging_source_reconciliation = pd.DataFrame([{
    "source_script": source_fp_manifest.get("script"),
    "saved_participants": source_fp_manifest.get("participants"),
    "saved_network_n": len(source_fp_manifest.get("eligible_networks", [])),
    "fingerprint_match": source_fp_checkpoint.get("fingerprint") == current_fingerprint,
    "eligible_network_order_match": (
        source_fp_manifest.get("eligible_networks") == ELIGIBLE_NETWORKS
    ),
}])
if not imaging_source_reconciliation[
    ["fingerprint_match", "eligible_network_order_match"]
].all(axis=None):
    print(imaging_source_reconciliation)
    raise AssertionError("Imaging rows do not match script 03 outputs")

display(imaging_identity)
display(imaging_source_reconciliation)



# Locate the Fluid Cognition Composite


def cognition_candidate_paths():
    paths = []
    patterns = [
        "AABC_Release2_Non-imaging_Data-XL.csv",
        "*Non-imaging*Data*.csv",
        "*AABC2*subjects*.csv",
        "AABC2_subjects.csv",
    ]
    for pattern in patterns:
        for path in DATA_DIR.rglob(pattern):
            path = path.resolve()
            if OUTPUT_DIR in path.parents or SOURCE_FP_OUTPUT_DIR in path.parents:
                continue
            paths.append(path)
    return list(dict.fromkeys(paths))


def endpoint_id_event(table):
    exact = [
        column for column in table
        if compact_token(column) in {"IDEVENT", "SUBJECTEVENT", "PARTICIPANTEVENT"}
    ]
    if len(exact) == 1:
        return table[exact[0]].astype(str).str.strip()
    x_columns = [column for column in table if compact_token(column) == "X"]
    if len(x_columns) == 1:
        values = table[x_columns[0]].astype(str).str.strip()
        if values.str.contains(r"_(?:V|VISIT|SESSION)?\d+$", case=False, regex=True).mean() > .8:
            return values
    participant = pick_column(
        table, ["id", "subject_id", "participant_id", "src_subject_id", "subjectkey"]
    )
    event = pick_column(table, ["eventname", "event_name", "event", "visit", "session"])
    if participant is None or event is None:
        raise KeyError(f"Cannot construct id_event; columns begin {list(table.columns)[:15]}")
    return (
        table[participant].astype(str).str.strip()
        + "_" + table[event].astype(str).str.strip()
    )


endpoint_sources = []
endpoint_tables = {}
for path in cognition_candidate_paths():
    header_row = None
    source_column = None
    for candidate_header_row in (0, 1, 2):
        try:
            header = pd.read_csv(path, header=candidate_header_row, nrows=0)
        except Exception:
            continue
        normalized = {
            str(column).replace("\ufeff", "").strip(): column
            for column in header.columns
        }
        if COGNITION_COLUMN in normalized:
            header_row = candidate_header_row
            source_column = normalized[COGNITION_COLUMN]
            break
    if header_row is None:
        continue
    table = pd.read_csv(path, header=header_row)
    if source_column != COGNITION_COLUMN:
        table = table.rename(columns={source_column: COGNITION_COLUMN})
    table["id_event"] = endpoint_id_event(table)
    table["cognition"] = pd.to_numeric(table[COGNITION_COLUMN], errors="coerce")
    conflicts = table.loc[table["cognition"].notna()].groupby("id_event")["cognition"].nunique()
    if (conflicts > 1).any():
        raise AssertionError(f"Conflicting cognition values within {path}")
    compact = table[["id_event", "cognition"]].drop_duplicates("id_event").copy()
    endpoint_tables[path] = compact
    endpoint_sources.append({
        "path": str(path),
        "sha256": sha256_file(path),
        "header_row_excel": int(header_row + 1),
        "rows": len(compact),
        "nonmissing_cognition": int(compact["cognition"].notna().sum()),
        "official_release_priority": int("Non-imaging_Data-XL" in path.name),
    })

if not endpoint_sources:
    raise FileNotFoundError(
        f"No CSV under {DATA_DIR} contains the endpoint {COGNITION_COLUMN}."
    )
cognition_source_audit = pd.DataFrame(endpoint_sources).sort_values(
    ["official_release_priority", "nonmissing_cognition", "path"],
    ascending=[False, False, True],
).reset_index(drop=True)
selected_cognition_path = Path(cognition_source_audit.loc[0, "path"])
cognition_table = endpoint_tables[selected_cognition_path].copy()

agreement_rows = []
for path, table in endpoint_tables.items():
    if path == selected_cognition_path:
        continue
    overlap = cognition_table.merge(
        table, on="id_event", how="inner", suffixes=("_selected", "_other")
    )
    valid = overlap[["cognition_selected", "cognition_other"]].notna().all(axis=1)
    maximum_difference = float(np.max(np.abs(
        overlap.loc[valid, "cognition_selected"]
        - overlap.loc[valid, "cognition_other"]
    ))) if valid.any() else np.nan
    agreement_rows.append({
        "selected_path": str(selected_cognition_path),
        "comparison_path": str(path),
        "overlap_nonmissing": int(valid.sum()),
        "maximum_absolute_difference": maximum_difference,
        "agreement_pass": bool(not valid.any() or maximum_difference <= 1e-10),
    })
cognition_source_agreement = pd.DataFrame(agreement_rows)
if cognition_source_agreement.empty:
    cognition_source_agreement = pd.DataFrame([{
        "selected_path": str(selected_cognition_path),
        "comparison_path": "",
        "overlap_nonmissing": 0,
        "maximum_absolute_difference": np.nan,
        "agreement_pass": True,
    }])
if len(cognition_source_agreement) and not cognition_source_agreement["agreement_pass"].all():
    raise AssertionError("Available sources disagree on the cognition endpoint")

cognition_table = cognition_table.merge(
    events[["id_event", "id"]].drop_duplicates(),
    on="id_event", how="left", validate="one_to_one",
)
full_cognition = cognition_table.loc[
    cognition_table["cognition"].notna() & cognition_table["id"].notna()
].copy()
if full_cognition.empty:
    raise AssertionError(f"No valid {COGNITION_COLUMN} observations matched the event table")
cognition_endpoint_audit = pd.DataFrame([{
    "endpoint_label": COGNITION_LABEL,
    "release_column": COGNITION_COLUMN,
    "direction": COGNITION_DIRECTION,
    "selected_source": str(selected_cognition_path),
    "selected_header_row_excel": int(cognition_source_audit.loc[0, "header_row_excel"]),
    "participants": int(full_cognition["id"].nunique()),
    "observations": int(len(full_cognition)),
}])

display(cognition_source_audit)
display(cognition_source_agreement)
display(cognition_endpoint_audit)



# Form the same-event cognition sample

key_covariates = [
    "id", "pedigree_group", "study", "sex", "site", "scanner", "site_scanner",
    "event_age", "baseline_age", "age0_c5", "time_years", "t5", "visit_number",
]
for column in key_covariates:
    conflict = amplitude_fp_long.groupby("id_event")[column].nunique(dropna=False)
    if (conflict > 1).any():
        raise AssertionError(f"{column} differs across states within an imaging session")

imaging_covariates = amplitude_fp_long[["id_event", *key_covariates]].drop_duplicates("id_event")
imaging_wide = amplitude_fp_long.pivot(index="id_event", columns="state", values="Y_FP_REL")
imaging_wide = imaging_wide.reindex(columns=COUPLING_STATES)
imaging_wide.columns = [f"fp_rel__{state}" for state in COUPLING_STATES]
imaging_wide = imaging_wide.reset_index().merge(
    imaging_covariates, on="id_event", how="left", validate="one_to_one"
)
four_state_complete = np.isfinite(imaging_wide[[f"fp_rel__{state}" for state in COUPLING_STATES]].to_numpy(float)).all(axis=1)
imaging_wide = imaging_wide.loc[four_state_complete].copy()

paired_all = imaging_wide.merge(
    cognition_table[["id_event", "cognition"]], on="id_event", how="left", validate="one_to_one"
)
paired_all["cognition_available"] = paired_all["cognition"].notna()
core_complete = np.isfinite(paired_all[["cognition", "age0_c5", "t5"]].to_numpy(float)).all(axis=1)
paired_candidate = paired_all.loc[core_complete].copy()
paired_visit_n = paired_candidate.groupby("id")["id_event"].nunique()
analysis_ids = set(paired_visit_n[paired_visit_n.ge(2)].index)
analysis = paired_candidate.loc[paired_candidate["id"].isin(analysis_ids)].copy()
analysis = analysis.sort_values(["id", "time_years", "id_event"]).reset_index(drop=True)
if analysis.empty:
    raise AssertionError("No participants have at least two paired fMRI-cognition visits")
if not set(analysis["id"].astype(str)).issubset(set(amplitude_fp_long["id"].astype(str))):
    raise AssertionError("The cognition cohort is not a subset of the imaging cohort")
if analysis.groupby("id")["id_event"].nunique().min() < 2:
    raise AssertionError("The primary cohort contains a participant with fewer than two paired visits")

for state in COUPLING_STATES:
    raw = f"fp_rel__{state}"
    between = f"fp_rel_between__{state}"
    within = f"fp_rel_within__{state}"
    analysis[between] = analysis.groupby("id")[raw].transform("mean")
    analysis[within] = analysis[raw] - analysis[between]

decomposition_rows = []
for state in COUPLING_STATES:
    reconstructed = analysis[f"fp_rel_between__{state}"] + analysis[f"fp_rel_within__{state}"]
    participant_within_means = analysis.groupby("id")[f"fp_rel_within__{state}"].mean()
    decomposition_rows.append({
        "state": state,
        "max_reconstruction_error": float(np.max(np.abs(reconstructed - analysis[f"fp_rel__{state}"]))),
        "max_participant_within_mean_error": float(np.max(np.abs(participant_within_means))),
        "tolerance": 1e-12,
    })
brain_decomposition_audit = pd.DataFrame(decomposition_rows)
brain_decomposition_audit["pass"] = (
    brain_decomposition_audit[["max_reconstruction_error", "max_participant_within_mean_error"]].max(axis=1) <= 1e-12
)
if not brain_decomposition_audit["pass"].all():
    display(brain_decomposition_audit)
    raise AssertionError("Within-person/between-person decomposition failed")

cohort_summary = pd.DataFrame([
    {
        "stage": "AMPLITUDE_STANDARDIZED_IMAGING_COHORT",
        "participants": int(amplitude_fp_long["id"].nunique()),
        "visits": int(amplitude_fp_long["id_event"].nunique()),
    },
    {
        "stage": "FOUR_STATE_COMPLETE_IMAGING_VISITS",
        "participants": int(imaging_wide["id"].nunique()), "visits": int(len(imaging_wide)),
    },
    {
        "stage": "SAME_EVENT_COGNITION_AVAILABLE",
        "participants": int(paired_candidate["id"].nunique()), "visits": int(len(paired_candidate)),
    },
    {
        "stage": "PRIMARY_TWO_PLUS_PAIRED_VISITS",
        "participants": int(analysis["id"].nunique()), "visits": int(len(analysis)),
    },
])
followup_summary = pd.DataFrame([{
    "participants": int(analysis["id"].nunique()),
    "participants_2plus": int((analysis.groupby("id")["id_event"].nunique() >= 2).sum()),
    "participants_3plus": int((analysis.groupby("id")["id_event"].nunique() >= 3).sum()),
    "median_followup_years": float(analysis.groupby("id")["time_years"].max().median()),
    "followup_q10": float(analysis.groupby("id")["time_years"].max().quantile(.10)),
    "followup_q90": float(analysis.groupby("id")["time_years"].max().quantile(.90)),
}])
exclusion_summary = pd.DataFrame([{
    "imaging_parent_participants": int(imaging_wide["id"].nunique()),
    "participants_with_any_cognition": int(paired_candidate["id"].nunique()),
    "participants_excluded_for_fewer_than_two_paired_visits": int(
        paired_candidate["id"].nunique() - analysis["id"].nunique()
    ),
    "imaging_visits_without_cognition": int((~paired_all["cognition_available"]).sum()),
}])

display(cohort_summary)
display(followup_summary)
display(exclusion_summary)
display(brain_decomposition_audit)



# Fit the within-/between-person cognition model

def stable_rank(matrix):
    matrix = np.asarray(matrix, float)
    if not np.isfinite(matrix).all():
        return -1
    scale = np.sqrt(np.mean(matrix ** 2, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    _, r = qr(matrix / scale, mode="economic", pivoting=False, check_finite=False)
    diagonal = np.abs(np.diag(r))
    if not len(diagonal):
        return 0
    tolerance = max(matrix.shape) * np.finfo(float).eps * max(diagonal.max(), 1.0)
    return int(np.sum(diagonal > tolerance))


class CognitionCouplingDesign:
    def fit(self, frame):
        self.sex_levels = sorted(frame["sex"].astype("string").fillna("MISSING").unique())
        self.study_levels = sorted(frame["study"].astype("string").fillna("MISSING").unique())
        self.acq_levels = sorted(frame["site_scanner"].astype("string").fillna("MISSING").unique())
        self.site_levels = sorted(frame["site"].astype("string").fillna("MISSING").unique())
        raw, names = self._raw(frame)
        if not np.isfinite(raw).all():
            bad = [names[j] for j in np.flatnonzero(~np.isfinite(raw).all(axis=0))]
            raise ValueError(f"Nonfinite cognition-design columns: {bad}")
        mandatory = ["Intercept", "age0_c5", "t5"]
        mandatory += [f"t5:site_sum[{site}]" for site in self.site_levels[:-1]]
        mandatory += [f"fp_rel_between__{state}" for state in COUPLING_STATES]
        mandatory += [f"fp_rel_within__{state}" for state in COUPLING_STATES]
        mandatory_index = [names.index(name) for name in mandatory]
        keep, rank = [], 0
        for j in mandatory_index + [k for k in range(raw.shape[1]) if k not in mandatory_index]:
            new_rank = stable_rank(raw[:, keep + [j]])
            if new_rank > rank:
                keep.append(j); rank = new_rank
            elif j in mandatory_index:
                raise AssertionError(f"Mandatory cognition estimand is aliased: {names[j]}")
        self.keep = sorted(keep)
        self.names = [names[j] for j in self.keep]
        self.dropped_names = [names[j] for j in range(len(names)) if j not in self.keep]
        return self

    @staticmethod
    def dummy(series, levels, prefix):
        text = series.astype("string").fillna("MISSING")
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(text.eq(level).to_numpy(float)); names.append(f"{prefix}[{level}]")
        return arrays, names

    @staticmethod
    def sum_code(series, levels, prefix):
        text = series.astype(str)
        arrays, names = [], []
        for level in levels[:-1]:
            arrays.append(np.where(text == level, 1.0, np.where(text == levels[-1], -1.0, 0.0)))
            names.append(f"{prefix}[{level}]")
        return arrays, names

    def _raw(self, frame):
        n = len(frame)
        age = pd.to_numeric(frame["age0_c5"], errors="coerce").to_numpy(float)
        time = pd.to_numeric(frame["t5"], errors="coerce").to_numpy(float)
        arrays = [np.ones(n), age, time]
        names = ["Intercept", "age0_c5", "t5"]
        for column, levels, prefix in [
            ("sex", self.sex_levels, "sex"),
            ("study", self.study_levels, "study"),
            ("site_scanner", self.acq_levels, "site_scanner"),
        ]:
            values, labels = self.dummy(frame[column], levels, prefix)
            arrays.extend(values); names.extend(labels)
        site_arrays, _ = self.sum_code(frame["site"], self.site_levels, "site_sum")
        for site, values in zip(self.site_levels[:-1], site_arrays):
            arrays.append(time * values); names.append(f"t5:site_sum[{site}]")
        for component in ["between", "within"]:
            for state in COUPLING_STATES:
                name = f"fp_rel_{component}__{state}"
                arrays.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(float)); names.append(name)
        return np.column_stack(arrays), names

    def transform(self, frame):
        raw, _ = self._raw(frame)
        return raw[:, self.keep]

    def selector(self, names):
        rows = []
        for name in names:
            row = np.zeros(len(self.names))
            if name not in self.names:
                raise AssertionError(f"Required coefficient missing after design fit: {name}")
            row[self.names.index(name)] = 1.0
            rows.append(row)
        return np.vstack(rows)


def fit_random_intercept_pedigree_sandwich(x, y, participant, pedigree):
    x = np.asarray(x, float); y = np.asarray(y, float)
    participant = np.asarray(participant).astype(str); pedigree = np.asarray(pedigree).astype(str)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y, participant, pedigree = x[valid], y[valid], participant[valid], pedigree[valid]
    if len(y) <= x.shape[1] + 12:
        return None
    column_scale = np.sqrt(np.mean(x ** 2, axis=0))
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 1e-10), column_scale, 1.0)
    xs = x / column_scale
    blocks = []
    for pid in np.unique(participant):
        index = np.flatnonzero(participant == pid)
        blocks.append((index, np.ones((len(index), len(index))), str(pedigree[index[0]])))

    def profile(theta):
        tau = float(np.expm1(theta[0]))
        xtx = np.zeros((xs.shape[1], xs.shape[1])); xty = np.zeros(xs.shape[1])
        yty = 0.0; logdet_v = 0.0
        for index, kernel, _ in blocks:
            a = np.eye(len(index)) + tau * kernel
            sign, logdet = np.linalg.slogdet(a)
            if sign <= 0:
                return {"objective": np.inf}
            try:
                ax = np.linalg.solve(a, xs[index]); ay = np.linalg.solve(a, y[index])
            except np.linalg.LinAlgError:
                return {"objective": np.inf}
            xtx += xs[index].T @ ax; xty += xs[index].T @ ay
            yty += float(y[index] @ ay); logdet_v += logdet
        sign, logdet_x = np.linalg.slogdet(xtx)
        if sign <= 0:
            return {"objective": np.inf}
        beta = np.linalg.pinv(xtx, rcond=1e-11) @ xty
        residual_sum = float(yty - 2 * beta @ xty + beta @ xtx @ beta)
        df = len(y) - len(beta)
        if residual_sum <= 0 or df <= 0:
            return {"objective": np.inf}
        sigma_e2 = residual_sum / df
        objective = .5 * (logdet_v + df * np.log(sigma_e2) + logdet_x)
        return {"objective": objective, "coef": beta, "xtx": xtx, "sigma_e2": sigma_e2}

    candidates = []
    for start in [0.0, np.log1p(.2), np.log1p(1.0), np.log1p(5.0)]:
        optimization = minimize(
            lambda theta: profile(theta)["objective"], np.array([start]), method="L-BFGS-B",
            bounds=[(0, np.log1p(1e5))], options={"maxiter": 200, "ftol": 1e-10},
        )
        fitted = profile(optimization.x)
        if np.isfinite(fitted.get("objective", np.inf)):
            candidates.append((optimization.x, fitted, optimization))
    if not candidates:
        return None
    theta, fitted, selected = min(candidates, key=lambda item: item[1]["objective"])
    tau = float(np.expm1(theta[0])); beta_scaled = fitted["coef"]
    residual = y - xs @ beta_scaled
    bread = np.linalg.pinv(fitted["xtx"], rcond=1e-11)
    pedigree_scores = {}
    for index, kernel, family in blocks:
        a = np.eye(len(index)) + tau * kernel
        score = xs[index].T @ np.linalg.solve(a, residual[index])
        pedigree_scores[family] = pedigree_scores.get(family, np.zeros_like(score)) + score
    family_n = len(pedigree_scores)
    correction = (family_n / max(family_n - 1, 1)) * ((len(y) - 1) / max(len(y) - len(beta_scaled), 1))
    meat = sum(np.outer(score, score) for score in pedigree_scores.values())
    covariance_scaled = correction * bread @ meat @ bread
    beta = beta_scaled / column_scale
    covariance = covariance_scaled / np.outer(column_scale, column_scale)
    return {
        "coef": beta, "robust_cov": covariance,
        "sigma_e2": fitted["sigma_e2"], "tau_participant": tau,
        "objective": fitted["objective"], "row_n": len(y),
        "participant_n": len(np.unique(participant)), "pedigree_n": family_n,
        "rank": int(np.linalg.matrix_rank(fitted["xtx"])),
        "optimizer_success": bool(selected.success),
        "optimizer_nfev": int(getattr(selected, "nfev", 0)),
    }


def wald_matrix_local(beta, covariance, matrix):
    matrix = np.asarray(matrix, float)
    value = matrix @ np.asarray(beta); variance = matrix @ np.asarray(covariance) @ matrix.T
    rank = int(np.linalg.matrix_rank(variance))
    if rank == 0:
        return np.nan, 0, np.nan
    statistic = float(value @ np.linalg.pinv(variance, rcond=1e-11) @ value)
    return statistic, rank, float(chi2.sf(statistic, rank))


def cognition_model_fingerprint(frame):
    columns = [
        "id", "id_event", "pedigree_group", "age0_c5", "t5", "sex", "study", "site", "site_scanner", "cognition",
        *[f"fp_rel_between__{state}" for state in COUPLING_STATES],
        *[f"fp_rel_within__{state}" for state in COUPLING_STATES],
    ]
    payload = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=True).to_numpy().tobytes()
    marker = {
        "endpoint": COGNITION_COLUMN, "states": COUPLING_STATES,
        "model": "ONE_STAGE_WITHIN_BETWEEN_RANDOM_INTERCEPT_PEDIGREE_SANDWICH",
    }
    return hashlib.sha256(payload + json.dumps(marker, sort_keys=True).encode()).hexdigest()


def jsonable(value):
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.bool_): return bool(value)
    raise TypeError(type(value).__name__)


model_checkpoint = CHECKPOINT_DIR / "fluid_cognition_fp_relative_coupling.json"
model_fingerprint = cognition_model_fingerprint(analysis)
if model_checkpoint.exists() and not FORCE_REFIT:
    saved = json.loads(model_checkpoint.read_text())
    model_result = saved["result"] if saved.get("fingerprint") == model_fingerprint else None
else:
    model_result = None

if model_result is None:
    design = CognitionCouplingDesign().fit(analysis)
    x = design.transform(analysis)
    fit = fit_random_intercept_pedigree_sandwich(
        x, analysis["cognition"], analysis["id"], analysis["pedigree_group"]
    )
    if fit is None:
        raise RuntimeError("One-stage cognition coupling model did not return a fit")
    beta, covariance = fit["coef"], fit["robust_cov"]
    within_matrix = design.selector([f"fp_rel_within__{state}" for state in COUPLING_STATES])
    between_matrix = design.selector([f"fp_rel_between__{state}" for state in COUPLING_STATES])
    heterogeneity_matrix = within_matrix[:3] - within_matrix[3]
    test_rows = []
    for test, matrix, role in [
        ("FOUR_WITHIN_PERSON_COUPLINGS_ZERO", within_matrix, "PRIMARY"),
        ("FOUR_BETWEEN_PERSON_ASSOCIATIONS_ZERO", between_matrix, "SECONDARY_CONTEXT"),
        ("WITHIN_PERSON_COUPLINGS_EQUAL_ACROSS_STATES", heterogeneity_matrix, "SECONDARY_NO_STATE_SELECTION"),
    ]:
        statistic, df, p = wald_matrix_local(beta, covariance, matrix)
        test_rows.append({"test": test, "role": role, "chi2": statistic, "df": df, "p": p})
    if test_rows[0]["df"] != 4 or test_rows[2]["df"] != 3:
        raise AssertionError(f"Required joint tests lost rank: {test_rows}")

    coefficient_rows = []
    for index, name in enumerate(design.names):
        se = math.sqrt(max(0.0, float(covariance[index, index])))
        estimate = float(beta[index]); z = estimate / se if se > 0 else np.nan
        coefficient_rows.append({
            "term": name, "estimate": estimate, "se": se, "z": z,
            "p_unadjusted": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        })
    q4 = norm.ppf(1 - .05 / 8)
    coupling_rows = []
    for component, matrix in [("WITHIN_PERSON", within_matrix), ("BETWEEN_PERSON", between_matrix)]:
        estimates = matrix @ beta
        component_covariance = matrix @ covariance @ matrix.T
        se = np.sqrt(np.maximum(0, np.diag(component_covariance)))
        p_values = 2 * norm.sf(np.abs(estimates / se))
        for state_index, state in enumerate(COUPLING_STATES):
            coupling_rows.append({
                "component": component, "state": state,
                "coefficient_per_1_log_unit": float(estimates[state_index]),
                "coefficient_per_0_1_log_unit": float(.1 * estimates[state_index]),
                "se": float(se[state_index]),
                "p_unadjusted_descriptive": float(p_values[state_index]),
                "simultaneous_ci_low": float(estimates[state_index] - q4 * se[state_index]),
                "simultaneous_ci_high": float(estimates[state_index] + q4 * se[state_index]),
                "interval_family": "Bonferroni across four states within component",
                "state_selected_as_best": False,
            })
    summary = {
        "fit_status": "PASS", "rows": fit["row_n"], "participants": fit["participant_n"],
        "pedigrees": fit["pedigree_n"], "design_columns": len(design.names),
        "design_rank": fit["rank"], "dropped_design_columns": json.dumps(design.dropped_names),
        "condition_number": float(np.linalg.cond(x)),
        "optimizer_success": fit["optimizer_success"], "optimizer_nfev": fit["optimizer_nfev"],
        "objective": fit["objective"], "sigma_e2": fit["sigma_e2"],
        "tau_participant": fit["tau_participant"],
        "robust_cov_rank": int(np.linalg.matrix_rank(covariance)),
        "robust_cov_min_eigenvalue": float(np.linalg.eigvalsh((covariance + covariance.T) / 2).min()),
        "participant_slopes_extracted": False, "posthoc_state_selection": False,
    }
    model_result = {
        "summary": summary, "tests": test_rows, "coefficients": coefficient_rows,
        "couplings": coupling_rows,
        "payload": {"coef": beta, "cov": covariance, "names": design.names},
    }
    model_checkpoint.write_text(json.dumps({"fingerprint": model_fingerprint, "result": model_result}, default=jsonable, indent=2) + "\n")

model_summary = pd.DataFrame([model_result["summary"]])
joint_tests = pd.DataFrame(model_result["tests"])
model_coefficients = pd.DataFrame(model_result["coefficients"])
brain_cognition_couplings = pd.DataFrame(model_result["couplings"])
if not model_summary.loc[0, "fit_status"] == "PASS":
    raise AssertionError("The cognition coupling fit failed")

robust_covariance = np.asarray(model_result["payload"]["cov"], dtype=float)
robust_covariance = (robust_covariance + robust_covariance.T) / 2
robust_covariance_eigenvalues = np.linalg.eigvalsh(robust_covariance)
robust_covariance_psd_tolerance = max(
    1e-10,
    float(np.max(np.abs(robust_covariance_eigenvalues))) * 1e-10,
)
numerical_stability_audit = pd.DataFrame([{
    "fit_status": model_summary.loc[0, "fit_status"],
    "design_columns": int(model_summary.loc[0, "design_columns"]),
    "design_rank": int(model_summary.loc[0, "design_rank"]),
    "condition_number": float(model_summary.loc[0, "condition_number"]),
    "condition_number_finite": bool(np.isfinite(model_summary.loc[0, "condition_number"])),
    "robust_cov_rank": int(model_summary.loc[0, "robust_cov_rank"]),
    "robust_cov_min_eigenvalue": float(model_summary.loc[0, "robust_cov_min_eigenvalue"]),
    "robust_cov_max_eigenvalue": float(robust_covariance_eigenvalues.max()),
    "robust_cov_psd_tolerance": float(robust_covariance_psd_tolerance),
    "robust_cov_psd_within_numerical_tolerance": bool(
        robust_covariance_eigenvalues.min() >= -robust_covariance_psd_tolerance
    ),
    "interpretation": "Small negative eigenvalues within tolerance are floating-point rounding, not negative variance",
}])
if not numerical_stability_audit.loc[0, "condition_number_finite"]:
    raise AssertionError("The fitted design condition number is nonfinite")
if not numerical_stability_audit.loc[0, "robust_cov_psd_within_numerical_tolerance"]:
    display(numerical_stability_audit)
    raise AssertionError("The pedigree-robust covariance is materially indefinite")

display(model_summary)
display(numerical_stability_audit)
display(joint_tests)
display(brain_cognition_couplings)


primary_test = joint_tests.loc[
    joint_tests["test"].eq("FOUR_WITHIN_PERSON_COUPLINGS_ZERO")
].iloc[0]
between_test = joint_tests.loc[
    joint_tests["test"].eq("FOUR_BETWEEN_PERSON_ASSOCIATIONS_ZERO")
].iloc[0]
heterogeneity_test = joint_tests.loc[
    joint_tests["test"].eq("WITHIN_PERSON_COUPLINGS_EQUAL_ACROSS_STATES")
].iloc[0]

if not FAST_MODE:
    if analysis["id"].nunique() != EXPECTED_COGNITION_PARTICIPANTS:
        raise AssertionError(
            f"Expected {EXPECTED_COGNITION_PARTICIPANTS} cognition participants; "
            f"found {analysis['id'].nunique()}"
        )
    if len(analysis) != EXPECTED_COGNITION_VISITS:
        raise AssertionError(
            f"Expected {EXPECTED_COGNITION_VISITS} paired visits; found {len(analysis)}"
        )



# Express coefficient precision on interpretable scales

analysis = analysis.copy()
analysis["cognition_between_observed"] = analysis.groupby("id")["cognition"].transform("mean")
analysis["cognition_within_observed"] = (
    analysis["cognition"] - analysis["cognition_between_observed"]
)
fluidiq_within_sd = float(analysis["cognition_within_observed"].std(ddof=1))
if not np.isfinite(fluidiq_within_sd) or fluidiq_within_sd <= 0:
    raise AssertionError("Observed within-person FluidIQ SD is invalid")

distribution_rows = []

def describe_scale(name, values, scale_role, state=""):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return {
        "measure": name,
        "state": state,
        "scale_role": scale_role,
        "n_rows": int(len(s)),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "q25": float(s.quantile(.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(.75)),
        "iqr": float(s.quantile(.75) - s.quantile(.25)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

distribution_rows.append(describe_scale(
    "FluidIQ",
    analysis["cognition"],
    "observed_paired_visit_values",
))
distribution_rows.append(describe_scale(
    "FluidIQ",
    analysis["cognition_within_observed"],
    "within_person_deviation",
))
for state in COUPLING_STATES:
    distribution_rows.append(describe_scale(
        "FP_relative",
        analysis[f"fp_rel__{state}"],
        "observed_paired_visit_values",
        state,
    ))
    distribution_rows.append(describe_scale(
        "FP_relative",
        analysis[f"fp_rel_within__{state}"],
        "within_person_deviation",
        state,
    ))

within_person_scale_distribution = pd.DataFrame(distribution_rows)

within_coefficients = (
    brain_cognition_couplings.loc[
        brain_cognition_couplings["component"].eq("WITHIN_PERSON")
    ]
    .set_index("state")
    .reindex(COUPLING_STATES)
    .reset_index()
)

precision_rows = []
for _, row in within_coefficients.iterrows():
    state = row["state"]
    x_sd = float(analysis[f"fp_rel_within__{state}"].std(ddof=1))
    if not np.isfinite(x_sd) or x_sd <= 0:
        raise AssertionError(f"Within-person FP-relative SD is invalid for {state}")

    beta_1 = float(row["coefficient_per_1_log_unit"])
    lo_1 = float(row["simultaneous_ci_low"])
    hi_1 = float(row["simultaneous_ci_high"])

    precision_rows.append({
        "state": state,
        "component": "WITHIN_PERSON",
        "joint_test_p": float(primary_test["p"]),
        "state_heterogeneity_p": float(heterogeneity_test["p"]),
        "estimate_fluidiq_points_per_0_1_log": 0.1 * beta_1,
        "ci_low_fluidiq_points_per_0_1_log": 0.1 * lo_1,
        "ci_high_fluidiq_points_per_0_1_log": 0.1 * hi_1,
        "fp_relative_within_person_sd": x_sd,
        "estimate_fluidiq_points_per_1_fp_within_sd": beta_1 * x_sd,
        "ci_low_fluidiq_points_per_1_fp_within_sd": lo_1 * x_sd,
        "ci_high_fluidiq_points_per_1_fp_within_sd": hi_1 * x_sd,
        "fluidiq_within_person_sd": fluidiq_within_sd,
        "estimate_standardized_using_observed_within_person_sds": (
            beta_1 * x_sd / fluidiq_within_sd
        ),
        "ci_low_standardized_using_observed_within_person_sds": (
            lo_1 * x_sd / fluidiq_within_sd
        ),
        "ci_high_standardized_using_observed_within_person_sds": (
            hi_1 * x_sd / fluidiq_within_sd
        ),
        "standardized_scale_label": (
            "standardized coefficient using observed within-person SDs; NOT correlation"
        ),
        "interval_family": row["interval_family"],
        "state_selected_as_best": False,
    })

within_person_precision = pd.DataFrame(precision_rows)

display(within_person_scale_distribution)
display(within_person_precision)



# Describe selection into the repeated-cognition sample

# Session inventory before the four-state-complete filter.
parent_session_state = (
    amplitude_fp_long[["id", "id_event", "state", "time_years"]]
    .drop_duplicates(["id", "id_event", "state"])
)
parent_session_inventory = (
    parent_session_state.groupby(["id", "id_event"], as_index=False)
    .agg(
        available_state_n=("state", "nunique"),
        session_time_years=("time_years", "first"),
    )
)
parent_session_inventory["four_state_complete"] = (
    parent_session_inventory["available_state_n"] == len(COUPLING_STATES)
)
parent_session_inventory = parent_session_inventory.merge(
    cognition_table[["id_event", "cognition"]],
    on="id_event",
    how="left",
    validate="one_to_one",
)
parent_session_inventory["same_event_fluidiq_available"] = (
    parent_session_inventory["cognition"].notna()
)
parent_session_inventory["valid_four_state_cognition_pair"] = (
    parent_session_inventory["four_state_complete"]
    & parent_session_inventory["same_event_fluidiq_available"]
)

# Core-complete valid pairs exactly reproduce paired_candidate.
core_pair_keys = set(
    zip(
        paired_candidate["id"].astype(str),
        paired_candidate["id_event"].astype(str),
    )
)
parent_session_inventory["core_complete_pair"] = [
    (str(i), str(e)) in core_pair_keys
    for i, e in zip(
        parent_session_inventory["id"],
        parent_session_inventory["id_event"],
    )
]

parent_ids = sorted(amplitude_fp_long["id"].astype(str).unique())
flow_rows = []
analysis_ids_str = {str(value) for value in analysis_ids}

for pid in parent_ids:
    part = parent_session_inventory.loc[
        parent_session_inventory["id"].astype(str).eq(pid)
    ].copy()
    valid_paired_n = int(part["core_complete_pair"].sum())
    four_state_n = int(part["four_state_complete"].sum())
    any_cog_n = int(part["same_event_fluidiq_available"].sum())
    cog_in_incomplete_n = int(
        (
            part["same_event_fluidiq_available"]
            & ~part["four_state_complete"]
        ).sum()
    )
    fourstate_cog_n = int(
        (
            part["same_event_fluidiq_available"]
            & part["four_state_complete"]
        ).sum()
    )
    core_loss_n = int(
        (
            part["same_event_fluidiq_available"]
            & part["four_state_complete"]
            & ~part["core_complete_pair"]
        ).sum()
    )

    if valid_paired_n >= 2:
        category = "INCLUDED"
    elif valid_paired_n == 1:
        category = "ONE_PAIRED_COGNITION_VISIT"
    else:
        category = "NO_PAIRED_COGNITION"

    technical_reasons = []
    if any_cog_n == 0:
        technical_reasons.append("NO_SAME_EVENT_FLUIDIQ_ON_IMAGING_PARENT_VISITS")
    if cog_in_incomplete_n > 0:
        technical_reasons.append(
            "FLUIDIQ_EXISTS_ON_VISIT_WITH_INCOMPLETE_FOUR_STATE_IMAGING"
        )
    if core_loss_n > 0:
        technical_reasons.append(
            "FOUR_STATE_FLUIDIQ_PAIR_LOST_FOR_CORE_AGE_TIME_COMPLETENESS"
        )
    if not technical_reasons:
        technical_reasons.append("NONE_ADDITIONAL")

    flow_rows.append({
        "id": pid,
        "selection_category": category,
        "imaging_parent_visits": int(part["id_event"].nunique()),
        "four_state_complete_imaging_visits": four_state_n,
        "same_event_fluidiq_visits_any_imaging_parent": any_cog_n,
        "same_event_fluidiq_on_four_state_complete_visits": fourstate_cog_n,
        "valid_core_paired_visits": valid_paired_n,
        "fluidiq_visits_on_incomplete_four_state_imaging": cog_in_incomplete_n,
        "four_state_fluidiq_pairs_lost_for_core_covariates": core_loss_n,
        "technical_reason_detail": ";".join(technical_reasons),
        "in_analysis_ids": pid in analysis_ids_str,
    })

cognition_selection_flow = pd.DataFrame(flow_rows)

if len(cognition_selection_flow) != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Selection flow has {len(cognition_selection_flow)} participants, expected 852"
    )
if cognition_selection_flow["id"].duplicated().any():
    raise AssertionError("Selection flow is not one row per participant")

included_ids_flow = set(
    cognition_selection_flow.loc[
        cognition_selection_flow["selection_category"].eq("INCLUDED"),
        "id",
    ]
)
if included_ids_flow != analysis_ids_str:
    raise AssertionError(
        "Cognition INCLUDED roster differs from the cognition analysis roster"
    )
if len(included_ids_flow) != EXPECTED_COGNITION_PARTICIPANTS:
    raise AssertionError(
        f"Cognition INCLUDED count is {len(included_ids_flow)}, expected 746"
    )

flow_summary = (
    cognition_selection_flow["selection_category"]
    .value_counts()
    .rename_axis("selection_category")
    .reset_index(name="participants")
)
flow_summary["percent_of_852"] = (
    100 * flow_summary["participants"] / EXPECTED_REPEAT_PARTICIPANTS
)

display(flow_summary)
display(
    cognition_selection_flow.loc[
        ~cognition_selection_flow["technical_reason_detail"].eq("NONE_ADDITIONAL")
    ].head(30)
)

fp_columns = [f"fp_rel__{state}" for state in COUPLING_STATES]
imaging_wide_audit = imaging_wide.sort_values(
    ["id", "time_years", "id_event"]
).copy()
imaging_wide_audit["fp_rel_equal_state_mean"] = (
    imaging_wide_audit[fp_columns].mean(axis=1)
)

participant_imaging = (
    imaging_wide_audit.groupby("id", as_index=False)
    .agg(
        baseline_age=("baseline_age", "first"),
        sex=("sex", "first"),
        study=("study", "first"),
        baseline_site_scanner=("site_scanner", "first"),
        n_eligible_fmri_visits=("id_event", "nunique"),
        first_time_years=("time_years", "min"),
        last_time_years=("time_years", "max"),
        first_equal_state_mean_fp_relative=("fp_rel_equal_state_mean", "first"),
        **{
            f"first_fp_relative_{state}": (f"fp_rel__{state}", "first")
            for state in COUPLING_STATES
        },
    )
)
participant_imaging["fmri_followup_years"] = (
    participant_imaging["last_time_years"]
    - participant_imaging["first_time_years"]
)
participant_imaging["id"] = participant_imaging["id"].astype(str)
participant_imaging = participant_imaging.merge(
    cognition_selection_flow[["id", "selection_category"]],
    on="id",
    how="right",
    validate="one_to_one",
)
participant_imaging["included"] = (
    participant_imaging["selection_category"].eq("INCLUDED")
)

if participant_imaging["included"].sum() != 746:
    raise AssertionError("Baseline selection audit does not recover 746 included participants")
if (~participant_imaging["included"]).sum() != 106:
    raise AssertionError("Baseline selection audit does not recover 106 excluded participants")

def continuous_smd_rows(frame, variables, group_col, group_a=True, group_b=False, placement_map=None):
    rows = []
    placement_map = placement_map or {}
    for variable in variables:
        a = pd.to_numeric(
            frame.loc[frame[group_col].eq(group_a), variable],
            errors="coerce",
        ).dropna()
        b = pd.to_numeric(
            frame.loc[frame[group_col].eq(group_b), variable],
            errors="coerce",
        ).dropna()
        va = float(a.var(ddof=1)) if len(a) > 1 else np.nan
        vb = float(b.var(ddof=1)) if len(b) > 1 else np.nan
        pooled = math.sqrt(max(0.0, (va + vb) / 2)) if np.isfinite(va + vb) else np.nan
        difference = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
        smd = difference / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
        rows.append({
            "variable": variable,
            "level": "",
            "variable_type": "CONTINUOUS",
            "included_n_nonmissing": int(len(a)),
            "excluded_n_nonmissing": int(len(b)),
            "included_value": float(a.mean()) if len(a) else np.nan,
            "excluded_value": float(b.mean()) if len(b) else np.nan,
            "value_definition": "mean",
            "raw_difference": difference,
            "standardized_difference": smd,
            "abs_standardized_difference": abs(smd) if np.isfinite(smd) else np.nan,
            "p_value": np.nan,
            "placement": placement_map.get(variable, "MAIN_AUDIT"),
        })
    return rows

def categorical_smd_rows(frame, variables, group_col, group_a=True, group_b=False, placement="MAIN_AUDIT"):
    rows = []
    for variable in variables:
        values = frame[variable].astype("string").fillna("MISSING")
        levels = sorted(values.unique().tolist())
        for level in levels:
            a = values.loc[frame[group_col].eq(group_a)].eq(level).astype(float)
            b = values.loc[frame[group_col].eq(group_b)].eq(level).astype(float)
            pa = float(a.mean()) if len(a) else np.nan
            pb = float(b.mean()) if len(b) else np.nan
            pooled = math.sqrt(max(0.0, (pa * (1-pa) + pb * (1-pb)) / 2))
            difference = pa - pb
            smd = difference / pooled if pooled > 0 else np.nan
            rows.append({
                "variable": variable,
                "level": str(level),
                "variable_type": "BINARY_INDICATOR_FOR_CATEGORY",
                "included_n_nonmissing": int(len(a)),
                "excluded_n_nonmissing": int(len(b)),
                "included_value": pa,
                "excluded_value": pb,
                "value_definition": "proportion",
                "raw_difference": difference,
                "standardized_difference": smd,
                "abs_standardized_difference": abs(smd) if np.isfinite(smd) else np.nan,
                "p_value": np.nan,
                "placement": placement,
            })
    return rows

continuous_variables = [
    "baseline_age",
    "n_eligible_fmri_visits",
    "fmri_followup_years",
    "first_equal_state_mean_fp_relative",
    *[f"first_fp_relative_{state}" for state in COUPLING_STATES],
]
placement_map = {
    **{f"first_fp_relative_{state}": "SUPPLEMENT_ONLY" for state in COUPLING_STATES}
}
balance_rows = continuous_smd_rows(
    participant_imaging,
    continuous_variables,
    "included",
    True,
    False,
    placement_map,
)
balance_rows += categorical_smd_rows(
    participant_imaging,
    ["sex", "study"],
    "included",
    True,
    False,
    placement="MAIN_AUDIT",
)
balance_rows += categorical_smd_rows(
    participant_imaging,
    ["baseline_site_scanner"],
    "included",
    True,
    False,
    placement="SUPPLEMENT_ONLY",
)

selection_balance_746_vs_106 = pd.DataFrame(balance_rows)
selection_balance_746_vs_106 = (
    selection_balance_746_vs_106
    .sort_values(
        ["placement", "abs_standardized_difference"],
        ascending=[True, False],
        na_position="last",
    )
    .reset_index(drop=True)
)

display(selection_balance_746_vs_106)

eligible_cognition_categories = {"INCLUDED", "ONE_PAIRED_COGNITION_VISIT"}
one_vs_two = participant_imaging.loc[
    participant_imaging["selection_category"].isin(eligible_cognition_categories)
].copy()
one_vs_two["two_plus_group"] = one_vs_two["selection_category"].eq("INCLUDED")

first_paired_cognition = (
    paired_candidate.sort_values(["id", "time_years", "id_event"])
    .groupby("id", as_index=False)
    .first()[["id", "cognition", "time_years", "id_event"]]
    .rename(columns={
        "cognition": "first_available_paired_fluidiq",
        "time_years": "first_paired_cognition_time_years",
        "id_event": "first_paired_cognition_id_event",
    })
)
first_paired_cognition["id"] = first_paired_cognition["id"].astype(str)
one_vs_two = one_vs_two.merge(
    first_paired_cognition,
    on="id",
    how="left",
    validate="one_to_one",
)

balance2_rows = continuous_smd_rows(
    one_vs_two,
    [
        "first_available_paired_fluidiq",
        "baseline_age",
        "n_eligible_fmri_visits",
        "fmri_followup_years",
        "first_equal_state_mean_fp_relative",
        *[f"first_fp_relative_{state}" for state in COUPLING_STATES],
    ],
    "two_plus_group",
    True,
    False,
    {
        **{f"first_fp_relative_{state}": "SUPPLEMENT_ONLY" for state in COUPLING_STATES}
    },
)
balance2_rows += categorical_smd_rows(
    one_vs_two,
    ["sex", "study"],
    "two_plus_group",
    True,
    False,
    placement="MAIN_AUDIT",
)
balance2_rows += categorical_smd_rows(
    one_vs_two,
    ["baseline_site_scanner"],
    "two_plus_group",
    True,
    False,
    placement="SUPPLEMENT_ONLY",
)

one_vs_two_plus_cognition_visit_balance = pd.DataFrame(balance2_rows)
one_vs_two_plus_cognition_visit_balance = (
    one_vs_two_plus_cognition_visit_balance
    .sort_values(
        ["placement", "abs_standardized_difference"],
        ascending=[True, False],
        na_position="last",
    )
    .reset_index(drop=True)
)

fluid_two = pd.to_numeric(
    one_vs_two.loc[
        one_vs_two["two_plus_group"],
        "first_available_paired_fluidiq",
    ],
    errors="coerce",
).dropna().to_numpy(float)
fluid_one = pd.to_numeric(
    one_vs_two.loc[
        ~one_vs_two["two_plus_group"],
        "first_available_paired_fluidiq",
    ],
    errors="coerce",
).dropna().to_numpy(float)

if len(fluid_one) < 2 or len(fluid_two) < 2:
    raise AssertionError(
        "Need at least two participants in both first-FluidIQ groups for variance audit"
    )

def summarize_array(label, values):
    values = np.asarray(values, float)
    q25, med, q75 = np.quantile(values, [.25, .5, .75])
    return {
        "record_type": "GROUP_SUMMARY",
        "group": label,
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "variance": float(np.var(values, ddof=1)),
        "q25": float(q25),
        "median": float(med),
        "q75": float(q75),
        "iqr": float(q75-q25),
        "variance_ratio_two_plus_over_one": np.nan,
        "variance_ratio_bootstrap_ci_low": np.nan,
        "variance_ratio_bootstrap_ci_high": np.nan,
        "bootstrap_B": 0,
    }

variance_ratio = float(
    np.var(fluid_two, ddof=1) / np.var(fluid_one, ddof=1)
)
rng = np.random.default_rng(BOOT_SEED_VARIANCE)
ratio_draws = np.empty(BOOT_B_VARIANCE, float)
for b in range(BOOT_B_VARIANCE):
    draw_two = rng.choice(fluid_two, size=len(fluid_two), replace=True)
    draw_one = rng.choice(fluid_one, size=len(fluid_one), replace=True)
    v_one = np.var(draw_one, ddof=1)
    ratio_draws[b] = (
        np.var(draw_two, ddof=1) / v_one
        if np.isfinite(v_one) and v_one > 0 else np.nan
    )
ratio_draws = ratio_draws[np.isfinite(ratio_draws)]
ratio_ci = (
    np.quantile(ratio_draws, [.025, .975])
    if len(ratio_draws) else np.array([np.nan, np.nan])
)

variance_rows = [
    summarize_array("TWO_PLUS_PAIRED_VISITS_INCLUDED", fluid_two),
    summarize_array("ONE_PAIRED_VISIT_EXCLUDED", fluid_one),
    {
        "record_type": "VARIANCE_RATIO",
        "group": "TWO_PLUS_OVER_ONE",
        "n": int(len(fluid_two) + len(fluid_one)),
        "mean": np.nan,
        "sd": np.nan,
        "variance": np.nan,
        "q25": np.nan,
        "median": np.nan,
        "q75": np.nan,
        "iqr": np.nan,
        "variance_ratio_two_plus_over_one": variance_ratio,
        "variance_ratio_bootstrap_ci_low": float(ratio_ci[0]),
        "variance_ratio_bootstrap_ci_high": float(ratio_ci[1]),
        "bootstrap_B": int(BOOT_B_VARIANCE),
    },
]
fluidiq_variance_audit = pd.DataFrame(variance_rows)

display(one_vs_two_plus_cognition_visit_balance)
display(fluidiq_variance_audit)


flow_counts = cognition_selection_flow["selection_category"].value_counts().to_dict()
max_abs_smd_746_106 = float(
    selection_balance_746_vs_106["abs_standardized_difference"].max()
)
max_abs_smd_one_two = float(
    one_vs_two_plus_cognition_visit_balance["abs_standardized_difference"].max()
)

cognition_summary = pd.DataFrame([
    {
        "result": "within_person_joint_test",
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "df": int(primary_test["df"]),
        "p": float(primary_test["p"]),
        "description": "Joint test of four contemporaneous within-person coefficients",
    },
    {
        "result": "within_person_state_heterogeneity",
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "df": int(heterogeneity_test["df"]),
        "p": float(heterogeneity_test["p"]),
        "description": "State heterogeneity of within-person coefficients",
    },
    {
        "result": "between_person_context_joint_test",
        "estimate": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "df": int(between_test["df"]),
        "p": float(between_test["p"]),
        "description": "Contextual joint test of four between-person coefficients",
    },
    {
        "result": "first_fluidiq_variance_ratio_two_plus_over_one",
        "estimate": float(variance_ratio),
        "ci_low": float(ratio_ci[0]),
        "ci_high": float(ratio_ci[1]),
        "df": np.nan,
        "p": np.nan,
        "description": "Descriptive selection audit",
    },
])

# Precision and selection figure.
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

plot_a = within_person_precision.set_index("state").reindex(COUPLING_STATES).reset_index()
y = np.arange(len(plot_a))
estimate = plot_a["estimate_standardized_using_observed_within_person_sds"].to_numpy(float)
lo = plot_a["ci_low_standardized_using_observed_within_person_sds"].to_numpy(float)
hi = plot_a["ci_high_standardized_using_observed_within_person_sds"].to_numpy(float)
axes[0].errorbar(
    estimate, y, xerr=np.vstack([estimate - lo, hi - estimate]), fmt="o", capsize=3
)
axes[0].axvline(0, lw=1, ls="--")
axes[0].set_yticks(y)
axes[0].set_yticklabels(COUPLING_STATES)
axes[0].set_xlabel("Standardized coefficient using observed within-person SDs\n(not correlation)")
axes[0].set_title("A. Within-person cognition coefficients")

plot_b = selection_balance_746_vs_106.loc[
    selection_balance_746_vs_106["placement"].eq("MAIN_AUDIT")
    & selection_balance_746_vs_106["standardized_difference"].notna()
].copy()
plot_b["label"] = np.where(
    plot_b["level"].astype(str).eq(""),
    plot_b["variable"],
    plot_b["variable"] + " = " + plot_b["level"].astype(str),
)
plot_b = plot_b.sort_values("standardized_difference").reset_index(drop=True)
y2 = np.arange(len(plot_b))
axes[1].plot(plot_b["standardized_difference"], y2, "o")
axes[1].axvline(0, lw=1, ls="--")
axes[1].set_yticks(y2)
axes[1].set_yticklabels(plot_b["label"], fontsize=8)
axes[1].set_xlabel("Standardized difference: included minus excluded")
axes[1].set_title("B. Selection into the repeated-cognition sample")

fig.tight_layout()
fig.savefig(FIG_DIR / "cognition_precision_and_selection.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG_DIR / "cognition_precision_and_selection.pdf", bbox_inches="tight")
plt.close(fig)

outputs = {
    "imaging_identity.csv": imaging_identity,
    "imaging_source_reconciliation.csv": imaging_source_reconciliation,
    "cognition_source_audit.csv": cognition_source_audit,
    "cognition_source_agreement.csv": cognition_source_agreement,
    "cognition_endpoint_audit.csv": cognition_endpoint_audit,
    "cohort_summary.csv": cohort_summary,
    "followup_summary.csv": followup_summary,
    "exclusion_summary.csv": exclusion_summary,
    "brain_decomposition_audit.csv": brain_decomposition_audit,
    "model_summary.csv": model_summary,
    "numerical_stability_audit.csv": numerical_stability_audit,
    "joint_tests.csv": joint_tests,
    "model_coefficients.csv": model_coefficients,
    "brain_cognition_couplings.csv": brain_cognition_couplings,
    "within_person_precision.csv": within_person_precision,
    "within_person_scale_distribution.csv": within_person_scale_distribution,
    "cognition_selection_flow.csv": cognition_selection_flow,
    "selection_flow_summary.csv": flow_summary,
    "selection_balance_746_vs_106.csv": selection_balance_746_vs_106,
    "one_vs_two_plus_cognition_visit_balance.csv": one_vs_two_plus_cognition_visit_balance,
    "fluidiq_variance_audit.csv": fluidiq_variance_audit,
    "cognition_summary.csv": cognition_summary,
}
for filename, table in outputs.items():
    table.to_csv(TABLE_DIR / filename, index=False)

run_manifest = {
    "script": "05_cognition.py",
    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "endpoint": COGNITION_COLUMN,
    "imaging_participants": int(amplitude_fp_long["id"].nunique()),
    "cognition_participants": int(analysis["id"].nunique()),
    "paired_visits": int(len(analysis)),
    "within_person_joint_p": float(primary_test["p"]),
    "within_person_state_heterogeneity_p": float(heterogeneity_test["p"]),
    "between_person_context_p": float(between_test["p"]),
    "selection_flow_counts": {str(k): int(v) for k, v in flow_counts.items()},
    "first_fluidiq_variance_ratio": float(variance_ratio),
    "first_fluidiq_variance_ratio_ci": [float(ratio_ci[0]), float(ratio_ci[1])],
    "variance_bootstrap_draws": int(BOOT_B_VARIANCE),
    "participant_slopes_extracted": False,
    "best_state_selected": False,
    "prediction_analysis_performed": False,
    "causal_mediation_claimed": False,
    "tables": list(outputs),
    "figures": [
        "cognition_precision_and_selection.png",
        "cognition_precision_and_selection.pdf",
    ],
}
(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, default=jsonable, indent=2) + "\n"
)

print("\nCOGNITION ANALYSIS COMPLETE")
print(f"Imaging participants: {amplitude_fp_long['id'].nunique():,}")
print(f"Cognition participants / paired visits: {analysis['id'].nunique():,} / {len(analysis):,}")
print(f"Within-person joint test: p={float(primary_test['p']):.6g}")
print(f"Within-person state heterogeneity: p={float(heterogeneity_test['p']):.6g}")
print(f"Between-person contextual test: p={float(between_test['p']):.6g}")
print(
    "Selection counts (included / one paired / none): "
    f"{flow_counts.get('INCLUDED', 0)} / "
    f"{flow_counts.get('ONE_PAIRED_COGNITION_VISIT', 0)} / "
    f"{flow_counts.get('NO_PAIRED_COGNITION', 0)}"
)
print(
    "First-FluidIQ variance ratio (two-plus / one): "
    f"{variance_ratio:.3f} [{float(ratio_ci[0]):.3f}, {float(ratio_ci[1]):.3f}]"
)
print(f"Outputs: {OUTPUT_DIR}")
