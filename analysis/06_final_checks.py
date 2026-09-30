#!/usr/bin/env python3
"""Run age, convergence, task-state, and partial-covariance checks."""

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
    """Compact replacement for interactive table display."""
    print(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Local multimodal project root containing the fMRI inputs.",
    )
    parser.add_argument(
        "--subject-table", type=Path, required=True,
        help="Longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Local results root used by scripts 02, 03, and 06.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development run with 999 convergence-bootstrap draws.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=None,
        help="Override convergence-bootstrap draws (paper run default: 4999).",
    )
    parser.add_argument(
        "--bootstrap-chunk", type=int, default=250,
        help="Number of multiplier draws processed per chunk.",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
SUBJECT_TABLE = ARGS.subject_table.expanduser().resolve()
RESULTS_ROOT = ARGS.results_dir.expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "06_final_checks"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PROFILE_OUTPUT_DIR = RESULTS_ROOT / "02_primary_profile_and_measurement"
SOURCE_FP_OUTPUT_DIR = RESULTS_ROOT / "03_fp_specificity_and_robustness"

if not DATA_DIR.is_dir():
    raise NotADirectoryError(DATA_DIR)
if not SUBJECT_TABLE.is_file():
    raise FileNotFoundError(SUBJECT_TABLE)
for dependency in [SOURCE_PROFILE_OUTPUT_DIR, SOURCE_FP_OUTPUT_DIR]:
    if not dependency.is_dir():
        raise FileNotFoundError(f"Required preceding results are missing: {dependency}")

FAST_MODE = bool(ARGS.fast)
MIN_STATE_PARTICIPANTS = 12 if FAST_MODE else 40
EXPECTED_REPEAT_PARTICIPANTS = 852
EXPECTED_ELIGIBLE_NETWORK_N = 10
MIN_PARCELS = 2
MIN_STATE_COVERAGE = .90
MIN_STATE_POSITIVE = .99
BOOTSTRAP_B = int(
    ARGS.bootstrap_b if ARGS.bootstrap_b is not None else (999 if FAST_MODE else 4999)
)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_CHUNK = int(ARGS.bootstrap_chunk)
SOURCE_BETA_TOL = 1e-7
SOURCE_COV_TOL = 2e-5
SOURCE_BASIS_TOL = 1e-10

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
FP_NETWORK = "Frontoparietal"

pd.set_option("display.max_columns", 260)
sns.set_theme(style="whitegrid", context="paper")



# Construct native and amplitude-standardized profile data

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

# Construct the native and amplitude-standardized analysis frames.

def analysis_frame(frame, profile_suffix=""):
    out = frame.copy()
    out["outcome_G"] = out[f"G{profile_suffix}"]
    for network in ELIGIBLE_NETWORKS:
        out[f"profile__{network}"] = out[f"R{profile_suffix}__{network}"]
    return out

FRAME_REGISTRY = {
    "FULL_PRIMARY": analysis_frame(profile_data),
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
        # Already-computed pedigree influence contributions are exposed here
        # solely for multiplier uncertainty propagation. This does not alter the fit.
        "influences": influence,
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



# Reconcile the profile fit with script 02


def load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


profile_checkpoint_path = (
    SOURCE_PROFILE_OUTPUT_DIR / "checkpoints" / "profile__amplitude_standardized.json"
)
profile_checkpoint = load_json(profile_checkpoint_path)
reference_result = profile_checkpoint["result"]
reference_payload = reference_result["payload"]
reference_beta = np.asarray(reference_payload["coef"], dtype=float)
reference_cov = np.asarray(reference_payload["cov"], dtype=float)
reference_basis = np.asarray(reference_payload["basis"], dtype=float)
reference_names = list(reference_payload["names"])

amp_frame = FRAME_REGISTRY["AMPLITUDE_STANDARDIZED"].copy()
if not FAST_MODE and int(amp_frame["id"].nunique()) != EXPECTED_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Expected {EXPECTED_REPEAT_PARTICIPANTS} participants; "
        f"found {amp_frame['id'].nunique()}"
    )

profile_columns = [f"profile__{network}" for network in ELIGIBLE_NETWORKS]
reference_design = DirectStateDesign(age_moderation=False).fit(amp_frame)
x_reference = reference_design.transform(amp_frame)
y_reference = amp_frame[profile_columns].to_numpy(dtype=float)
recovered_fit = fit_multivariate_profile(
    x_reference,
    y_reference,
    amp_frame["id"],
    amp_frame["state"],
    amp_frame["pedigree_group"],
)
if recovered_fit is None:
    raise RuntimeError("Could not recover the amplitude-standardized profile fit")
if reference_design.names != reference_names:
    raise AssertionError("Profile design columns do not match script 02")

reconciliation_rows = [
    {
        "quantity": "coefficient",
        "max_absolute_error": float(np.max(np.abs(recovered_fit["coef"] - reference_beta))),
        "tolerance": SOURCE_BETA_TOL,
    },
    {
        "quantity": "robust_covariance",
        "max_absolute_error": float(np.max(np.abs(recovered_fit["robust_cov"] - reference_cov))),
        "tolerance": SOURCE_COV_TOL,
    },
    {
        "quantity": "profile_basis",
        "max_absolute_error": float(np.max(np.abs(recovered_fit["basis"] - reference_basis))),
        "tolerance": SOURCE_BASIS_TOL,
    },
]
profile_fit_reconciliation = pd.DataFrame(reconciliation_rows)
profile_fit_reconciliation["pass"] = (
    profile_fit_reconciliation["max_absolute_error"]
    <= profile_fit_reconciliation["tolerance"]
)
if not profile_fit_reconciliation["pass"].all():
    print(profile_fit_reconciliation)
    raise AssertionError("The reconstructed profile fit does not match script 02")
display(profile_fit_reconciliation)



# Document task-state inputs and available QC resources


AABC_IMAGING_DOC_URL = (
    "https://www.humanconnectome.org/study/hcp-lifespan-aging/data-releases"
)
HCP_PIPELINES_URL = "https://github.com/Washington-University/HCPpipelines"

doc_candidates = sorted({
    path.resolve()
    for path in DATA_DIR.rglob("AABC_Release2_Imaging_Data_Packages.pdf")
    if path.is_file() and OUTPUT_DIR not in path.parents
})
if len(doc_candidates) > 1 and len({sha256_file(path) for path in doc_candidates}) > 1:
    raise RuntimeError("Multiple nonidentical AABC imaging-package appendices found")
imaging_doc_path = doc_candidates[0] if doc_candidates else None

name_inventory = []
for _, row in source_manifest.iterrows():
    path = Path(str(row["path"])).expanduser()
    name_inventory.append(str(path))
    if path.is_file() and path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                name_inventory.extend(archive.namelist())
        except Exception:
            pass

tclean_hits = [
    name for name in name_inventory
    if re.search(r"(^|[^a-z])tclean([^a-z]|$)", name.lower())
]
qc_columns = []
if qc_path is not None and Path(qc_path).is_file():
    try:
        qc_columns = list(read_table(qc_path).columns)
    except Exception:
        qc_columns = []
motion_tokens = re.compile(r"motion|mean[_ ]?fd|framewise[_ ]?displacement", flags=re.I)
retained_tokens = re.compile(
    r"retained.*frame|frame.*retained|usable.*frame|censor|scrub|"
    r"frames?[_ ]?(kept|used|remaining)",
    flags=re.I,
)
motion_columns = [column for column in qc_columns if motion_tokens.search(str(column))]
retained_columns = [column for column in qc_columns if retained_tokens.search(str(column))]
motion_available = bool(
    len(qc_long)
    and "motion" in qc_long
    and pd.to_numeric(qc_long["motion"], errors="coerce").notna().any()
)
retained_available = bool(retained_columns)
input_names = " ".join(name_inventory).lower()
parcel_timeseries_used = bool(re.search(r"ptseries|parcel.*time.*series", input_names))
documentation_source = str(imaging_doc_path) if imaging_doc_path else AABC_IMAGING_DOC_URL

task_and_resource_semantics = pd.DataFrame([
    {
        "resource": "recommended task-fMRI preprocessing",
        "available": True,
        "current_analysis_input": False,
        "interpretation": (
            "AABC Release 2 documents individually processed CARIT, FACENAME, and "
            "VISMOTOR data with spatial- and temporal-ICA cleaning and MSMAll alignment"
        ),
        "evidence": documentation_source,
    },
    {
        "resource": "tclean filename token",
        "available": bool(tclean_hits),
        "current_analysis_input": bool(tclean_hits),
        "interpretation": "A filename token alone does not establish task-GLM residualization",
        "evidence": "; ".join(tclean_hits[:8]) if tclean_hits else "no tclean token found",
    },
    {
        "resource": "task-evoked response removal before covariance estimation",
        "available": False,
        "current_analysis_input": False,
        "interpretation": "Task-GLM residualization is not documented for the released covariance IDPs",
        "evidence": documentation_source,
    },
    {
        "resource": "released within-network covariance IDPs",
        "available": True,
        "current_analysis_input": True,
        "interpretation": "The analysis uses released covariance rather than correlation or Fisher-z",
        "evidence": "analytic source inventory",
    },
    {
        "resource": "parcel-level task-fMRI time series",
        "available": bool(parcel_timeseries_used),
        "current_analysis_input": bool(parcel_timeseries_used),
        "interpretation": "Not treated as an input without an explicit ptseries or parcel-time-series product",
        "evidence": "analytic source inventory",
    },
    {
        "resource": "run-level motion",
        "available": motion_available,
        "current_analysis_input": motion_available,
        "interpretation": "Run/state-level motion was not available for adjustment" if not motion_available else str(motion_columns),
        "evidence": str(qc_path) if qc_path is not None else "no state-specific QC file",
    },
    {
        "resource": "retained-frame or censoring count",
        "available": retained_available,
        "current_analysis_input": retained_available,
        "interpretation": "Retained-frame information was not available" if not retained_available else str(retained_columns),
        "evidence": str(qc_path) if qc_path is not None else "no state-specific QC file",
    },
    {
        "resource": "generic MR QC issue codes",
        "available": bool(generic_qc_col is not None),
        "current_analysis_input": False,
        "interpretation": "Generic issue codes are not a motion or retained-frame measure",
        "evidence": "subject-event table",
    },
])
display(task_and_resource_semantics)



# Test linear baseline-age moderation

def state_interaction_rows_scalar(
    label,
    role,
    design,
    fit,
    interval_family="BONFERRONI_4_STATE",
):
    matrix = design.age_moderation_matrix()
    estimates = matrix @ fit["coef"]
    cov = matrix @ fit["robust_cov"] @ matrix.T
    se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    if interval_family == "BONFERRONI_4_STATE":
        critical = float(norm.ppf(1 - 0.05 / (2 * len(STATES))))
    else:
        critical = float(norm.ppf(.975))
    rows = []
    for j, state in enumerate(STATES):
        rows.append({
            "analysis": label,
            "role": role,
            "state": state,
            "network": "",
            "term": "baseline_age_x_time",
            "estimate_per_10y_age_change_in_5y_slope": float(estimates[j]),
            "se": float(se[j]),
            "ci95_low": float(estimates[j] - critical * se[j]),
            "ci95_high": float(estimates[j] + critical * se[j]),
            "interval_family": interval_family,
        })
    return rows


def fit_scalar_age_moderation(frame, outcome_col, label, role):
    frame = frame.copy()
    design = DirectStateDesign(age_moderation=True).fit(frame)
    x = design.transform(frame)
    y = pd.to_numeric(frame[outcome_col], errors="coerce").to_numpy(float)
    fit = fit_direct_profile_covariance(
        x,
        y,
        frame["id"],
        frame["state"],
        frame["pedigree_group"],
    )
    if fit is None:
        raise RuntimeError(f"{label}: age-moderation fit failed.")
    matrix = design.age_moderation_matrix()
    stat, df, p = wald_matrix(
        fit["coef"], fit["robust_cov"], matrix
    )
    if int(df) != 4:
        raise AssertionError(f"{label}: expected age-moderation df=4; got {df}")
    test = {
        "analysis": label,
        "role": role,
        "test": "baseline_age_x_time_x_state_joint",
        "chi2": float(stat),
        "df": int(df),
        "p": float(p),
        "moderation_detected_p_lt_0_05": bool(p < .05),
    }
    rows = state_interaction_rows_scalar(
        label, role, design, fit
    )
    return design, fit, test, rows


# ---- Primary: amplitude-standardized ten-network profile ----
profile_age_design = DirectStateDesign(age_moderation=True).fit(amp_frame)
x_profile_age = profile_age_design.transform(amp_frame)
y_profile_age = amp_frame[profile_columns].to_numpy(dtype=float)

profile_age_fit = fit_multivariate_profile(
    x_profile_age,
    y_profile_age,
    amp_frame["id"],
    amp_frame["state"],
    amp_frame["pedigree_group"],
)
if profile_age_fit is None:
    raise RuntimeError("Primary age-moderation profile fit failed.")

K = len(ELIGIBLE_NETWORKS)
Q = K - 1

# PRIMARY estimand:
# equal-state-average baseline-age × elapsed-time moderation of the
# amplitude-standardized ten-network relative profile.
#
# DirectStateDesign(age_moderation=True) contains one age×time slope
# interaction for each state. Averaging those four state-specific design
# vectors yields the state-general age×time effect. Expanding it through
# the K-1 Helmert profile coordinates gives the requested 9-df
# age × time × network omnibus, without adding a network×state moderation
# family to the primary question.
profile_age_state_matrix = profile_age_design.age_moderation_matrix()
if profile_age_state_matrix.shape != (len(STATES), len(profile_age_design.names)):
    raise AssertionError(
        "Unexpected state-specific age-moderation design dimensions."
    )

profile_age_equal_state_vector = profile_age_state_matrix.mean(axis=0, keepdims=True)
profile_age_matrix = block_expand(
    profile_age_equal_state_vector,
    Q,
)

profile_age_stat, profile_age_df, profile_age_p = wald_matrix(
    profile_age_fit["coef"],
    profile_age_fit["robust_cov"],
    profile_age_matrix,
)

expected_profile_df = K - 1
if int(profile_age_df) != expected_profile_df:
    raise AssertionError(
        f"Expected equal-state profile moderation df={expected_profile_df}; "
        f"got {profile_age_df}"
    )

age_joint_rows = [{
    "analysis": "AMPLITUDE_STANDARDIZED_TEN_NETWORK_PROFILE",
    "role": "PRIMARY",
    "test": "baseline_age_x_time_x_network_equal_state_average",
    "chi2": float(profile_age_stat),
    "df": int(profile_age_df),
    "p": float(profile_age_p),
    "moderation_detected_p_lt_0_05": bool(profile_age_p < .05),
}]

age_coef_rows = []

# State-specific network interaction estimates below are descriptive
# components of the fitted moderation model. They are not an additional
# state-comparison test family and do not replace the 9-df primary omnibus.
basis_age = np.asarray(profile_age_fit["basis"], float)
for state in STATES:
    age_design_vector = (
        profile_age_design.age_moderation_matrix()[
            STATES.index(state)
        ]
    )
    for k, network in enumerate(ELIGIBLE_NETWORKS):
        unit = np.zeros(K)
        unit[k] = 1.0
        coordinate_weight = unit @ basis_age
        contrast = np.kron(
            coordinate_weight,
            age_design_vector,
        )
        estimate, se, z, p = scalar_contrast(
            profile_age_fit["coef"],
            profile_age_fit["robust_cov"],
            contrast,
        )
        critical = float(norm.ppf(.975))
        age_coef_rows.append({
            "analysis": "AMPLITUDE_STANDARDIZED_TEN_NETWORK_PROFILE",
            "role": "PRIMARY_COMPONENT_DESCRIPTIVE_AFTER_JOINT",
            "state": state,
            "network": network,
            "term": "baseline_age_x_time_x_network",
            "estimate_per_10y_age_change_in_5y_slope": estimate,
            "se": se,
            "ci95_low": estimate - critical * se,
            "ci95_high": estimate + critical * se,
            "interval_family": "POINTWISE_COMPONENT_INTERVAL__OMNIBUS_IS_PRIMARY",
        })

# ---- Sensitive secondary: FP-relative scalar ----
fp_amp = amp_frame.copy()
fp_other = [
    network for network in ELIGIBLE_NETWORKS
    if network != FP_NETWORK
]
fp_amp["Y_FP_REL"] = (
    pd.to_numeric(
        fp_amp[f"L_STD__{FP_NETWORK}"],
        errors="coerce",
    ).astype("float64")
    - fp_amp[
        [f"L_STD__{n}" for n in fp_other]
    ].to_numpy(dtype="float64").mean(axis=1)
)
fp_amp["Y_FP_REL_FROM_PROFILE"] = (
    (K / (K - 1.0))
    * pd.to_numeric(
        fp_amp[f"R_STD__{FP_NETWORK}"],
        errors="coerce",
    ).astype("float64")
)
fp_identity_error = float(np.nanmax(np.abs(
    fp_amp["Y_FP_REL"] - fp_amp["Y_FP_REL_FROM_PROFILE"]
)))
if fp_identity_error > 1e-10:
    raise AssertionError(
        f"FP-relative identity failed: {fp_identity_error:.3e}"
    )

(
    fp_age_design,
    fp_age_fit,
    fp_age_test,
    fp_age_rows,
) = fit_scalar_age_moderation(
    fp_amp,
    "Y_FP_REL",
    "FP_RELATIVE_AMPLITUDE_STANDARDIZED",
    "SENSITIVE_SECONDARY",
)
age_joint_rows.append(fp_age_test)
age_coef_rows.extend(fp_age_rows)

# ---- Context: native and amplitude-standardized common levels ----
native_frame = FRAME_REGISTRY["FULL_PRIMARY"].copy()

(
    common_native_age_design,
    common_native_age_fit,
    common_native_age_test,
    common_native_age_rows,
) = fit_scalar_age_moderation(
    native_frame,
    "outcome_G",
    "NATIVE_COMMON_COVARIANCE",
    "CONTEXT_ONLY",
)
age_joint_rows.append(common_native_age_test)
age_coef_rows.extend(common_native_age_rows)

(
    common_std_age_design,
    common_std_age_fit,
    common_std_age_test,
    common_std_age_rows,
) = fit_scalar_age_moderation(
    amp_frame,
    "outcome_G",
    "AMPLITUDE_STANDARDIZED_COMMON_COVARIANCE",
    "CONTEXT_ONLY",
)
age_joint_rows.append(common_std_age_test)
age_coef_rows.extend(common_std_age_rows)

age_moderation_joint_tests = pd.DataFrame(age_joint_rows)
age_moderation_coefficients = pd.DataFrame(age_coef_rows)

age_moderation_joint_tests.to_csv(
    TABLE_DIR / "age_moderation_joint_tests.csv",
    index=False,
)
age_moderation_coefficients.to_csv(
    TABLE_DIR / "age_moderation_coefficients.csv",
    index=False,
)

display(age_moderation_joint_tests)
display(age_moderation_coefficients.head(12))



# Propagate uncertainty for profile convergence

def baseline_design_vector(design, state):
    c = np.zeros(len(design.names), dtype=float)
    if "Intercept" not in design.names:
        raise AssertionError("Intercept missing from reference design.")
    c[design.names.index("Intercept")] = 1.0
    for coded in STATES[:-1]:
        name = f"state_sum[{coded}]"
        if name in design.names:
            c[design.names.index(name)] = (
                1.0 if state == coded
                else -1.0 if state == STATES[-1]
                else 0.0
            )
    return c


def network_coef_contrast(network, design_vector, basis):
    k = ELIGIBLE_NETWORKS.index(network)
    unit = np.zeros(len(ELIGIBLE_NETWORKS))
    unit[k] = 1.0
    coordinate_weight = unit @ basis
    return np.kron(coordinate_weight, design_vector)


basis = reference_basis
contrast_registry = {}
for state in STATES:
    level_d = baseline_design_vector(reference_design, state)
    slope_d = reference_design.state_slope_vector(state)
    level_c = np.vstack([
        network_coef_contrast(network, level_d, basis)
        for network in ELIGIBLE_NETWORKS
    ])
    slope_c = np.vstack([
        network_coef_contrast(network, slope_d, basis)
        for network in ELIGIBLE_NETWORKS
    ])
    contrast_registry[state] = {
        "level": level_c,
        "slope": slope_c,
    }


def rowwise_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(
        np.sum(xc**2, axis=1)
        * np.sum(yc**2, axis=1)
    )
    out = np.full(len(x), np.nan)
    good = denom > 1e-14
    out[good] = (
        np.sum(xc[good] * yc[good], axis=1)
        / denom[good]
    )
    return out


def geometry_metrics(level0, slope):
    level0 = np.asarray(level0, float)
    slope = np.asarray(slope, float)
    level5 = level0 + slope

    r0 = rowwise_corr(level0, slope)
    r5 = rowwise_corr(level5, slope)

    sd0 = np.std(level0, axis=1, ddof=0)
    sd5 = np.std(level5, axis=1, ddof=0)
    ratio = np.divide(
        sd5,
        sd0,
        out=np.full_like(sd5, np.nan),
        where=sd0 > 1e-14,
    )
    diff = sd5 - sd0

    x_mean = level0.mean(axis=1, keepdims=True)
    y_mean = slope.mean(axis=1, keepdims=True)
    xc = level0 - x_mean
    yc = slope - y_mean
    xvar = np.sum(xc**2, axis=1)
    trend = np.divide(
        np.sum(xc * yc, axis=1),
        xvar,
        out=np.full(len(level0), np.nan),
        where=xvar > 1e-14,
    )
    fitted = y_mean + trend[:, None] * xc
    residual = slope - fitted

    fp_idx = ELIGIBLE_NETWORKS.index(FP_NETWORK)
    fp_resid = residual[:, fp_idx]
    fp_rank = (
        1
        + np.sum(
            residual > fp_resid[:, None] + 1e-12,
            axis=1,
        )
    ).astype(float)

    return {
        "baseline_level_slope_r": r0,
        "five_year_level_slope_r": r5,
        "baseline_across_network_sd": sd0,
        "five_year_across_network_sd": sd5,
        "sd_ratio_5y_over_baseline": ratio,
        "sd_difference_5y_minus_baseline": diff,
        "fp_residual_from_baseline_trend": fp_resid,
        "fp_residual_rank_largest_positive_1": fp_rank,
    }


def coefficient_geometry_rows(
    beta_draws,
    draw_ids,
):
    rows = []

    state_level = {}
    state_slope = {}
    for state in STATES:
        state_level[state] = (
            beta_draws @ contrast_registry[state]["level"].T
        )
        state_slope[state] = (
            beta_draws @ contrast_registry[state]["slope"].T
        )
        metrics = geometry_metrics(
            state_level[state],
            state_slope[state],
        )
        for i, draw_id in enumerate(draw_ids):
            rows.append({
                "draw": int(draw_id),
                "state": state,
                **{
                    key: float(value[i])
                    for key, value in metrics.items()
                },
            })

    avg_level = np.mean(
        np.stack(
            [state_level[s] for s in STATES],
            axis=0,
        ),
        axis=0,
    )
    avg_slope = np.mean(
        np.stack(
            [state_slope[s] for s in STATES],
            axis=0,
        ),
        axis=0,
    )
    metrics = geometry_metrics(avg_level, avg_slope)
    for i, draw_id in enumerate(draw_ids):
        rows.append({
            "draw": int(draw_id),
            "state": "EQUAL_STATE_AVERAGE",
            **{
                key: float(value[i])
                for key, value in metrics.items()
            },
        })
    return rows


# Observed geometry at the script-02 coefficient center.
observed_geometry_rows = coefficient_geometry_rows(
    reference_beta[None, :],
    [-1],
)
observed_geometry = pd.DataFrame(observed_geometry_rows)

# Shared Gaussian pedigree multipliers, chunked.
influences = recovered_fit["influences"]
pedigrees = sorted(influences)
influence_matrix = np.vstack([
    np.asarray(influences[pg], float).ravel(order="F")
    for pg in pedigrees
])
if influence_matrix.shape[1] != len(reference_beta):
    raise AssertionError(
        "Recovered influence dimension does not match the reference coefficient vector."
    )

rng = np.random.default_rng(BOOTSTRAP_SEED)
bootstrap_rows = []

for start in tqdm(
    range(0, BOOTSTRAP_B, BOOTSTRAP_CHUNK),
    desc="Pedigree multiplier convergence bootstrap",
):
    stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_B)
    z = rng.standard_normal(
        size=(stop - start, len(pedigrees))
    )
    beta_draws = reference_beta[None, :] + z @ influence_matrix
    bootstrap_rows.extend(
        coefficient_geometry_rows(
            beta_draws,
            np.arange(start, stop),
        )
    )

profile_convergence_bootstrap = pd.DataFrame(
    bootstrap_rows
)

metric_columns = [
    "baseline_level_slope_r",
    "five_year_level_slope_r",
    "baseline_across_network_sd",
    "five_year_across_network_sd",
    "sd_ratio_5y_over_baseline",
    "sd_difference_5y_minus_baseline",
    "fp_residual_from_baseline_trend",
    "fp_residual_rank_largest_positive_1",
]

finite_mask = np.isfinite(
    profile_convergence_bootstrap[metric_columns]
).all(axis=1)
profile_convergence_bootstrap = (
    profile_convergence_bootstrap.loc[finite_mask]
    .reset_index(drop=True)
)

profile_convergence_bootstrap.to_csv(
    TABLE_DIR / "profile_convergence_bootstrap.csv",
    index=False,
)

print({
    "requested_draws": BOOTSTRAP_B,
    "rows_written": len(profile_convergence_bootstrap),
    "state_labels": profile_convergence_bootstrap["state"].nunique(),
})


summary_rows = []
for state in [*STATES, "EQUAL_STATE_AVERAGE"]:
    observed = observed_geometry.loc[observed_geometry["state"].eq(state)].iloc[0]
    draws = profile_convergence_bootstrap.loc[
        profile_convergence_bootstrap["state"].eq(state)
    ]
    for metric in metric_columns:
        values = draws[metric].to_numpy(float)
        summary_rows.append({
            "state": state,
            "metric": metric,
            "observed": float(observed[metric]),
            "bootstrap_q025": float(np.quantile(values, .025)),
            "bootstrap_median": float(np.quantile(values, .5)),
            "bootstrap_q975": float(np.quantile(values, .975)),
            "bootstrap_draws": int(len(values)),
            "inference": "PEDIGREE_CLUSTER_GAUSSIAN_MULTIPLIER",
        })

profile_convergence_summary = pd.DataFrame(summary_rows)
equal_state_summary = profile_convergence_summary.loc[
    profile_convergence_summary["state"].eq("EQUAL_STATE_AVERAGE")
]
equal_state_correlation = equal_state_summary.loc[
    equal_state_summary["metric"].eq("baseline_level_slope_r")
].iloc[0]
equal_state_sd_ratio = equal_state_summary.loc[
    equal_state_summary["metric"].eq("sd_ratio_5y_over_baseline")
].iloc[0]
partial_convergence_supported = bool(
    float(equal_state_correlation["bootstrap_q975"]) < 0
    and float(equal_state_sd_ratio["bootstrap_q975"]) < 1
)
profile_convergence_summary["partial_convergence_supported"] = partial_convergence_supported
profile_convergence_summary.to_csv(
    TABLE_DIR / "profile_convergence_summary.csv", index=False
)
display(profile_convergence_summary)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
display_states = [*STATES, "EQUAL_STATE_AVERAGE"]
display_labels = [*STATES, "Equal-state avg."]
for ax, metric, null, xlabel, panel in [
    (axes[0], "baseline_level_slope_r", 0.0, "Baseline level-slope correlation", "A"),
    (axes[1], "sd_ratio_5y_over_baseline", 1.0, "Across-network SD ratio (5 y / baseline)", "B"),
]:
    frame = (
        profile_convergence_summary.loc[profile_convergence_summary["metric"].eq(metric)]
        .set_index("state").reindex(display_states)
    )
    y = np.arange(len(frame))[::-1]
    estimate = frame["observed"].to_numpy(float)
    low = frame["bootstrap_q025"].to_numpy(float)
    high = frame["bootstrap_q975"].to_numpy(float)
    ax.errorbar(
        estimate, y, xerr=np.vstack([estimate - low, high - estimate]),
        fmt="o", capsize=2.5, elinewidth=1.0, markersize=4.5,
    )
    ax.axvline(null, lw=.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel(xlabel)
    ax.set_title(panel, loc="left", weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "profile_convergence_uncertainty.png", dpi=600, bbox_inches="tight")
fig.savefig(FIG_DIR / "profile_convergence_uncertainty.pdf", bbox_inches="tight")
plt.close(fig)



# Summarize matched partial covariance

fp_state_path = (
    SOURCE_FP_OUTPUT_DIR
    / "tables"
    / "fp_relative_state_slopes.csv"
)
fp_test_path = (
    SOURCE_FP_OUTPUT_DIR
    / "tables"
    / "fp_relative_tests.csv"
)
if not fp_state_path.is_file() or not fp_test_path.is_file():
    raise FileNotFoundError(
        "Script 03 FP tables are required."
    )

fp_slopes = pd.read_csv(fp_state_path, low_memory=False)
fp_tests = pd.read_csv(fp_test_path, low_memory=False)

full_fp = fp_slopes.loc[
    fp_slopes["scenario"].astype(str).eq(
        "AMPLITUDE_STANDARDIZED"
    )
].copy()
partial_fp = fp_slopes.loc[
    fp_slopes["scenario"].astype(str).eq(
        "PARTIAL_MATCHED"
    )
].copy()

if len(full_fp) != 4 or len(partial_fp) != 4:
    raise AssertionError(
        f"Expected four full and four partial state slopes; "
        f"found {len(full_fp)} and {len(partial_fp)}."
    )

full_test = fp_tests.loc[
    fp_tests["scenario"].astype(str).eq(
        "AMPLITUDE_STANDARDIZED"
    )
    & fp_tests["test"].astype(str).eq(
        "all_four_state_slopes_zero"
    )
]
partial_test = fp_tests.loc[
    fp_tests["scenario"].astype(str).eq(
        "PARTIAL_MATCHED"
    )
    & fp_tests["test"].astype(str).eq(
        "all_four_state_slopes_zero"
    )
]
if len(full_test) != 1 or len(partial_test) != 1:
    raise AssertionError("FP joint tests are not unique.")

full_map = full_fp.set_index("state")
partial_map = partial_fp.set_index("state")

partial_rows = []
for state in STATES:
    f = full_map.loc[state]
    p = partial_map.loc[state]
    same_direction = bool(
        np.sign(float(f["slope_per_5y"]))
        == np.sign(float(p["slope_per_5y"]))
    )
    partial_rows.append({
        "state": state,
        "full_covariance_fp_relative_slope_per_5y": float(
            f["slope_per_5y"]
        ),
        "full_simultaneous_ci_low": float(
            f["simultaneous_ci_low"]
        ),
        "full_simultaneous_ci_high": float(
            f["simultaneous_ci_high"]
        ),
        "partial_covariance_fp_relative_slope_per_5y": float(
            p["slope_per_5y"]
        ),
        "partial_simultaneous_ci_low": float(
            p["simultaneous_ci_low"]
        ),
        "partial_simultaneous_ci_high": float(
            p["simultaneous_ci_high"]
        ),
        "direction_preserved": same_direction,
        "partial_absolute_point_estimate_smaller": bool(
            abs(float(p["slope_per_5y"]))
            < abs(float(f["slope_per_5y"]))
        ),
        "full_joint_p": float(full_test.iloc[0]["p"]),
        "partial_joint_p": float(partial_test.iloc[0]["p"]),
    })

partial_covariance_comparison = pd.DataFrame(
    partial_rows
)

all_direction = bool(
    partial_covariance_comparison[
        "direction_preserved"
    ].all()
)
smaller_n = int(
    partial_covariance_comparison[
        "partial_absolute_point_estimate_smaller"
    ].sum()
)

if all_direction and smaller_n == 4:
    partial_sentence = (
        "The FP-relative trajectory was directionally preserved under partial "
        "covariance, with smaller point estimates in all four states than under "
        "full covariance."
    )
elif all_direction:
    partial_sentence = (
        "The FP-relative trajectory was directionally preserved under partial "
        f"covariance; partial-covariance point estimates were smaller in "
        f"{smaller_n} of 4 states."
    )
else:
    partial_sentence = (
        "The partial-covariance sensitivity did not preserve the FP-relative "
        "direction in all four states."
    )

partial_covariance_comparison["summary_sentence"] = partial_sentence
partial_covariance_comparison.to_csv(
    TABLE_DIR / "partial_covariance_comparison.csv",
    index=False,
)

display(partial_covariance_comparison)
print(partial_sentence)


profile_age_test = age_moderation_joint_tests.loc[
    age_moderation_joint_tests["analysis"].eq("AMPLITUDE_STANDARDIZED_TEN_NETWORK_PROFILE")
].iloc[0]
fp_age_test_row = age_moderation_joint_tests.loc[
    age_moderation_joint_tests["analysis"].eq("FP_RELATIVE_AMPLITUDE_STANDARDIZED")
].iloc[0]
partial_joint_p = float(partial_covariance_comparison["partial_joint_p"].iloc[0])

final_checks_summary = pd.DataFrame([
    {
        "result": "profile_age_moderation",
        "estimate": float(profile_age_test["chi2"]),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "df": int(profile_age_test["df"]),
        "p": float(profile_age_test["p"]),
    },
    {
        "result": "fp_relative_age_moderation",
        "estimate": float(fp_age_test_row["chi2"]),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "df": int(fp_age_test_row["df"]),
        "p": float(fp_age_test_row["p"]),
    },
    {
        "result": "equal_state_baseline_level_slope_correlation",
        "estimate": float(equal_state_correlation["observed"]),
        "ci_low": float(equal_state_correlation["bootstrap_q025"]),
        "ci_high": float(equal_state_correlation["bootstrap_q975"]),
        "df": np.nan,
        "p": np.nan,
    },
    {
        "result": "equal_state_sd_ratio_5y_over_baseline",
        "estimate": float(equal_state_sd_ratio["observed"]),
        "ci_low": float(equal_state_sd_ratio["bootstrap_q025"]),
        "ci_high": float(equal_state_sd_ratio["bootstrap_q975"]),
        "df": np.nan,
        "p": np.nan,
    },
])
final_checks_summary.to_csv(TABLE_DIR / "final_checks_summary.csv", index=False)
profile_fit_reconciliation.to_csv(TABLE_DIR / "profile_fit_reconciliation.csv", index=False)
task_and_resource_semantics.to_csv(TABLE_DIR / "task_and_resource_semantics.csv", index=False)

run_manifest = {
    "script": "06_final_checks.py",
    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "participants": int(amp_frame["id"].nunique()),
    "eligible_networks": ELIGIBLE_NETWORKS,
    "profile_age_moderation_p": float(profile_age_test["p"]),
    "fp_relative_age_moderation_p": float(fp_age_test_row["p"]),
    "partial_convergence_supported": partial_convergence_supported,
    "convergence_bootstrap_draws": BOOTSTRAP_B,
    "partial_covariance_direction_preserved_all_states": bool(all_direction),
    "partial_covariance_joint_p": partial_joint_p,
    "task_state_description": "cleaned task-state covariance",
    "run_level_motion_available": motion_available,
    "retained_frame_information_available": retained_available,
    "tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
    "figures": sorted(path.name for path in FIG_DIR.iterdir() if path.is_file()),
}
(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2) + "\n"
)

print("\nFINAL CHECKS COMPLETE")
print(f"Participants: {amp_frame['id'].nunique():,}")
print(
    "Profile age moderation: "
    f"chi2({int(profile_age_test['df'])})={float(profile_age_test['chi2']):.3f}, "
    f"p={float(profile_age_test['p']):.6g}"
)
print(
    "FP-relative age moderation: "
    f"chi2({int(fp_age_test_row['df'])})={float(fp_age_test_row['chi2']):.3f}, "
    f"p={float(fp_age_test_row['p']):.6g}"
)
print(
    "Equal-state baseline level-slope correlation: "
    f"{float(equal_state_correlation['observed']):.3f} "
    f"[{float(equal_state_correlation['bootstrap_q025']):.3f}, "
    f"{float(equal_state_correlation['bootstrap_q975']):.3f}]"
)
print(
    "Equal-state SD ratio (5 y / baseline): "
    f"{float(equal_state_sd_ratio['observed']):.3f} "
    f"[{float(equal_state_sd_ratio['bootstrap_q025']):.3f}, "
    f"{float(equal_state_sd_ratio['bootstrap_q975']):.3f}]"
)
print(f"Partial convergence supported: {partial_convergence_supported}")
print(
    "Partial covariance direction preserved / joint p: "
    f"{bool(all_direction)} / {partial_joint_p:.6g}"
)
print(f"Task-state description: cleaned task-state covariance")
print(f"Outputs: {OUTPUT_DIR}")
