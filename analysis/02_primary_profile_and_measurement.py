#!/usr/bin/env python3
"""Fit the common-level and network-profile models and measurement analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
from scipy.stats import chi2, norm

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Local multimodal project root containing the fMRI caches/archives.",
    )
    parser.add_argument(
        "--subject-table", type=Path, required=True,
        help="Frozen longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Local results root; outputs go into 02_primary_profile_and_measurement.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development run: lower state-coverage threshold and use 999 Gate-1 draws.",
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Ignore compatible model checkpoints and refit every scenario.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=None,
        help="Override Gate-1 multiplier draws (paper run default: 4999).",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
SUBJECT_TABLE = ARGS.subject_table.expanduser().resolve()
RESULTS_ROOT = ARGS.results_dir.expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "02_primary_profile_and_measurement"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.is_dir():
    raise NotADirectoryError(DATA_DIR)
if not SUBJECT_TABLE.is_file():
    raise FileNotFoundError(SUBJECT_TABLE)

FAST_MODE = bool(ARGS.fast)
FORCE_REFIT = bool(ARGS.force_refit)
MIN_STATE_PARTICIPANTS = 12 if FAST_MODE else 40
EXPECTED_FROZEN_REPEAT_PARTICIPANTS = 852
MIN_PARCELS = 2
MIN_STATE_COVERAGE = .90
MIN_STATE_POSITIVE = .99
SEED = 20260828

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
sns.set_theme(style="whitegrid", context="notebook")


# ===== PRIMARY COMMON-LEVEL AND TEN-NETWORK PROFILE =====

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
            raise KeyError(f"{path} lacks frozen metrics: {missing}")
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

def resolve_longitudinal_subject_table():
    # First reuse the exact table recorded by the successfully executed ASL
    # longitudinal analysis. This avoids selecting a non-imaging
    # table whose identifiers cannot form id_event.
    manifest_candidates = sorted(DATA_DIR.rglob(
        "AABC2_longitudinal_regional_ASL_topography_final_repair_outputs/source_manifest.csv"
    ))
    search_roots = [DATA_DIR]
    sibling_gate = DATA_DIR.parent / "gate data"
    if sibling_gate.exists():
        search_roots.append(sibling_gate)
    for manifest_path in manifest_candidates:
        manifest = read_table(manifest_path)
        role_col = pick_column(manifest, ["role"])
        path_col = pick_column(manifest, ["path"])
        if role_col is None or path_col is None:
            continue
        preferred = manifest.loc[
            manifest[role_col].astype(str).str.contains("subject|event", case=False, regex=True), path_col
        ]
        for raw_path in preferred.astype(str):
            direct = Path(raw_path).expanduser()
            if direct.is_file():
                return direct.resolve(), "ASL_FINAL_REPAIR_SOURCE_MANIFEST"
            for root in search_roots:
                matches = list(root.rglob(direct.name))
                if len(matches) == 1:
                    return matches[0].resolve(), "ASL_FINAL_REPAIR_SOURCE_MANIFEST_BASENAME"
    candidates = []
    for root in search_roots:
        candidates.extend(root.rglob("*AABC2_subjects*.csv"))
        candidates.extend(root.rglob("AABC2_longitudinal_event_manifest.csv"))
        candidates.extend(root.rglob("AABC2_event_manifest.csv"))
    candidates = sorted({p.resolve() for p in candidates if p.is_file() and outside_output(p)})
    valid = []
    for path in candidates:
        probe = pd.read_csv(path, nrows=10, low_memory=False)
        has_id_event = pick_column(probe, ["id_event", "subject_event", "participant_event"]) is not None
        has_pair = (
            pick_column(probe, ["id", "subject_id", "participant_id", "src_subject_id"]) is not None
            and pick_column(probe, ["eventname", "event_name", "event", "visit", "session"]) is not None
        )
        if has_id_event or has_pair:
            valid.append(path)
    if len(valid) == 1:
        return valid[0], "SCHEMA_VALIDATED_SEARCH"
    if len(valid) > 1:
        hashes = {sha256_file(p) for p in valid}
        if len(hashes) == 1:
            return valid[0], "SCHEMA_VALIDATED_IDENTICAL_COPIES"
        detail = "\n".join(f"  {p}" for p in valid)
        raise RuntimeError(f"Multiple nonidentical schema-valid longitudinal subject tables:\n{detail}")
    raise FileNotFoundError(
        "Could not resolve the longitudinal subject-event table. Set AABC2_DATA_DIR to the multimodal "
        "project root or copy the ASL final-repair source_manifest.csv with its subject table."
    )


subject_path, subject_resolution = SUBJECT_TABLE, "COMMAND_LINE"
subjects_raw = read_table(subject_path)
source_rows.append({
    "role": "longitudinal_subject_event_table", "path": str(subject_path),
    "sha256": sha256_file(subject_path), "resolution": subject_resolution,
})
source_manifest = pd.DataFrame(source_rows)
print(source_manifest)
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
print(qc_resource_audit)

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
        raise FileNotFoundError("The frozen fmri_node_parcel_manifest.csv is required")
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

# Reproduce the frozen DMN/FP repeat parent before testing other networks.
parent_ok = (
    np.isfinite(full_rows[["age0_c5", "t5"]].to_numpy(float)).all(axis=1)
    & (full_rows["cov__Default"] > 0)
    & (full_rows["cov__Frontoparietal"] > 0)
)
parent_candidate = full_rows.loc[parent_ok].copy()
parent_session_n = parent_candidate[["id", "id_event"]].drop_duplicates().groupby("id")["id_event"].nunique()
frozen_repeat_ids = set(parent_session_n[parent_session_n.ge(2)].index)
parent_rows = parent_candidate.loc[parent_candidate["id"].isin(frozen_repeat_ids)].copy()
if not FAST_MODE and len(frozen_repeat_ids) != EXPECTED_FROZEN_REPEAT_PARTICIPANTS:
    raise AssertionError(
        f"Frozen repeat cohort mismatch: expected {EXPECTED_FROZEN_REPEAT_PARTICIPANTS}, found {len(frozen_repeat_ids)}"
    )

# Eligibility uses only the frozen parent opportunities and is completed before slopes.
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
    raise AssertionError(f"Only {len(ELIGIBLE_NETWORKS)} networks passed the frozen eligibility gate")
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


def derive_profile(frame, covariance_prefix="cov__", output_suffix=""):
    out = frame.copy()
    l_columns = []

    for network in ELIGIBLE_NETWORKS:
        target = f"L{output_suffix}__{network}"

        # Force double precision before log/profile calculations.
        values = pd.to_numeric(
            out[f"{covariance_prefix}{network}"],
            errors="coerce",
        ).astype(np.float64)

        out[target] = np.log(values.where(values > 0))
        l_columns.append(target)

    out[f"G{output_suffix}"] = (
        out[l_columns]
        .to_numpy(dtype=np.float64)
        .mean(axis=1)
    )

    for network, column in zip(ELIGIBLE_NETWORKS, l_columns):
        out[f"R{output_suffix}__{network}"] = (
            out[column].to_numpy(dtype=np.float64)
            - out[f"G{output_suffix}"].to_numpy(dtype=np.float64)
        )

    return out


profile_data = derive_profile(profile_data)
profile_identity_audit = pd.DataFrame([{
    "identity": "R_Default - R_Frontoparietal = L_Default - L_Frontoparietal",
    "max_absolute_error": float(np.nanmax(np.abs(
        (profile_data["R__Default"] - profile_data["R__Frontoparietal"])
        - (profile_data["L__Default"] - profile_data["L__Frontoparietal"])
    ))),
    "max_profile_sum_error": float(np.nanmax(np.abs(
        profile_data[[f"R__{network}" for network in ELIGIBLE_NETWORKS]].sum(axis=1)
    ))),
}])
if profile_identity_audit[["max_absolute_error", "max_profile_sum_error"]].to_numpy().max() > 1e-10:
    raise AssertionError("Common-level/profile algebra failed")

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

print(parcel_counts)
print(network_state_eligibility)
print(network_eligibility)
print(profile_identity_audit)
print("Eligible networks frozen before slopes:", ELIGIBLE_NETWORKS)
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
        raise KeyError(f"{member} lacks {len(missing)} frozen parcels; first {missing[:5]}")
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

print(cohort_summary)
print(amplitude_coverage)
print(partial_positivity_coverage)

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

def fingerprint(frame, spec, model):
    columns = ["id", "id_event", "state", "pedigree_group", "age0_c5", "t5", "site", "site_scanner", "sex", "study"]
    columns += ["outcome_G"] if model == "common" else [f"profile__{network}" for network in ELIGIBLE_NETWORKS]
    payload = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=True).to_numpy().tobytes()
    return hashlib.sha256(payload + json.dumps({**spec, "model": model, "networks": ELIGIBLE_NETWORKS}, sort_keys=True).encode()).hexdigest()


def common_fit(spec):
    frame = FRAME_REGISTRY[spec["frame"]].copy()
    required = ["outcome_G", "age0_c5", "t5"]
    frame = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(axis=1)].copy()
    counts = frame.groupby("state")["id"].nunique().reindex(STATES, fill_value=0)
    if counts.min() < MIN_STATE_PARTICIPANTS:
        return {"summary": {**spec, "model": "COMMON_LEVEL", "fit_status": "INSUFFICIENT_STATE_COVERAGE",
                            "status_detail": json.dumps(counts.to_dict())}, "tests": [], "state": [], "payload": None}
    fp = fingerprint(frame, spec, "common")
    checkpoint = CHECKPOINT_DIR / f"common__{spec['scenario'].lower()}.json"
    if checkpoint.exists() and not FORCE_REFIT:
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") == fp:
            return saved["result"]
    try:
        design = DirectStateDesign().fit(frame)
        x = design.transform(frame)
        fit = fit_direct_profile_covariance(x, frame["outcome_G"], frame["id"], frame["state"], frame["pedigree_group"])
        if fit is None:
            raise RuntimeError("Common-level covariance fit returned None")
        beta, cov = fit["coef"], fit["robust_cov"]
        c_state = design.state_slope_matrix()
        tests = []
        for test, matrix in [
            ("all_four_common_level_slopes_zero", c_state),
            ("common_level_state_heterogeneity", c_state[:3] - c_state[3]),
            ("common_level_site_x_time", design.site_time_matrix()),
            ("common_level_site_x_time_x_state", design.site_time_state_matrix()),
        ]:
            stat, df, p = wald_matrix(beta, cov, matrix)
            tests.append({"scenario": spec["scenario"], "model": "COMMON_LEVEL", "test": test, "chi2": stat, "df": df, "p": p})
        estimates = c_state @ beta
        state_cov = c_state @ cov @ c_state.T
        se = np.sqrt(np.maximum(0, np.diag(state_cov)))
        pvals = 2 * norm.sf(np.abs(estimates / se))
        pholm = holm_adjust(pvals)
        q4 = norm.ppf(1 - .05 / 8)
        state_rows = [{
            "scenario": spec["scenario"], "model": "COMMON_LEVEL", "state": state,
            "slope_per_5y": float(estimates[j]), "se": float(se[j]), "p_unadjusted": float(pvals[j]),
            "p_holm": float(pholm[j]), "simultaneous_ci_low": float(estimates[j] - q4 * se[j]),
            "simultaneous_ci_high": float(estimates[j] + q4 * se[j]),
        } for j, state in enumerate(STATES)]
        summary = {**spec, "model": "COMMON_LEVEL", "fit_status": "PASS", "status_detail": "",
                   "rows": fit["row_n"], "participants": fit["participant_n"], "pedigrees": fit["pedigree_n"],
                   "design_columns": len(design.names), "design_rank": fit["rank"],
                   "optimizer_success": fit["optimizer_success"], "optimizer_nfev": fit["optimizer_nfev"],
                   "tau_participant": fit["tau_participant"], "tau_participant_state": fit["tau_participant_state"]}
        result = {"summary": summary, "tests": tests, "state": state_rows,
                  "payload": {"coef": beta, "cov": cov, "names": design.names}}
        checkpoint.write_text(json.dumps({"fingerprint": fp, "result": result}, default=jsonable, indent=2) + "\n")
        return result
    except Exception as exc:
        return {"summary": {**spec, "model": "COMMON_LEVEL", "fit_status": "FIT_ERROR",
                            "status_detail": f"{type(exc).__name__}: {exc}"}, "tests": [], "state": [], "payload": None}


def contrast_weight(definition):
    weight = np.zeros(K)
    index = {network: j for j, network in enumerate(ELIGIBLE_NETWORKS)}
    for network, value in definition["positive"].items():
        if network not in index: return None
        weight[index[network]] += value
    for network, value in definition["negative"].items():
        if network not in index: return None
        weight[index[network]] -= value
    if abs(weight.sum()) > 1e-12:
        raise AssertionError(f"Planned contrast is not profile-invariant: {definition['contrast']}")
    return weight


def profile_fit(spec):
    frame = FRAME_REGISTRY[spec["frame"]].copy()
    profile_columns = [f"profile__{network}" for network in ELIGIBLE_NETWORKS]
    required = [*profile_columns, "age0_c5", "t5"]
    frame = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(axis=1)].copy()
    counts = frame.groupby("state")["id"].nunique().reindex(STATES, fill_value=0)
    if counts.min() < MIN_STATE_PARTICIPANTS:
        return {"summary": {**spec, "model": "NETWORK_PROFILE", "fit_status": "INSUFFICIENT_STATE_COVERAGE",
                            "status_detail": json.dumps(counts.to_dict())}, "tests": [], "network": [],
                "contrast_tests": [], "contrast_state": [], "payload": None}
    fp = fingerprint(frame, spec, "profile")
    checkpoint = CHECKPOINT_DIR / f"profile__{spec['scenario'].lower()}.json"
    if checkpoint.exists() and not FORCE_REFIT:
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") == fp:
            return saved["result"]
    try:
        design = DirectStateDesign().fit(frame)
        x = design.transform(frame)
        fit = fit_multivariate_profile(x, frame[profile_columns], frame["id"], frame["state"], frame["pedigree_group"])
        if fit is None:
            raise RuntimeError("Multivariate network-profile fit returned None")
        beta, cov = fit["coef"], fit["robust_cov"]
        basis = np.asarray(fit["basis"], float)
        q = basis.shape[1]
        c_state = design.state_slope_matrix()
        c_pooled = c_state.mean(axis=0)

        pooled_network = block_expand(c_pooled[None, :], q)
        state_diff = c_state[:3] - c_state[3]
        profile_state = block_expand(state_diff, q)
        site_network = block_expand(design.site_time_matrix(), q)
        site_state_network = block_expand(design.site_time_state_matrix(), q)
        tests = []
        for test, matrix in [
            ("time_x_network", pooled_network),
            ("time_x_network_x_state", profile_state),
            ("site_x_time_x_network", site_network),
            ("site_x_time_x_state_x_network", site_state_network),
        ]:
            stat, df, p = wald_matrix(beta, cov, matrix)
            tests.append({"scenario": spec["scenario"], "model": "NETWORK_PROFILE", "test": test,
                          "chi2": stat, "df": df, "p": p})

        def network_vector(weight, c):
            return np.kron(np.asarray(weight) @ basis, np.asarray(c))

        network_rows = []
        q_network = norm.ppf(1 - .05 / (2 * K * len(STATES)))
        for network_index, network in enumerate(ELIGIBLE_NETWORKS):
            weight = np.zeros(K); weight[network_index] = 1.0
            for state_index, state in enumerate(STATES):
                vector = network_vector(weight, c_state[state_index])
                estimate, se, z, p = scalar_contrast(beta, cov, vector)
                network_rows.append({
                    "scenario": spec["scenario"], "network": network, "state": state,
                    "relative_profile_slope_per_5y": estimate, "se": se, "p_unadjusted": p,
                    "simultaneous_ci_low": estimate - q_network * se,
                    "simultaneous_ci_high": estimate + q_network * se,
                    "interval_family": f"Bonferroni across {K} networks x 4 states",
                })

        contrast_test_rows, contrast_state_rows = [], []
        for definition in planned_contrast_definitions:
            weight = contrast_weight(definition)
            if weight is None:
                continue
            state_matrix = np.vstack([network_vector(weight, c) for c in c_state])
            for test, matrix in [
                ("all_four_state_contrast_slopes_zero", state_matrix),
                ("contrast_state_heterogeneity", state_matrix[:3] - state_matrix[3]),
                ("contrast_site_x_time", np.vstack([network_vector(weight, row) for row in design.site_time_matrix()])
                    if design.site_time_matrix().shape[0] else np.zeros((0, len(beta)))),
                ("contrast_site_x_time_x_state", np.vstack([network_vector(weight, row) for row in design.site_time_state_matrix()])
                    if design.site_time_state_matrix().shape[0] else np.zeros((0, len(beta)))),
            ]:
                stat, df, p = wald_matrix(beta, cov, matrix)
                contrast_test_rows.append({"scenario": spec["scenario"], "contrast": definition["contrast"],
                                           "test": test, "chi2": stat, "df": df, "p": p})
            omnibus_p = contrast_test_rows[-4]["p"]
            if np.isfinite(omnibus_p) and omnibus_p < .05:
                estimates = state_matrix @ beta
                state_cov = state_matrix @ cov @ state_matrix.T
                se = np.sqrt(np.maximum(0, np.diag(state_cov)))
                pvals = 2 * norm.sf(np.abs(estimates / se))
                pholm = holm_adjust(pvals)
                q4 = norm.ppf(1 - .05 / 8)
                contrast_state_rows.extend([{
                    "scenario": spec["scenario"], "contrast": definition["contrast"], "state": state,
                    "slope_per_5y": float(estimates[j]), "se": float(se[j]), "p_unadjusted": float(pvals[j]),
                    "p_holm_four_states": float(pholm[j]),
                    "simultaneous_ci_low": float(estimates[j] - q4 * se[j]),
                    "simultaneous_ci_high": float(estimates[j] + q4 * se[j]),
                } for j, state in enumerate(STATES)])

        summary = {**spec, "model": "NETWORK_PROFILE", "fit_status": "PASS", "status_detail": "",
                   "rows": fit["row_n"], "participants": fit["participant_n"], "pedigrees": fit["pedigree_n"],
                   "eligible_networks": K, "profile_coordinates": q, "design_columns": len(design.names),
                   "design_rank": fit["design_rank"], "optimizer_success": fit["optimizer_success"],
                   "optimizer_nfev": fit["optimizer_nfev"], "tau_participant": fit["tau_participant"],
                   "tau_participant_state": fit["tau_participant_state"]}
        result = {"summary": summary, "tests": tests, "network": network_rows,
                  "contrast_tests": contrast_test_rows, "contrast_state": contrast_state_rows,
                  "payload": {"coef": beta, "cov": cov, "basis": basis, "names": design.names}}
        checkpoint.write_text(json.dumps({"fingerprint": fp, "result": result}, default=jsonable, indent=2) + "\n")
        return result
    except Exception as exc:
        return {"summary": {**spec, "model": "NETWORK_PROFILE", "fit_status": "FIT_ERROR",
                            "status_detail": f"{type(exc).__name__}: {exc}"}, "tests": [], "network": [],
                "contrast_tests": [], "contrast_state": [], "payload": None}


common_results = [common_fit(spec) for spec in tqdm(SCENARIOS, desc="Common-level fits")]
profile_results = [profile_fit(spec) for spec in tqdm(SCENARIOS, desc="Joint network-profile fits")]

model_summary = pd.DataFrame([result["summary"] for result in common_results + profile_results])
common_tests = pd.DataFrame([row for result in common_results for row in result["tests"]])
common_state_slopes = pd.DataFrame([row for result in common_results for row in result["state"]])
profile_tests = pd.DataFrame([row for result in profile_results for row in result["tests"]])
network_profile_slopes = pd.DataFrame([row for result in profile_results for row in result["network"]])
planned_contrast_tests = pd.DataFrame([row for result in profile_results for row in result["contrast_tests"]])
planned_contrast_state_slopes = pd.DataFrame([row for result in profile_results for row in result["contrast_state"]])

critical = model_summary.loc[model_summary["scenario"].eq("FULL_PRIMARY")]
if len(critical) != 2 or not critical["fit_status"].eq("PASS").all():
    print(critical)
    raise AssertionError("The primary common-level or joint profile fit failed")
print(model_summary)
print(common_tests)
print(profile_tests)
print(planned_contrast_tests)

primary_profile_family = profile_tests.loc[
    profile_tests["scenario"].eq("FULL_PRIMARY")
    & profile_tests["test"].isin(["time_x_network", "time_x_network_x_state"])
].copy()
primary_profile_family["p_holm_two_profile_tests"] = holm_adjust(primary_profile_family["p"])

planned_family = planned_contrast_tests.loc[
    planned_contrast_tests["scenario"].eq("FULL_PRIMARY")
    & planned_contrast_tests["test"].eq("all_four_state_contrast_slopes_zero")
].copy()
planned_family["p_holm_planned_contrasts"] = holm_adjust(planned_family["p"])

primary_planned_state_slopes = planned_contrast_state_slopes.loc[
    planned_contrast_state_slopes["scenario"].eq("FULL_PRIMARY")
].merge(
    planned_family[["contrast", "p_holm_planned_contrasts"]],
    on="contrast", how="left", validate="many_to_one",
)
primary_planned_state_slopes["reportable_after_planned_family_holm"] = (
    primary_planned_state_slopes["p_holm_planned_contrasts"] < .05
)
primary_planned_state_slopes = primary_planned_state_slopes.loc[
    primary_planned_state_slopes["reportable_after_planned_family_holm"]
].copy()

site_interpretation = pd.concat([
    common_tests.loc[
        common_tests["scenario"].eq("FULL_PRIMARY")
        & common_tests["test"].isin(["common_level_site_x_time", "common_level_site_x_time_x_state"])
    ].assign(scope="COMMON_LEVEL"),
    profile_tests.loc[
        profile_tests["scenario"].eq("FULL_PRIMARY")
        & profile_tests["test"].isin(["site_x_time_x_network", "site_x_time_x_state_x_network"])
    ].assign(scope="WHOLE_PROFILE"),
], ignore_index=True, sort=False)
contrast_site = planned_contrast_tests.loc[
    planned_contrast_tests["scenario"].eq("FULL_PRIMARY")
    & planned_contrast_tests["test"].isin(["contrast_site_x_time", "contrast_site_x_time_x_state"])
].copy()

def site_scope_row(scope, table):
    finite = pd.to_numeric(table["p"], errors="coerce").dropna()
    return {
        "site_scope": scope,
        "minimum_direct_site_p": float(finite.min()) if len(finite) else np.nan,
        "site_moderation_detected": bool((finite < .05).any()) if len(finite) else np.nan,
        "interpretation": "Direct site tests are bounded diagnostics; absence of detection is not site equivalence.",
    }

separate_site_decisions = pd.DataFrame([
    site_scope_row("COMMON_LEVEL", site_interpretation.loc[site_interpretation["scope"].eq("COMMON_LEVEL")]),
    site_scope_row("WHOLE_NETWORK_PROFILE", site_interpretation.loc[site_interpretation["scope"].eq("WHOLE_PROFILE")]),
    site_scope_row("FP_MINUS_MEAN_DMN_CO", contrast_site.loc[contrast_site["contrast"].eq("FP_MINUS_MEAN_DMN_CO")]),
    site_scope_row("DMN_MINUS_CO", contrast_site.loc[contrast_site["contrast"].eq("DMN_MINUS_CO")]),
])

full_network_slopes = network_profile_slopes.loc[network_profile_slopes["scenario"].eq("FULL_PRIMARY")].copy()
full_network_rank = full_network_slopes.copy()
full_network_rank["absolute_slope"] = full_network_rank["relative_profile_slope_per_5y"].abs()
full_network_rank["rank_within_state"] = full_network_rank.groupby("state")["absolute_slope"].rank(
    method="min", ascending=False
)

sensitivity_network_slopes = network_profile_slopes.loc[~network_profile_slopes["scenario"].eq("FULL_PRIMARY")].copy()
sensitivity_contrasts = planned_contrast_state_slopes.loc[
    ~planned_contrast_state_slopes["scenario"].eq("FULL_PRIMARY")
].copy()

def test_p(table, scenario, test):
    row = table.loc[table["scenario"].eq(scenario) & table["test"].eq(test)]
    return float(row["p"].iloc[0]) if len(row) else np.nan

profile_main_p = float(primary_profile_family.loc[primary_profile_family["test"].eq("time_x_network"),
                                                  "p_holm_two_profile_tests"].iloc[0])
profile_state_p = float(primary_profile_family.loc[primary_profile_family["test"].eq("time_x_network_x_state"),
                                                   "p_holm_two_profile_tests"].iloc[0])
common_p = test_p(common_tests, "FULL_PRIMARY", "all_four_common_level_slopes_zero")

def planned_p(name):
    row = planned_family.loc[planned_family["contrast"].eq(name)]
    return float(row["p_holm_planned_contrasts"].iloc[0]) if len(row) else np.nan

fp_p = planned_p("FP_MINUS_MEAN_DMN_CO")
dmn_co_p = planned_p("DMN_MINUS_CO")
assoc_p = planned_p("ASSOCIATION_MINUS_SENSORY")

if profile_main_p < .05:
    profile_decision = "BROAD_NETWORK_PROFILE_REDISTRIBUTION"
    if profile_state_p < .05:
        profile_decision += "__STATE_DEPENDENT"
else:
    profile_decision = "NO_JOINT_PROFILE_REDISTRIBUTION_DETECTED"

if fp_p < .05 and dmn_co_p < .05:
    focal_decision = "FP_DIFFERS_FROM_DMN_CO_MEAN_AND_DMN_DIFFERS_FROM_CO"
elif fp_p < .05:
    focal_decision = "FP_DIFFERS_FROM_DMN_CO_MEAN_WITHOUT_DETECTED_DMN_CO_DIFFERENCE"
elif dmn_co_p < .05:
    focal_decision = "DMN_CO_DIFFERENCE_WITHOUT_FP_VERSUS_MEAN_DMN_CO_SIGNAL"
else:
    focal_decision = "NO_PLANNED_THREE_NETWORK_CONTRAST_SURVIVES"

decision_table = pd.DataFrame([
    {"dimension": "common_covariance_level", "decision": "CHANGE_DETECTED" if common_p < .05 else "NO_CHANGE_DETECTED",
     "key_p": common_p},
    {"dimension": "whole_network_profile", "decision": profile_decision, "key_p": profile_main_p,
     },
    {"dimension": "focal_DMN_CO_FP", "decision": focal_decision, "key_p": fp_p,
     },
    {"dimension": "association_vs_sensory", "decision": (
        "AVAILABLE_SIGNIFICANT" if np.isfinite(assoc_p) and assoc_p < .05 else
        "AVAILABLE_NOT_SIGNIFICANT" if np.isfinite(assoc_p) else "UNAVAILABLE_ELIGIBILITY_GATE"
     ), "key_p": assoc_p},
    {"dimension": "amplitude_standardization", "decision": "SENSITIVITY_COMPLETED", "key_p": test_p(profile_tests, "AMPLITUDE_STANDARDIZED", "time_x_network"),
     },
    {"dimension": "partial_covariance", "decision": "ROW_MATCHED_SENSITIVITY_COMPLETED", "key_p": test_p(profile_tests, "PARTIAL_MATCHED", "time_x_network"),
     },
    {"dimension": "site", "decision": "SEPARATE_LEVEL_PROFILE_AND_CONTRAST_TESTS",
     "key_p": np.nan},
])

print(primary_profile_family)
print(planned_family)
print(site_interpretation)
print(contrast_site)
print(separate_site_decisions)
print(full_network_rank)
print(decision_table)

# Common-level state slopes.
common_plot = common_state_slopes.loc[common_state_slopes["scenario"].isin([
    "FULL_PRIMARY", "AMPLITUDE_STANDARDIZED", "FULL_PARTIAL_MATCHED", "PARTIAL_MATCHED", "SAME_SCANNER"
])].copy()
if len(common_plot):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.pointplot(data=common_plot, x="state", y="slope_per_5y", hue="scenario", order=STATES,
                  errorbar=None, dodge=.45, ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Common within-network log-covariance level change")
    ax.set_ylabel("G slope per 5 years"); ax.set_xlabel("state")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "common_level_state_slopes.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "common_level_state_slopes.pdf", bbox_inches="tight")
    plt.close("all")

# Full-covariance relative-profile heatmap.
heat = full_network_slopes.pivot(index="network", columns="state", values="relative_profile_slope_per_5y")
heat = heat.reindex(index=ELIGIBLE_NETWORKS, columns=STATES)
fig, ax = plt.subplots(figsize=(8.2, max(5.5, .42 * K)))
sns.heatmap(heat, cmap="vlag", center=0, annot=True, fmt=".3f", linewidths=.3, ax=ax,
            cbar_kws={"label": "relative-profile slope per 5 years"})
ax.set_title("Full-covariance longitudinal network-profile redistribution")
fig.tight_layout()
fig.savefig(FIG_DIR / "full_covariance_network_profile_heatmap.png", dpi=220, bbox_inches="tight")
fig.savefig(FIG_DIR / "full_covariance_network_profile_heatmap.pdf", bbox_inches="tight")
plt.close("all")

# Planned contrast forest for the primary profile.
contrast_plot = primary_planned_state_slopes.copy()
if len(contrast_plot):
    contrasts = [definition["contrast"] for definition in planned_contrast_definitions]
    fig, axes = plt.subplots(1, len(contrasts), figsize=(5 * len(contrasts), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, contrast in zip(axes, contrasts):
        part = contrast_plot.loc[contrast_plot["contrast"].eq(contrast)].set_index("state").reindex(STATES).reset_index()
        y = np.arange(len(STATES))
        if len(part) and part["slope_per_5y"].notna().any():
            ax.errorbar(part["slope_per_5y"], y,
                        xerr=np.vstack([part["slope_per_5y"] - part["simultaneous_ci_low"],
                                        part["simultaneous_ci_high"] - part["slope_per_5y"]]),
                        fmt="o", capsize=3)
        ax.axvline(0, color="black", lw=1)
        ax.set_title(contrast.replace("_", " "))
        ax.set_yticks(y); ax.set_yticklabels(STATES)
        ax.set_xlabel("slope per 5 years")
    fig.suptitle("Planned network-profile contrasts (simultaneous 95% intervals)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "planned_profile_contrast_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "planned_profile_contrast_forest.pdf", bbox_inches="tight")
    plt.close("all")

# Raw versus amplitude-standardized profile.
raw_amp = network_profile_slopes.loc[network_profile_slopes["scenario"].isin([
    "RAW_AMPLITUDE_MATCHED", "AMPLITUDE_STANDARDIZED"
])].copy()
if len(raw_amp):
    g = sns.catplot(data=raw_amp, x="state", y="relative_profile_slope_per_5y", hue="scenario",
                    col="network", col_wrap=4, kind="point", order=STATES, errorbar=None,
                    height=3.1, aspect=1.05, sharey=False)
    for ax in g.axes.flat:
        ax.axhline(0, color="black", lw=.8)
        ax.tick_params(axis="x", rotation=30)
    g.set_axis_labels("state", "relative slope / 5y")
    g.fig.subplots_adjust(top=.92)
    g.fig.suptitle("Raw versus amplitude-standardized network profiles")
    g.savefig(FIG_DIR / "raw_vs_amplitude_standardized_profile.png", dpi=200, bbox_inches="tight")
    g.savefig(FIG_DIR / "raw_vs_amplitude_standardized_profile.pdf", bbox_inches="tight")
    plt.close("all")

exports = {
    "source_manifest.csv": source_manifest,
    "full_node_sources.csv": full_node_sources,
    "partial_node_sources.csv": partial_node_sources,
    "parcel_counts.csv": parcel_counts,
    "network_state_eligibility.csv": network_state_eligibility,
    "network_eligibility_decision.csv": network_eligibility,
    "profile_identity_audit.csv": profile_identity_audit,
    "cohort_summary.csv": cohort_summary,
    "amplitude_coverage.csv": amplitude_coverage,
    "partial_positivity_coverage.csv": partial_positivity_coverage,
    "model_summary.csv": model_summary,
    "common_level_tests.csv": common_tests,
    "common_level_state_slopes.csv": common_state_slopes,
    "profile_omnibus_tests.csv": profile_tests,
    "network_profile_state_slopes.csv": network_profile_slopes,
    "full_network_profile_rank.csv": full_network_rank,
    "planned_contrast_tests.csv": planned_contrast_tests,
    "planned_contrast_state_slopes_gated.csv": planned_contrast_state_slopes,
    "primary_profile_holm_family.csv": primary_profile_family,
    "planned_contrast_holm_family.csv": planned_family,
    "primary_planned_contrast_state_slopes_after_holm.csv": primary_planned_state_slopes,
    "separate_site_interpretation.csv": site_interpretation,
    "planned_contrast_site_tests.csv": contrast_site,
    "separate_site_decisions.csv": separate_site_decisions,
    "sensitivity_network_slopes.csv": sensitivity_network_slopes,
    "sensitivity_planned_contrasts.csv": sensitivity_contrasts,
    "final_claim_decision.csv": decision_table,
}
for filename, table in exports.items():
    table.to_csv(TABLE_DIR / filename, index=False)

run_manifest = {
    "script": "02_primary_profile_and_measurement.py",
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(), "fast_mode": FAST_MODE,
    "frozen_repeat_participants": len(frozen_repeat_ids), "eligible_networks": ELIGIBLE_NETWORKS,
    "profile_participants": int(profile_data["id"].nunique()), "profile_rows": len(profile_data),
    "association_sensory_available": association_sensory_available,
    "critical_failures": int((critical["fit_status"] != "PASS").sum()),
    "optimizer_warnings": int((model_summary.loc[model_summary["fit_status"].eq("PASS"), "optimizer_success"] == False).sum()),
    "amplitude_standardized_called_correlation": False,
    "nonpositive_values_imputed": False,
}
(OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

print(decision_table)
print(f"Saved {len(exports)} tables and four figure families to {OUTPUT_DIR}")



# Pair-amplitude measurement analysis
BOOTSTRAP_B = int(ARGS.bootstrap_b if ARGS.bootstrap_b is not None else (999 if FAST_MODE else 4999))
BOOTSTRAP_SEED = int(os.environ.get("AABC2_GATE1_BOOTSTRAP_SEED", "20260907"))
IDENTITY_TOL = float(os.environ.get("AABC2_GATE1_IDENTITY_TOL", "1e-10"))
REPRO_TOL = float(os.environ.get("AABC2_GATE1_REPRO_TOL", "5e-8"))

# Fixed inference rules. An external rule file may be supplied with a SHA-256 check.
FROZEN_GATE1_RULES = {
    "residual_fraction": {
        "definition": "q_s=abs(beta_std_s/beta_raw_s) under one fixed common working covariance",
        "Qmax_definition": "max_s(q_s)",
        "threshold": 0.50,
        "inference": "pedigree Gaussian multiplier; shared multipliers for raw and standardized; 95% simultaneous q intervals; one-sided 95% Qmax upper bound",
    },
    "correlation": {
        "source_semantics_authoritative": True,
        "numerical_range_is_diagnostic_only": True,
        "raw_r": "use directly",
        "fisher_z": "apply tanh entrywise before aggregation",
        "uncertain": "UNAVAILABLE",
        "never_reconstruct_from_covariance_self_entries": True,
    },
}
RULES_CANONICAL = json.dumps(FROZEN_GATE1_RULES, sort_keys=True, separators=(",", ":"))
EMBEDDED_RULES_SHA256 = hashlib.sha256(RULES_CANONICAL.encode()).hexdigest()

external_rule_candidates = sorted(
    p.resolve() for p in DATA_DIR.rglob("paper_inference_gate_rules.json")
    if p.is_file() and OUTPUT_DIR not in p.parents
)
EXTERNAL_RULES_PATH = external_rule_candidates[0] if len(external_rule_candidates) == 1 else None
EXPECTED_EXTERNAL_RULES_SHA256 = os.environ.get("AABC2_GATE_RULES_SHA256", "").strip().lower()
if len(external_rule_candidates) > 1:
    raise RuntimeError(f"Multiple paper_inference_gate_rules.json files found: {external_rule_candidates}")
if EXTERNAL_RULES_PATH is not None:
    external_sha = sha256_file(EXTERNAL_RULES_PATH) if "sha256_file" in globals() else hashlib.sha256(EXTERNAL_RULES_PATH.read_bytes()).hexdigest()
    if EXPECTED_EXTERNAL_RULES_SHA256 and external_sha != EXPECTED_EXTERNAL_RULES_SHA256:
        raise AssertionError(
            f"Gate-rule hash mismatch: expected {EXPECTED_EXTERNAL_RULES_SHA256}, observed {external_sha}"
        )
    EXTERNAL_RULES_STATUS = "PRESENT_HASH_VERIFIED" if EXPECTED_EXTERNAL_RULES_SHA256 else "PRESENT_HASH_RECORDED"
else:
    external_sha = ""
    EXTERNAL_RULES_STATUS = "ABSENT__EMBEDDED_FROZEN_GATE1_RULES_USED"

rules_snapshot = {
    "embedded_gate1_rules_sha256": EMBEDDED_RULES_SHA256,
    "external_rules_status": EXTERNAL_RULES_STATUS,
    "external_rules_path": str(EXTERNAL_RULES_PATH) if EXTERNAL_RULES_PATH else "",
    "external_rules_sha256": external_sha,
    "rules": FROZEN_GATE1_RULES,
}
(OUTPUT_DIR / "gate1_rules_snapshot.json").write_text(json.dumps(rules_snapshot, indent=2) + "\n")
print({
    "BOOTSTRAP_B": BOOTSTRAP_B,
    "embedded_gate1_rules_sha256": EMBEDDED_RULES_SHA256,
    "external_rules_status": EXTERNAL_RULES_STATUS,
})

legacy_run_manifest = {"eligible_networks": list(ELIGIBLE_NETWORKS)}
legacy_common_slopes = common_state_slopes.copy()
legacy_state_order = list(STATES)

# The exact pair-amplitude identity is reconstructed on the amplitude-complete matched rows.
gate1_frame = amplitude_data.copy()
logp_cols = []
l_cols = []
for network in ELIGIBLE_NETWORKS:
    lp = f"logP__{network}"
    gate1_frame[lp] = np.log(pd.to_numeric(gate1_frame[f"P__{network}"], errors="coerce"))
    logp_cols.append(lp)
    l_cols.append(f"L__{network}")

gate1_frame["G_RAW"] = gate1_frame[l_cols].mean(axis=1)
gate1_frame["G_A"] = gate1_frame[logp_cols].mean(axis=1)
gate1_frame["G_STD"] = (gate1_frame[l_cols].to_numpy(float) - gate1_frame[logp_cols].to_numpy(float)).mean(axis=1)

identity_error = gate1_frame["G_RAW"] - (gate1_frame["G_A"] + gate1_frame["G_STD"])
OBSERVATION_IDENTITY_MAX_ERROR = float(np.nanmax(np.abs(identity_error)))
assert OBSERVATION_IDENTITY_MAX_ERROR < IDENTITY_TOL, OBSERVATION_IDENTITY_MAX_ERROR

# Cross-check against the inherited derived columns, without using them to define the new outcomes.
raw_g_error = float(np.nanmax(np.abs(gate1_frame["G_RAW"] - pd.to_numeric(gate1_frame["G"], errors="coerce"))))
std_lookup = amplitude_standardized[["id_event", "state", "G_STD"]].copy()
std_check = gate1_frame[["id_event", "state", "G_STD"]].merge(
    std_lookup.rename(columns={"G_STD": "G_STD_legacy"}),
    on=["id_event", "state"], how="left", validate="one_to_one"
)
std_g_error = float(np.nanmax(np.abs(std_check["G_STD"] - std_check["G_STD_legacy"])))
assert raw_g_error < IDENTITY_TOL, raw_g_error
assert std_g_error < IDENTITY_TOL, std_g_error

# Exact row fingerprint and amplitude source provenance.
row_keys = gate1_frame[["id", "id_event", "state"]].astype(str).sort_values(["id", "id_event", "state"])
AMPLITUDE_MATCHED_ROW_SHA256 = hashlib.sha256(row_keys.to_csv(index=False).encode()).hexdigest()

amplitude_sources = source_manifest.loc[source_manifest["role"].astype(str).str.contains("FullAmplitudes", na=False)].copy()
assert amplitude_sources["role"].nunique() == 4
assert set(amplitude_sources["role"]) == {f"{s}_FullAmplitudes" for s in STATES}

provenance_audit = pd.DataFrame([
    {"item": "parent_repeat_participants", "observed": int(profile_data["id"].nunique()), "expected": EXPECTED_FROZEN_REPEAT_PARTICIPANTS, "pass": int(profile_data["id"].nunique()) == EXPECTED_FROZEN_REPEAT_PARTICIPANTS},
    {"item": "eligible_network_n", "observed": len(ELIGIBLE_NETWORKS), "expected": 10, "pass": len(ELIGIBLE_NETWORKS) == 10},
    {"item": "eligible_network_order", "observed": json.dumps(ELIGIBLE_NETWORKS), "expected": json.dumps(legacy_run_manifest["eligible_networks"]), "pass": list(ELIGIBLE_NETWORKS) == list(legacy_run_manifest["eligible_networks"])},
    {"item": "state_order", "observed": json.dumps(STATES), "expected": json.dumps(legacy_state_order), "pass": STATES == legacy_state_order},
    {"item": "amplitude_matched_rows", "observed": len(gate1_frame), "expected": "reconstructed from frozen pipeline", "pass": len(gate1_frame) > 0},
    {"item": "amplitude_matched_participants", "observed": int(gate1_frame["id"].nunique()), "expected": ">=2 sessions after amplitude completeness", "pass": int(gate1_frame["id"].nunique()) > 0},
    {"item": "FullAmplitudes_sources", "observed": int(amplitude_sources["role"].nunique()), "expected": 4, "pass": amplitude_sources["role"].nunique() == 4},
    {"item": "observation_identity_max_error", "observed": OBSERVATION_IDENTITY_MAX_ERROR, "expected": f"<{IDENTITY_TOL}", "pass": OBSERVATION_IDENTITY_MAX_ERROR < IDENTITY_TOL},
    {"item": "raw_G_reproduction_max_error", "observed": raw_g_error, "expected": f"<{IDENTITY_TOL}", "pass": raw_g_error < IDENTITY_TOL},
    {"item": "std_G_reproduction_max_error", "observed": std_g_error, "expected": f"<{IDENTITY_TOL}", "pass": std_g_error < IDENTITY_TOL},
])
assert provenance_audit["pass"].all(), provenance_audit.loc[~provenance_audit["pass"]]
print(provenance_audit)
print("Amplitude-matched row fingerprint:", AMPLITUDE_MATCHED_ROW_SHA256)
print(amplitude_sources[["role", "path", "sha256"]])

# ---------------------------
# Part B: ordinary outcome-specific working-covariance fits
# ---------------------------

MODEL_REQUIRED = ["G_RAW", "G_A", "G_STD", "age0_c5", "age0_c10", "t5"]
model_frame = gate1_frame.loc[
    np.isfinite(gate1_frame[MODEL_REQUIRED].to_numpy(float)).all(axis=1)
].copy()
assert len(model_frame) == len(gate1_frame), "Unexpected model-row loss after exact amplitude matching."

design = DirectStateDesign().fit(model_frame)
X = design.transform(model_frame)
C_STATE = design.state_slope_matrix()
C_HETERO = C_STATE[:3] - C_STATE[3]
SIMULTANEOUS_Z4 = norm.ppf(1 - .05 / (2 * len(STATES)))

ORDINARY_OUTCOMES = OrderedDict([
    ("RAW_MATCHED", "G_RAW"),
    ("PAIR_AMPLITUDE_COMPONENT", "G_A"),
    ("AMPLITUDE_STANDARDIZED", "G_STD"),
])

ordinary_fits = {}
ordinary_rows = []
for scenario, outcome_col in ORDINARY_OUTCOMES.items():
    fit = fit_direct_profile_covariance(
        X, model_frame[outcome_col],
        model_frame["id"], model_frame["state"], model_frame["pedigree_group"]
    )
    if fit is None:
        raise RuntimeError(f"Ordinary fit failed for {scenario}")
    ordinary_fits[scenario] = fit
    beta, cov = fit["coef"], fit["robust_cov"]
    slopes = C_STATE @ beta
    slope_cov = C_STATE @ cov @ C_STATE.T
    slope_se = np.sqrt(np.maximum(0, np.diag(slope_cov)))
    omnibus = wald_matrix(beta, cov, C_STATE)
    hetero = wald_matrix(beta, cov, C_HETERO)
    residual_var = float(fit["sigma_e2"])
    participant_var = residual_var * float(fit["tau_participant"])
    participant_state_var = residual_var * float(fit["tau_participant_state"])
    for j, state in enumerate(STATES):
        z = slopes[j] / slope_se[j] if slope_se[j] > 0 else np.nan
        ordinary_rows.append({
            "scenario": scenario,
            "state": state,
            "slope_per_5y": float(slopes[j]),
            "se": float(slope_se[j]),
            "simultaneous_ci_low": float(slopes[j] - SIMULTANEOUS_Z4 * slope_se[j]),
            "simultaneous_ci_high": float(slopes[j] + SIMULTANEOUS_Z4 * slope_se[j]),
            "p_unadjusted": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "four_state_chi2": omnibus[0], "four_state_df": omnibus[1], "four_state_p": omnibus[2],
            "state_heterogeneity_chi2": hetero[0], "state_heterogeneity_df": hetero[1], "state_heterogeneity_p": hetero[2],
            "tau_participant_over_residual": float(fit["tau_participant"]),
            "tau_participant_state_over_residual": float(fit["tau_participant_state"]),
            "residual_variance": residual_var,
            "participant_variance": participant_var,
            "participant_state_variance": participant_state_var,
            "rows": int(fit["row_n"]), "participants": int(fit["participant_n"]), "pedigrees": int(fit["pedigree_n"]),
            "optimizer_success": bool(fit["optimizer_success"]),
            "interval_family": "Bonferroni simultaneous across four state slopes",
        })

gate1_ordinary_common_models = pd.DataFrame(ordinary_rows)

# Demonstrate, rather than assume, non-additivity under separately re-estimated working covariance.
ordinary_coef_identity = (
    ordinary_fits["RAW_MATCHED"]["coef"]
    - ordinary_fits["PAIR_AMPLITUDE_COMPONENT"]["coef"]
    - ordinary_fits["AMPLITUDE_STANDARDIZED"]["coef"]
)
ordinary_state_identity = (
    C_STATE @ ordinary_fits["RAW_MATCHED"]["coef"]
    - C_STATE @ ordinary_fits["PAIR_AMPLITUDE_COMPONENT"]["coef"]
    - C_STATE @ ordinary_fits["AMPLITUDE_STANDARDIZED"]["coef"]
)
ORDINARY_COEF_NONADDITIVITY_MAX = float(np.max(np.abs(ordinary_coef_identity)))
ORDINARY_STATE_NONADDITIVITY_MAX = float(np.max(np.abs(ordinary_state_identity)))

# Check the corresponding common-model estimates on identical rows.
legacy_compare = []
for old_scenario, new_scenario in [
    ("RAW_AMPLITUDE_MATCHED", "RAW_MATCHED"),
    ("AMPLITUDE_STANDARDIZED", "AMPLITUDE_STANDARDIZED"),
]:
    old = legacy_common_slopes.loc[legacy_common_slopes["scenario"].eq(old_scenario), ["state", "slope_per_5y"]]
    new = gate1_ordinary_common_models.loc[gate1_ordinary_common_models["scenario"].eq(new_scenario), ["state", "slope_per_5y"]]
    cmp = old.merge(new, on="state", suffixes=("_legacy", "_gate1"), validate="one_to_one")
    cmp["absolute_difference"] = (cmp["slope_per_5y_legacy"] - cmp["slope_per_5y_gate1"]).abs()
    cmp["legacy_scenario"] = old_scenario
    legacy_compare.append(cmp)
legacy_reproduction = pd.concat(legacy_compare, ignore_index=True)
LEGACY_SLOPE_REPRO_MAX_ERROR = float(legacy_reproduction["absolute_difference"].max())
assert LEGACY_SLOPE_REPRO_MAX_ERROR < REPRO_TOL, legacy_reproduction

gate1_ordinary_common_models.to_csv(TABLE_DIR / "gate1_ordinary_common_models.csv", index=False)
print(gate1_ordinary_common_models)
print(legacy_reproduction)
print({
    "ordinary_coef_nonadditivity_max": ORDINARY_COEF_NONADDITIVITY_MAX,
    "ordinary_state_nonadditivity_max": ORDINARY_STATE_NONADDITIVITY_MAX,
    "legacy_slope_reproduction_max_error": LEGACY_SLOPE_REPRO_MAX_ERROR,
})

# ---------------------------
# Part C: exact common-weight GLS decomposition
# ---------------------------

def fit_fixed_working_covariance_gls(x, y, participant, state, pedigree, tau_participant, tau_participant_state):
    """GLS with one frozen covariance SHAPE, plus pedigree sandwich influence vectors."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    participant = np.asarray(participant).astype(str)
    state = np.asarray(state).astype(str)
    pedigree = np.asarray(pedigree).astype(str)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if not valid.all():
        raise AssertionError("Fixed-weight fits must use the identical pre-frozen rows and X.")
    blocks = []
    for pid in np.unique(participant):
        idx = np.flatnonzero(participant == pid)
        z_state = pd.get_dummies(pd.Series(state[idx]), dtype=float).to_numpy()
        A = (
            np.eye(len(idx))
            + float(tau_participant) * np.ones((len(idx), len(idx)))
            + float(tau_participant_state) * (z_state @ z_state.T)
        )
        blocks.append((idx, A, str(pedigree[idx[0]])))

    xtvx = np.zeros((x.shape[1], x.shape[1]))
    xtvy = np.zeros(x.shape[1])
    for idx, A, _ in blocks:
        AX = np.linalg.solve(A, x[idx])
        Ay = np.linalg.solve(A, y[idx])
        xtvx += x[idx].T @ AX
        xtvy += x[idx].T @ Ay
    bread_inv = np.linalg.pinv(xtvx, rcond=1e-11)
    beta = bread_inv @ xtvy
    residual = y - x @ beta

    pedigree_scores = {}
    q = 0.0
    for idx, A, pg in blocks:
        Aresid = np.linalg.solve(A, residual[idx])
        q += float(residual[idx] @ Aresid)
        score = x[idx].T @ Aresid
        pedigree_scores[pg] = pedigree_scores.get(pg, np.zeros(x.shape[1])) + score

    g = len(pedigree_scores)
    correction = (g / max(g - 1, 1)) * ((len(y) - 1) / max(len(y) - len(beta), 1))
    influences = {
        pg: math.sqrt(correction) * (bread_inv @ score)
        for pg, score in pedigree_scores.items()
    }
    robust_cov = sum(np.outer(v, v) for v in influences.values())
    sigma_e2 = q / max(len(y) - len(beta), 1)
    return {
        "coef": beta,
        "robust_cov": robust_cov,
        "influences": influences,
        "sigma_e2": float(sigma_e2),
        "tau_participant": float(tau_participant),
        "tau_participant_state": float(tau_participant_state),
        "row_n": len(y),
        "participant_n": len(np.unique(participant)),
        "pedigree_n": g,
    }

RAW_TAU_P = ordinary_fits["RAW_MATCHED"]["tau_participant"]
RAW_TAU_S = ordinary_fits["RAW_MATCHED"]["tau_participant_state"]

fixed_fits = {}
for scenario, outcome_col in ORDINARY_OUTCOMES.items():
    fixed_fits[scenario] = fit_fixed_working_covariance_gls(
        X, model_frame[outcome_col],
        model_frame["id"], model_frame["state"], model_frame["pedigree_group"],
        RAW_TAU_P, RAW_TAU_S,
    )

coef_identity_error = (
    fixed_fits["RAW_MATCHED"]["coef"]
    - fixed_fits["PAIR_AMPLITUDE_COMPONENT"]["coef"]
    - fixed_fits["AMPLITUDE_STANDARDIZED"]["coef"]
)
FIXED_COEF_IDENTITY_MAX_ERROR = float(np.max(np.abs(coef_identity_error)))
assert FIXED_COEF_IDENTITY_MAX_ERROR < IDENTITY_TOL, FIXED_COEF_IDENTITY_MAX_ERROR

decomposition_rows = []
for j, term in enumerate(design.names):
    decomposition_rows.append({
        "row_type": "FIXED_EFFECT_COEFFICIENT",
        "term_or_state": term,
        "raw_estimate": float(fixed_fits["RAW_MATCHED"]["coef"][j]),
        "amplitude_component_estimate": float(fixed_fits["PAIR_AMPLITUDE_COMPONENT"]["coef"][j]),
        "standardized_estimate": float(fixed_fits["AMPLITUDE_STANDARDIZED"]["coef"][j]),
        "identity_error": float(coef_identity_error[j]),
        "raw_se": float(math.sqrt(max(0, fixed_fits["RAW_MATCHED"]["robust_cov"][j, j]))),
        "amplitude_component_se": float(math.sqrt(max(0, fixed_fits["PAIR_AMPLITUDE_COMPONENT"]["robust_cov"][j, j]))),
        "standardized_se": float(math.sqrt(max(0, fixed_fits["AMPLITUDE_STANDARDIZED"]["robust_cov"][j, j]))),
        "working_tau_participant_from_raw": float(RAW_TAU_P),
        "working_tau_participant_state_from_raw": float(RAW_TAU_S),
    })

fixed_state = {}
for scenario in ORDINARY_OUTCOMES:
    fit = fixed_fits[scenario]
    fixed_state[scenario] = {
        "estimate": C_STATE @ fit["coef"],
        "cov": C_STATE @ fit["robust_cov"] @ C_STATE.T,
    }

fixed_state_identity = (
    fixed_state["RAW_MATCHED"]["estimate"]
    - fixed_state["PAIR_AMPLITUDE_COMPONENT"]["estimate"]
    - fixed_state["AMPLITUDE_STANDARDIZED"]["estimate"]
)
FIXED_STATE_IDENTITY_MAX_ERROR = float(np.max(np.abs(fixed_state_identity)))
assert FIXED_STATE_IDENTITY_MAX_ERROR < IDENTITY_TOL, FIXED_STATE_IDENTITY_MAX_ERROR

for j, state in enumerate(STATES):
    row = {
        "row_type": "STATE_SLOPE_PER_5Y",
        "term_or_state": state,
        "raw_estimate": float(fixed_state["RAW_MATCHED"]["estimate"][j]),
        "amplitude_component_estimate": float(fixed_state["PAIR_AMPLITUDE_COMPONENT"]["estimate"][j]),
        "standardized_estimate": float(fixed_state["AMPLITUDE_STANDARDIZED"]["estimate"][j]),
        "identity_error": float(fixed_state_identity[j]),
        "raw_se": float(math.sqrt(max(0, fixed_state["RAW_MATCHED"]["cov"][j, j]))),
        "amplitude_component_se": float(math.sqrt(max(0, fixed_state["PAIR_AMPLITUDE_COMPONENT"]["cov"][j, j]))),
        "standardized_se": float(math.sqrt(max(0, fixed_state["AMPLITUDE_STANDARDIZED"]["cov"][j, j]))),
        "working_tau_participant_from_raw": float(RAW_TAU_P),
        "working_tau_participant_state_from_raw": float(RAW_TAU_S),
    }
    decomposition_rows.append(row)

gate1_common_weight_decomposition = pd.DataFrame(decomposition_rows)
gate1_common_weight_decomposition.to_csv(TABLE_DIR / "gate1_common_weight_decomposition.csv", index=False)
print(gate1_common_weight_decomposition.tail(len(STATES)))
print({
    "fixed_coefficient_identity_max_error": FIXED_COEF_IDENTITY_MAX_ERROR,
    "fixed_state_identity_max_error": FIXED_STATE_IDENTITY_MAX_ERROR,
    "interpretation": "ALGEBRAIC_IDENTITY_UNDER_ONE_GLS_OPERATOR_NOT_INDEPENDENT_CORROBORATION",
})

# ---------------------------
# Part D: residual-fraction inference
# ---------------------------

def state_influence_matrix(fit):
    pedigrees = sorted(fit["influences"])
    matrix = np.vstack([C_STATE @ fit["influences"][pg] for pg in pedigrees])
    return pedigrees, matrix

pg_raw, raw_inf = state_influence_matrix(fixed_fits["RAW_MATCHED"])
pg_std, std_inf = state_influence_matrix(fixed_fits["AMPLITUDE_STANDARDIZED"])
assert pg_raw == pg_std, "Raw and standardized fixed-weight fits must have identical pedigree clusters."

raw_point = np.asarray(fixed_state["RAW_MATCHED"]["estimate"], float)
std_point = np.asarray(fixed_state["AMPLITUDE_STANDARDIZED"]["estimate"], float)
q_point = np.abs(std_point / raw_point)
QMAX_POINT = float(np.max(q_point))

rng = np.random.default_rng(BOOTSTRAP_SEED)
q_draws = np.empty((BOOTSTRAP_B, len(STATES)), float)
raw_draws = np.empty_like(q_draws)
std_draws = np.empty_like(q_draws)

# Batched only to keep memory use low on the 8-GB target machine.
batch = 250
for start in tqdm(range(0, BOOTSTRAP_B, batch), desc="Gate 1 pedigree multiplier"):
    stop = min(BOOTSTRAP_B, start + batch)
    xi = rng.standard_normal((stop - start, len(pg_raw)))
    raw_b = raw_point[None, :] + xi @ raw_inf
    std_b = std_point[None, :] + xi @ std_inf
    with np.errstate(divide="ignore", invalid="ignore"):
        q_b = np.abs(std_b / raw_b)
    raw_draws[start:stop] = raw_b
    std_draws[start:stop] = std_b
    q_draws[start:stop] = q_b

# No draw is trimmed. Infinite ratios therefore propagate into the confidence band when
# denominator instability is sufficiently common. Use a max-|t| multiplier family when
# finite scales are estimable; otherwise fall back to the unstudentized max-deviation band.
finite_for_scale = np.where(np.isfinite(q_draws), q_draws, np.nan)
q_scale = np.nanstd(finite_for_scale, axis=0, ddof=1)
if np.isfinite(q_scale).all() and (q_scale > 0).all():
    with np.errstate(divide="ignore", invalid="ignore"):
        tmax = np.max(np.abs(q_draws - q_point[None, :]) / q_scale[None, :], axis=1)
    Q_CRITICAL_95 = float(np.quantile(tmax, .95))
    q_ci_low = np.maximum(0.0, q_point - Q_CRITICAL_95 * q_scale)
    q_ci_high = q_point + Q_CRITICAL_95 * q_scale
    Q_INTERVAL_METHOD = "max-|t| pedigree multiplier simultaneous band across four q_s"
else:
    max_dev = np.max(np.abs(q_draws - q_point[None, :]), axis=1)
    Q_CRITICAL_95 = float(np.quantile(max_dev, .95))
    q_ci_low = np.maximum(0.0, q_point - Q_CRITICAL_95)
    q_ci_high = q_point + Q_CRITICAL_95
    Q_INTERVAL_METHOD = "fallback unstudentized max absolute pedigree-multiplier band"

QMAX_DRAWS = np.max(q_draws, axis=1)
QMAX_DRAW_P95 = float(np.quantile(QMAX_DRAWS, .95))
# A simultaneous upper band for all q_s immediately supplies an upper confidence bound
# for max_s q_s. The bound cannot fall below the point Qmax.
QMAX_U95 = float(np.max(q_ci_high))

numeric_tol = math.sqrt(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(raw_point))))
residual_rows = []
for j, state in enumerate(STATES):
    abs_den = np.abs(raw_draws[:, j])
    sign_cross = np.mean(np.sign(raw_draws[:, j]) != np.sign(raw_point[j]))
    finite_ratio = np.isfinite(q_draws[:, j])
    residual_rows.append({
        "state": state,
        "raw_fixed_weight_slope_per_5y": float(raw_point[j]),
        "standardized_fixed_weight_slope_per_5y": float(std_point[j]),
        "q_point": float(q_point[j]),
        "simultaneous_95_ci_low": float(q_ci_low[j]),
        "simultaneous_95_ci_high": float(q_ci_high[j]),
        "Qmax_point": QMAX_POINT,
        "Qmax_upper_95": QMAX_U95,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_denominator_abs_min_draw": float(np.min(abs_den)),
        "raw_denominator_abs_p025_draw": float(np.quantile(abs_den, .025)),
        "raw_denominator_sign_crossing_fraction": float(sign_cross),
        "raw_denominator_machine_near_zero_fraction": float(np.mean(abs_den <= numeric_tol)),
        "finite_ratio_fraction": float(np.mean(finite_ratio)),
        "draw_trimming": "NONE",
        "simultaneous_interval_method": Q_INTERVAL_METHOD,
        "Qmax_draw_p95_diagnostic": QMAX_DRAW_P95,
    })
residual_rows.append({
    "state": "Q_MAX",
    "raw_fixed_weight_slope_per_5y": np.nan,
    "standardized_fixed_weight_slope_per_5y": np.nan,
    "q_point": QMAX_POINT,
    "simultaneous_95_ci_low": np.nan,
    "simultaneous_95_ci_high": QMAX_U95,
    "Qmax_point": QMAX_POINT,
    "Qmax_upper_95": QMAX_U95,
    "bootstrap_B": BOOTSTRAP_B,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "raw_denominator_abs_min_draw": float(np.min(np.abs(raw_draws))),
    "raw_denominator_abs_p025_draw": float(np.quantile(np.min(np.abs(raw_draws), axis=1), .025)),
    "raw_denominator_sign_crossing_fraction": float(np.max([
        np.mean(np.sign(raw_draws[:, j]) != np.sign(raw_point[j])) for j in range(len(STATES))
    ])),
    "raw_denominator_machine_near_zero_fraction": float(np.mean(np.abs(raw_draws) <= numeric_tol)),
    "finite_ratio_fraction": float(np.mean(np.isfinite(QMAX_DRAWS))),
    "draw_trimming": "NONE",
    "simultaneous_interval_method": "upper edge of the simultaneous four-state q band",
    "Qmax_draw_p95_diagnostic": QMAX_DRAW_P95,
})
gate1_residual_fraction_intervals = pd.DataFrame(residual_rows)
gate1_residual_fraction_intervals.to_csv(TABLE_DIR / "gate1_residual_fraction_intervals.csv", index=False)
print(gate1_residual_fraction_intervals)

# ---------------------------
# Part E: released-correlation semantic gate
# ---------------------------

def resolve_full_correlation_archives():
    rows = []
    mapping = {}
    for state in STATES:
        hits = sorted({
            p.resolve() for p in DATA_DIR.rglob("*FullCorrelationConnectivity.zip")
            if classify_state_from_name(p.name) == state and OUTPUT_DIR not in p.parents
        })
        mapping[state] = hits[0] if len(hits) == 1 else None
        rows.append({
            "state": state, "archive_n": len(hits),
            "archive_path": str(hits[0]) if len(hits) == 1 else "",
            "archive_status": "PRESENT_UNVERIFIED" if len(hits) == 1 else ("ABSENT" if not hits else "AMBIGUOUS"),
        })
    return mapping, pd.DataFrame(rows)

def archive_numeric_diagnostic(path, max_members=3, nrows=100):
    if path is None:
        return {"member_n": 0, "sample_numeric_n": 0, "sample_min": np.nan, "sample_max": np.nan, "sample_fraction_abs_le_1": np.nan}
    values = []
    with ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for member in members[:max_members]:
            with zf.open(member) as stream:
                sample = pd.read_csv(stream, nrows=nrows, low_memory=False)
            id_guess = None
            try:
                id_guess = identify_id_column(sample.columns)
            except Exception:
                pass
            cols = [c for c in sample.columns if c != id_guess]
            if cols:
                arr = sample[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel()
                arr = arr[np.isfinite(arr)]
                if len(arr):
                    values.append(arr)
    joined = np.concatenate(values) if values else np.array([], float)
    return {
        "member_n": len(members) if path is not None else 0,
        "sample_numeric_n": int(len(joined)),
        "sample_min": float(np.min(joined)) if len(joined) else np.nan,
        "sample_max": float(np.max(joined)) if len(joined) else np.nan,
        "sample_fraction_abs_le_1": float(np.mean(np.abs(joined) <= 1)) if len(joined) else np.nan,
    }

correlation_archives, correlation_inventory = resolve_full_correlation_archives()
for col in ["member_n", "sample_numeric_n", "sample_min", "sample_max", "sample_fraction_abs_le_1"]:
    correlation_inventory[col] = [
        archive_numeric_diagnostic(correlation_archives[s]).get(col) for s in STATES
    ]

all_four_corr_archives = bool(correlation_inventory["archive_n"].eq(1).all())
semantics_override = os.environ.get("AABC2_CORRELATION_SEMANTICS", "").strip().upper()
semantics_source_text = os.environ.get("AABC2_CORRELATION_SEMANTICS_SOURCE", "").strip()
allowed_semantics = {"", "RAW_R", "FISHER_Z"}
if semantics_override not in allowed_semantics:
    raise ValueError(f"AABC2_CORRELATION_SEMANTICS must be one of {sorted(allowed_semantics)}")

if not all_four_corr_archives:
    CORRELATION_STATUS = "UNAVAILABLE__FOUR_RELEASED_ARCHIVES_NOT_PRESENT_UNIQUELY"
    CORRELATION_SEMANTICS = ""
    semantics_source_path = None
    semantics_source_sha = ""
elif not semantics_override:
    CORRELATION_STATUS = "UNAVAILABLE__SEMANTICS_UNVERIFIED"
    CORRELATION_SEMANTICS = ""
    semantics_source_path = None
    semantics_source_sha = ""
else:
    if not semantics_source_text:
        raise AssertionError(
            "A correlation-semantics override requires AABC2_CORRELATION_SEMANTICS_SOURCE pointing to authoritative metadata/extraction code."
        )
    semantics_source_path = Path(semantics_source_text).expanduser().resolve()
    if not semantics_source_path.exists():
        raise FileNotFoundError(semantics_source_path)
    semantics_source_sha = sha256_file(semantics_source_path)
    source_text = semantics_source_path.read_text(errors="replace").lower()
    if semantics_override == "RAW_R":
        if not re.search(r"pearson|raw\s+r|correlation coefficient", source_text):
            raise AssertionError("RAW_R override source lacks explicit Pearson/raw-r/correlation-coefficient language.")
    if semantics_override == "FISHER_Z":
        if not re.search(r"fisher|atanh|arctanh", source_text):
            raise AssertionError("FISHER_Z override source lacks explicit Fisher/atanh/arctanh evidence.")
    CORRELATION_STATUS = "AVAILABLE__SEMANTICS_VERIFIED_FROM_SUPPLIED_SOURCE"
    CORRELATION_SEMANTICS = semantics_override

correlation_inventory["overall_analysis_status"] = CORRELATION_STATUS
correlation_inventory["verified_semantics"] = CORRELATION_SEMANTICS
correlation_inventory["semantics_source"] = str(semantics_source_path) if semantics_source_path else ""
correlation_inventory["semantics_source_sha256"] = semantics_source_sha
correlation_inventory["range_diagnostic_is_semantic_evidence"] = False
correlation_inventory["covariance_self_entries_used"] = False
correlation_inventory["transformation_if_used"] = (
    "NONE" if CORRELATION_SEMANTICS == "RAW_R"
    else ("ENTRYWISE_TANH_BEFORE_NETWORK_AGGREGATION" if CORRELATION_SEMANTICS == "FISHER_Z" else "NONE__BRANCH_BLOCKED")
)
gate1_correlation_semantics = correlation_inventory.copy()
gate1_correlation_semantics.to_csv(TABLE_DIR / "gate1_correlation_semantics.csv", index=False)
print(gate1_correlation_semantics)
print("Correlation branch:", CORRELATION_STATUS)


# The released inputs do not provide the correlation branch.
gate1_measurement_ladder = None

# ---------------------------
# Bounded covariance/amplitude normalization diagnostic
# ---------------------------

DIAGNOSTIC_SESSIONS_PER_STATE = 3


def resolve_raw_covariance_archives():
    resolved = {state: [] for state in STATES}
    for path in DATA_DIR.rglob("*FullCovarianceConnectivity.zip"):
        if not path.is_file() or OUTPUT_DIR in path.resolve().parents:
            continue
        state = classify_state_from_name(path.name)
        if state is not None:
            resolved[state].append(path.resolve())
    for state in STATES:
        if len(resolved[state]) != 1:
            raise FileNotFoundError(
                f"Expected one {state} *FullCovarianceConnectivity.zip; "
                f"found {len(resolved[state])}"
            )
    return {state: paths[0] for state, paths in resolved.items()}


def zip_member_for_area(zip_path, area):
    token = compact_token(area)
    with ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    exact = [name for name in members if compact_token(Path(name).stem) == token]
    if len(exact) == 1:
        return exact[0]
    hits = [name for name in members if compact_token(Path(name).stem).endswith(token)]
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f"Could not uniquely map parcel {area} in {zip_path.name}")


def zip_header(zip_path, member):
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        return list(pd.read_csv(stream, nrows=0).columns)


def destination_map(columns):
    area_lookup = {compact_token(area): area for area in parcel_map["Brain Area"].astype(str)}
    return {
        area_lookup[compact_token(column)]: column
        for column in columns if compact_token(column) in area_lookup
    }


def load_parcel_amplitude_matrix(state):
    zip_path = resolve_amplitude_zip(state)
    member = amplitude_member(zip_path)
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        table = pd.read_csv(stream, low_memory=False)
    id_column = identify_id_column(table.columns)
    destinations = destination_map(table.columns)
    areas = parcel_map.sort_values("Real Index")["Brain Area"].astype(str).tolist()
    missing = [area for area in areas if area not in destinations]
    if missing:
        raise KeyError(f"{member} lacks {len(missing)} frozen parcels; first {missing[:5]}")
    output = table[[id_column, *[destinations[area] for area in areas]]].copy()
    output = output.rename(
        columns={id_column: "id_event", **{destinations[area]: area for area in areas}}
    )
    output["id_event"] = output["id_event"].astype(str).str.strip()
    for area in areas:
        output[area] = pd.to_numeric(output[area], errors="coerce")
    if output["id_event"].duplicated().any():
        raise AssertionError(f"Duplicate IDs in {state}/FullAmplitudes")
    return output


def read_selected_rows(zip_path, member, usecols, target_ids):
    wanted = set(map(str, target_ids))
    pieces = []
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        for chunk in pd.read_csv(stream, usecols=usecols, chunksize=512, low_memory=False):
            id_column = identify_id_column(chunk.columns)
            keep = chunk[id_column].astype(str).str.strip().isin(wanted)
            if keep.any():
                pieces.append(chunk.loc[keep].copy())
    if not pieces:
        return pd.DataFrame(columns=usecols)
    return pd.concat(pieces, ignore_index=True)


raw_covariance_archives = resolve_raw_covariance_archives()


def select_diagnostic_sessions(state, amplitudes):
    first_area = parcel_map.sort_values("Real Index")["Brain Area"].iloc[0]
    member = zip_member_for_area(raw_covariance_archives[state], first_area)
    header = zip_header(raw_covariance_archives[state], member)
    id_column = identify_id_column(header)
    with ZipFile(raw_covariance_archives[state]) as archive, archive.open(member) as stream:
        covariance_ids = (
            pd.read_csv(stream, usecols=[id_column], low_memory=False)[id_column]
            .astype(str).str.strip()
        )
    candidates = sorted(set(amplitudes["id_event"].astype(str)) & set(covariance_ids))
    if not candidates:
        raise RuntimeError(f"No common amplitude/covariance sessions for {state}")
    selected = []
    if state == "REST":
        rest_column = "full_covariance__rest__default"
        finite = pd.to_numeric(full_node_wide[rest_column], errors="coerce").notna()
        reference_id = str(full_node_wide.loc[finite, "id_event"].iloc[0]).strip()
        if reference_id in candidates:
            selected.append(reference_id)
    remaining = [value for value in candidates if value not in selected]
    needed = max(0, DIAGNOSTIC_SESSIONS_PER_STATE - len(selected))
    if needed and remaining:
        positions = np.linspace(
            0, len(remaining) - 1, min(needed, len(remaining))
        ).round().astype(int)
        selected.extend(remaining[position] for position in positions)
    return selected


def run_covariance_amplitude_diagnostic():
    parcel_order = (
        parcel_map.sort_values("Real Index")["Brain Area"].astype(str).tolist()
    )
    self_rows, session_rows = [], []
    pair_values, denominator_values, top_records = {}, {}, {}

    for state in STATES:
        amplitudes = load_parcel_amplitude_matrix(state)
        selected_ids = select_diagnostic_sessions(state, amplitudes)
        amplitude_index = amplitudes.set_index("id_event")
        for order, id_event in enumerate(selected_ids, 1):
            key = (state, id_event)
            pair_values[key], denominator_values[key], top_records[key] = [], [], []
            session_rows.append(
                {"state": state, "id_event": id_event, "selection_order": order}
            )

        for source_index, source_area in enumerate(
            tqdm(parcel_order, desc=f"{state} normalization diagnostic", leave=False)
        ):
            later = parcel_order[source_index + 1:]
            member = zip_member_for_area(raw_covariance_archives[state], source_area)
            header = zip_header(raw_covariance_archives[state], member)
            id_column = identify_id_column(header)
            destinations = destination_map(header)
            required_areas = [source_area, *later]
            missing = [area for area in required_areas if area not in destinations]
            if missing:
                raise KeyError(
                    f"{state}/{member} lacks {len(missing)} destinations; first {missing[:5]}"
                )
            usecols = [id_column, *[destinations[area] for area in required_areas]]
            selected = read_selected_rows(
                raw_covariance_archives[state], member, usecols, selected_ids
            )
            selected[id_column] = selected[id_column].astype(str).str.strip()
            selected = selected.set_index(id_column)

            for id_event in selected_ids:
                if id_event not in selected.index:
                    raise RuntimeError(f"{state}/{member} lacks {id_event}")
                row = selected.loc[id_event]
                if isinstance(row, pd.DataFrame):
                    raise RuntimeError(f"Duplicate {id_event} in {state}/{member}")
                amplitude_i = float(amplitude_index.loc[id_event, source_area])
                self_covariance = float(pd.to_numeric(row[destinations[source_area]], errors="coerce"))
                amplitude_i_squared = amplitude_i ** 2 if np.isfinite(amplitude_i) else np.nan
                scale = max(
                    abs(self_covariance), abs(amplitude_i), abs(amplitude_i_squared), 1e-12
                )
                self_rows.append({
                    "state": state,
                    "id_event": id_event,
                    "parcel": source_area,
                    "covariance_self_entry": self_covariance,
                    "full_amplitude": amplitude_i,
                    "full_amplitude_squared": amplitude_i_squared,
                    "relative_error_vs_amplitude": abs(self_covariance - amplitude_i) / scale,
                    "relative_error_vs_amplitude_squared": abs(
                        self_covariance - amplitude_i_squared
                    ) / scale,
                    "self_over_amplitude": (
                        self_covariance / amplitude_i if amplitude_i != 0 else np.nan
                    ),
                    "self_over_amplitude_squared": (
                        self_covariance / amplitude_i_squared
                        if amplitude_i_squared != 0 else np.nan
                    ),
                })
                for destination_area in later:
                    covariance = float(
                        pd.to_numeric(row[destinations[destination_area]], errors="coerce")
                    )
                    amplitude_j = float(amplitude_index.loc[id_event, destination_area])
                    denominator = amplitude_i * amplitude_j
                    ratio = (
                        covariance / denominator
                        if np.isfinite(denominator) and denominator != 0 else np.nan
                    )
                    key = (state, id_event)
                    if np.isfinite(ratio):
                        pair_values[key].append(ratio)
                        denominator_values[key].append(denominator)
                        if abs(ratio) > 1.01:
                            top_records[key].append({
                                "state": state,
                                "id_event": id_event,
                                "source_parcel": source_area,
                                "destination_parcel": destination_area,
                                "covariance": covariance,
                                "source_full_amplitude": amplitude_i,
                                "destination_full_amplitude": amplitude_j,
                                "amplitude_product": denominator,
                                "ratio": ratio,
                                "absolute_ratio": abs(ratio),
                            })
                            top_records[key] = sorted(
                                top_records[key],
                                key=lambda record: record["absolute_ratio"],
                                reverse=True,
                            )[:10]

    self_table = pd.DataFrame(self_rows)
    pair_rows = []
    for (state, id_event), ratio_list in pair_values.items():
        ratios = np.asarray(ratio_list, float)
        denominators = np.asarray(denominator_values[(state, id_event)], float)
        violations = np.abs(ratios) > 1.01
        pair_rows.append({
            "state": state,
            "id_event": id_event,
            "finite_pair_n": len(ratios),
            "violation_n_abs_ratio_gt_1_01": int(violations.sum()),
            "violation_fraction": float(violations.mean()) if len(ratios) else np.nan,
            "ratio_min": float(np.min(ratios)) if len(ratios) else np.nan,
            "ratio_max": float(np.max(ratios)) if len(ratios) else np.nan,
            "absolute_ratio_max": float(np.max(np.abs(ratios))) if len(ratios) else np.nan,
            "denominator_min": float(np.min(denominators)) if len(denominators) else np.nan,
            "denominator_median": float(np.median(denominators)) if len(denominators) else np.nan,
            "violation_denominator_median": (
                float(np.median(denominators[violations])) if violations.any() else np.nan
            ),
        })
    top = pd.DataFrame(
        [record for records in top_records.values() for record in records]
    )
    return self_table, pd.DataFrame(pair_rows), top, pd.DataFrame(session_rows)


(
    covariance_amplitude_self_entries,
    covariance_amplitude_pair_summary,
    covariance_amplitude_top_violations,
    diagnostic_sessions,
) = run_covariance_amplitude_diagnostic()

covariance_amplitude_self_summary = (
    covariance_amplitude_self_entries.groupby(["state", "id_event"], as_index=False)
    .agg(
        parcel_n=("parcel", "size"),
        median_relative_error_vs_amplitude=("relative_error_vs_amplitude", "median"),
        median_relative_error_vs_amplitude_squared=(
            "relative_error_vs_amplitude_squared", "median"
        ),
        median_self_over_amplitude=("self_over_amplitude", "median"),
        median_self_over_amplitude_squared=("self_over_amplitude_squared", "median"),
    )
)
covariance_amplitude_self_summary["closer_literal_identity"] = np.where(
    covariance_amplitude_self_summary["median_relative_error_vs_amplitude_squared"]
    < covariance_amplitude_self_summary["median_relative_error_vs_amplitude"],
    "SELF_APPROX_AMPLITUDE_SQUARED",
    "SELF_APPROX_AMPLITUDE",
)
violation_widespread = bool(
    covariance_amplitude_pair_summary["violation_fraction"].median() > .01
)
small_denominator_localization = bool(
    covariance_amplitude_pair_summary["violation_denominator_median"].notna().any()
    and (
        covariance_amplitude_pair_summary["violation_denominator_median"]
        < .10 * covariance_amplitude_pair_summary["denominator_median"]
    ).mean() >= .75
)
normalization_assumption_decision = pd.DataFrame([{
    "attempted_formula": "FullCovariance_ij/(FullAmplitude_i*FullAmplitude_j)",
    "diagnostic_sessions": len(diagnostic_sessions),
    "all_four_states_checked": set(diagnostic_sessions["state"]) == set(STATES),
    "violations_widespread_gt_1pct_median": violation_widespread,
    "violations_localized_to_very_small_denominators": small_denominator_localization,
    "pearson_interpretation": "UNSUPPORTED_BY_RELEASE_DOCUMENTATION_AND_NOT_USED",
    "analysis_action": "CONTINUE_NATIVE_COVARIANCE; DROP_MANUFACTURED_R_BRANCH",
    "blocks_part_b": False,
}])

normalization_parts = []
for name, table in [
    ("normalization_assumption_decision.csv", normalization_assumption_decision),
    ("covariance_amplitude_self_summary.csv", covariance_amplitude_self_summary),
    ("covariance_amplitude_pair_summary.csv", covariance_amplitude_pair_summary),
    ("covariance_amplitude_top_violations.csv", covariance_amplitude_top_violations),
]:
    part = table.copy()
    part.insert(0, "source_table", name)
    normalization_parts.append(part)
gate1_normalization_diagnostics = pd.concat(
    normalization_parts, ignore_index=True, sort=False
)
gate1_normalization_diagnostics.to_csv(
    TABLE_DIR / "gate1_normalization_diagnostics.csv", index=False
)
# ---------------------------
# Part H: within-session amplitude-reliability resource inventory
# ---------------------------

def classify_replicate_hint(text):
    t = str(text)
    low = t.lower()
    hints = []
    if re.search(r"split.?half|splithalf", low): hints.append("SPLIT_HALF")
    if re.search(r"run.?1|run.?2|run.?01|run.?02", low): hints.append("RUN_SPECIFIC")
    if re.search(r"(^|[_-])pa([_.-]|$)|(^|[_-])ap([_.-]|$)", low): hints.append("PHASE_ENCODING_LABEL")
    if re.search(r"repeat|retest", low): hints.append("REPEAT_RETEST")
    return ";".join(hints)

candidate_paths = sorted({
    p.resolve() for p in DATA_DIR.rglob("*")
    if p.is_file()
    and "amplitude" in p.name.lower()
    and OUTPUT_DIR not in p.parents
    and classify_replicate_hint(p.name)
})
inventory_rows = []
for path in candidate_paths:
    state = classify_state_from_name(path.name)
    inventory_rows.append({
        "resource": str(path),
        "state": state or "",
        "candidate_hint": classify_replicate_hint(path.name),
        "resource_type": path.suffix.lower().lstrip("."),
        "explicit_tidy_replicate_table": False,
        "eligible_for_automatic_reliability": False,
        "note": "filename/resource inventory only; cross-state or years-apart values are not substitutes",
    })

# Conservative automatic route: only a tidy table with explicit id_event/state/replicate/value-like columns qualifies.
def try_explicit_replicate_table(path):
    if path.suffix.lower() not in {".csv", ".parquet"}:
        return None
    try:
        table = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    idcol = pick_column(table, ["id_event", "subject_event", "participant_event"])
    statecol = pick_column(table, ["state", "task"])
    repcol = pick_column(table, ["replicate", "run", "phase_encoding", "split_half", "half"])
    valcol = pick_column(table, ["amplitude", "value", "full_amplitude", "signal_amplitude"])
    if not all([idcol, statecol, repcol, valcol]):
        return None
    d = table[[idcol, statecol, repcol, valcol]].rename(
        columns={idcol: "id_event", statecol: "state", repcol: "replicate", valcol: "value"}
    )
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["value"])
    if d["replicate"].nunique() < 2:
        return None
    return d

explicit_tables = []
for path in candidate_paths:
    d = try_explicit_replicate_table(path)
    if d is not None:
        explicit_tables.append((path, d))

reliability_estimates = []
if explicit_tables:
    # Only exact paired replicate labels within the same state/session are used.
    for path, d in explicit_tables:
        for state, ds in d.groupby("state"):
            reps = sorted(ds["replicate"].astype(str).unique())
            if len(reps) < 2:
                continue
            a, b = reps[:2]
            wide = (
                ds.loc[ds["replicate"].astype(str).isin([a, b])]
                .assign(replicate=lambda x: x["replicate"].astype(str))
                .pivot_table(index="id_event", columns="replicate", values="value", aggfunc="mean")
            )
            if a not in wide or b not in wide:
                continue
            pair = wide[[a, b]].dropna()
            if len(pair) < 20:
                continue
            # ICC(A,1), two-way absolute-agreement single-measure, k=2.
            Y = pair.to_numpy(float)
            n, k = Y.shape
            grand = Y.mean()
            row_mean = Y.mean(axis=1)
            col_mean = Y.mean(axis=0)
            ss_row = k * np.sum((row_mean - grand) ** 2)
            ss_col = n * np.sum((col_mean - grand) ** 2)
            ss_err = np.sum((Y - row_mean[:, None] - col_mean[None, :] + grand) ** 2)
            ms_row = ss_row / (n - 1)
            ms_col = ss_col / (k - 1)
            ms_err = ss_err / ((n - 1) * (k - 1))
            denom = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
            icc = (ms_row - ms_err) / denom if denom != 0 else np.nan
            reliability_estimates.append({
                "resource": str(path), "state": str(state), "replicate_a": a, "replicate_b": b,
                "paired_sessions": n, "reliability": float(icc), "method": "ICC(A,1)",
            })

if reliability_estimates:
    WITHIN_SESSION_AMPLITUDE_RELIABILITY = "AVAILABLE"
    for row in inventory_rows:
        if any(row["resource"] == est["resource"] for est in reliability_estimates):
            row["explicit_tidy_replicate_table"] = True
            row["eligible_for_automatic_reliability"] = True
            row["note"] = "valid explicit same-session replicate table used"
else:
    WITHIN_SESSION_AMPLITUDE_RELIABILITY = "UNAVAILABLE"

inventory_rows.append({
    "resource": "SUMMARY",
    "state": "",
    "candidate_hint": "",
    "resource_type": "",
    "explicit_tidy_replicate_table": bool(reliability_estimates),
    "eligible_for_automatic_reliability": bool(reliability_estimates),
    "note": f"WITHIN_SESSION_AMPLITUDE_RELIABILITY={WITHIN_SESSION_AMPLITUDE_RELIABILITY}",
})
gate1_amplitude_reliability_inventory = pd.DataFrame(inventory_rows)
if reliability_estimates:
    rel_df = pd.DataFrame(reliability_estimates)
    # Preserve estimates within the requested inventory output rather than creating an extra inferential file.
    for col in rel_df.columns:
        if col not in gate1_amplitude_reliability_inventory.columns:
            gate1_amplitude_reliability_inventory[col] = np.nan
    extra = rel_df.copy()
    for col in gate1_amplitude_reliability_inventory.columns:
        if col not in extra:
            extra[col] = np.nan
    gate1_amplitude_reliability_inventory = pd.concat(
        [gate1_amplitude_reliability_inventory, extra[gate1_amplitude_reliability_inventory.columns]],
        ignore_index=True,
    )

gate1_amplitude_reliability_inventory.to_csv(
    TABLE_DIR / "gate1_amplitude_reliability_inventory.csv", index=False
)
print(gate1_amplitude_reliability_inventory)

# Measurement figure and summary

if QMAX_U95 < 0.50:
    amplitude_decision = "STRONG_AMPLITUDE_SENSITIVITY_SUPPORTED"
elif QMAX_U95 < 1.0:
    amplitude_decision = "ATTENUATION_SUPPORTED_BUT_LESS_THAN_HALF_RULE_NOT_MET"
else:
    amplitude_decision = "AMPLITUDE_DESCRIPTIVE_CONTEXT_ONLY"

# Measurement summary figure.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(STATES))
offsets = [-0.18, 0.0, 0.18]
for offset, scenario in zip(offsets, ORDINARY_OUTCOMES):
    d = gate1_ordinary_common_models.loc[gate1_ordinary_common_models["scenario"].eq(scenario)].set_index("state").loc[STATES]
    y = d["slope_per_5y"].to_numpy(float)
    lo = d["simultaneous_ci_low"].to_numpy(float)
    hi = d["simultaneous_ci_high"].to_numpy(float)
    axes[0].errorbar(x + offset, y, yerr=np.vstack([y - lo, hi - y]), marker="o", capsize=3, label=scenario)
axes[0].axhline(0, linewidth=1)
axes[0].set_xticks(x, STATES)
axes[0].set_ylabel("Common-level slope per 5 years")
axes[0].set_title("Ordinary outcome-specific working-covariance fits")
axes[0].legend(fontsize=8)

q_state = gate1_residual_fraction_intervals.loc[
    gate1_residual_fraction_intervals["state"].isin(STATES)
].set_index("state").loc[STATES]
q = q_state["q_point"].to_numpy(float)
qlo = q_state["simultaneous_95_ci_low"].to_numpy(float)
qhi = q_state["simultaneous_95_ci_high"].to_numpy(float)
axes[1].errorbar(x, q, yerr=np.vstack([q - qlo, qhi - q]), marker="o", capsize=3)
axes[1].axhline(0.50, linestyle="--", linewidth=1)
axes[1].set_xticks(x, STATES)
axes[1].set_ylabel(r"$q_s=|\beta_{std}/\beta_{raw}|$")
axes[1].set_title(f"Fixed-weight residual fraction; Qmax U95={QMAX_U95:.3f}")
fig.tight_layout()
fig.savefig(FIG_DIR / "gate1_amplitude_measurement.png", dpi=220, bbox_inches="tight")
fig.savefig(FIG_DIR / "gate1_amplitude_measurement.pdf", bbox_inches="tight")
plt.close("all")

gate1_decision = pd.DataFrame([{
    "gate": "GATE_1_AMPLITUDE_MEASUREMENT",
    "observation_identity_max_error": OBSERVATION_IDENTITY_MAX_ERROR,
    "ordinary_coefficient_nonadditivity_max": ORDINARY_COEF_NONADDITIVITY_MAX,
    "ordinary_state_nonadditivity_max": ORDINARY_STATE_NONADDITIVITY_MAX,
    "fixed_weight_coefficient_identity_max_error": FIXED_COEF_IDENTITY_MAX_ERROR,
    "fixed_weight_state_identity_max_error": FIXED_STATE_IDENTITY_MAX_ERROR,
    "Qmax_point": QMAX_POINT,
    "Qmax_upper_95": QMAX_U95,
    "threshold": 0.50,
    "amplitude_decision": amplitude_decision,
    "G_A_role": "DECOMPOSITION_COMPONENT_NOT_INDEPENDENT_VALIDATION",
    "correlation_status": CORRELATION_STATUS,
    "correlation_semantics": CORRELATION_SEMANTICS,
    "measurement_ladder_written": bool(gate1_measurement_ladder is not None),
    "within_session_amplitude_reliability": WITHIN_SESSION_AMPLITUDE_RELIABILITY,
    "bootstrap_B": BOOTSTRAP_B,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "draw_trimming": "NONE",
    "legacy_common_slope_reproduction_max_error": LEGACY_SLOPE_REPRO_MAX_ERROR,
    "amplitude_matched_row_sha256": AMPLITUDE_MATCHED_ROW_SHA256,
    "embedded_rules_sha256": EMBEDDED_RULES_SHA256,
    "external_rules_status": EXTERNAL_RULES_STATUS,
}])
gate1_decision.to_csv(TABLE_DIR / "gate1_decision.csv", index=False)
print(gate1_decision.T)

required_outputs = [
    TABLE_DIR / "gate1_ordinary_common_models.csv",
    TABLE_DIR / "gate1_common_weight_decomposition.csv",
    TABLE_DIR / "gate1_residual_fraction_intervals.csv",
    TABLE_DIR / "gate1_correlation_semantics.csv",
    TABLE_DIR / "gate1_normalization_diagnostics.csv",
    TABLE_DIR / "gate1_amplitude_reliability_inventory.csv",
    TABLE_DIR / "gate1_decision.csv",
    FIG_DIR / "gate1_amplitude_measurement.png",
]
if gate1_measurement_ladder is not None:
    required_outputs.append(TABLE_DIR / "gate1_measurement_ladder.csv")
missing_outputs = [str(p) for p in required_outputs if not p.exists()]
assert not missing_outputs, missing_outputs


run_manifest = {
    "script": "02_primary_profile_and_measurement.py",
    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    "data_dir": str(DATA_DIR),
    "subject_table": str(SUBJECT_TABLE),
    "output_dir": str(OUTPUT_DIR),
    "frozen_parent_repeat_participants": int(profile_data["id"].nunique()),
    "eligible_networks": ELIGIBLE_NETWORKS,
    "states": STATES,
    "amplitude_matched_rows": len(model_frame),
    "amplitude_matched_participants": int(model_frame["id"].nunique()),
    "amplitude_matched_row_sha256": AMPLITUDE_MATCHED_ROW_SHA256,
    "observation_identity_max_error": OBSERVATION_IDENTITY_MAX_ERROR,
    "fixed_weight_coefficient_identity_max_error": FIXED_COEF_IDENTITY_MAX_ERROR,
    "Qmax_upper_95": QMAX_U95,
    "amplitude_decision": amplitude_decision,
    "correlation_status": CORRELATION_STATUS,
    "within_session_amplitude_reliability": WITHIN_SESSION_AMPLITUDE_RELIABILITY,
    "bootstrap_B": BOOTSTRAP_B,
    "bootstrap_seed": BOOTSTRAP_SEED,
}
(OUTPUT_DIR / "gate1_run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2) + "\n"
)

assert provenance_audit["pass"].all()
assert OBSERVATION_IDENTITY_MAX_ERROR < IDENTITY_TOL
assert FIXED_COEF_IDENTITY_MAX_ERROR < IDENTITY_TOL
assert FIXED_STATE_IDENTITY_MAX_ERROR < IDENTITY_TOL
assert np.isfinite(QMAX_POINT)
assert QMAX_U95 >= QMAX_POINT or np.isclose(QMAX_U95, QMAX_POINT)

print("\nPRIMARY PROFILE AND MEASUREMENT EXTRACTION COMPLETE")
print(f"Frozen repeat participants: {profile_data['id'].nunique():,}")
print(f"Eligible networks: {len(ELIGIBLE_NETWORKS)}")
print(f"Session-state rows: {len(profile_data):,}")
print(
    gate1_decision[[
        "amplitude_decision", "Qmax_point", "Qmax_upper_95",
        "correlation_status", "within_session_amplitude_reliability",
    ]].to_string(index=False)
)
print(f"Outputs: {OUTPUT_DIR}")
