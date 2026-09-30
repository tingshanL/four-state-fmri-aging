#!/usr/bin/env python3
"""Analyze frontoparietal-relative change, specificity, and robustness."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Local multimodal project root containing the fMRI caches and archives.",
    )
    parser.add_argument(
        "--subject-table", type=Path, required=True,
        help="Longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Local results root used by scripts 02 and 03.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Development run with 999 multiplier draws.",
    )
    parser.add_argument(
        "--force-refit", action="store_true",
        help="Ignore compatible scalar-model checkpoints and refit them.",
    )
    parser.add_argument(
        "--bootstrap-b", type=int, default=None,
        help="Override familywise multiplier draws (paper run default: 4999).",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_DIR = ARGS.data_dir.expanduser().resolve()
SUBJECT_TABLE = ARGS.subject_table.expanduser().resolve()
RESULTS_ROOT = ARGS.results_dir.expanduser().resolve()
OUTPUT_DIR = RESULTS_ROOT / "03_fp_specificity_and_robustness"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for directory in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, CHECKPOINT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PROFILE_OUTPUT_DIR = RESULTS_ROOT / "02_primary_profile_and_measurement"
SOURCE_PROFILE_TABLE_DIR = SOURCE_PROFILE_OUTPUT_DIR / "tables"
SOURCE_PROFILE_CHECKPOINT_DIR = SOURCE_PROFILE_OUTPUT_DIR / "checkpoints"

if not DATA_DIR.is_dir():
    raise NotADirectoryError(DATA_DIR)
if not SUBJECT_TABLE.is_file():
    raise FileNotFoundError(SUBJECT_TABLE)
if not SOURCE_PROFILE_OUTPUT_DIR.is_dir():
    raise FileNotFoundError(
        f"Script 02 results are missing: {SOURCE_PROFILE_OUTPUT_DIR}"
    )

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

BOOT_B = int(ARGS.bootstrap_b if ARGS.bootstrap_b is not None else (999 if FAST_MODE else 4999))
BOOT_SEED = 20260907
ALPHA = .05
IDENTITY_TOL = 1e-10
SOURCE_ESTIMATE_TOL = 1e-7
SOURCE_COV_TOL = 1e-6

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
FP_NETWORK = "Frontoparietal"
FOCAL_NETWORKS = ["Frontoparietal", "Default", "Cingulo-Opercular"]
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


# Data construction

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
            raise KeyError(f"{path} lacks prespecified metrics: {missing}")
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
    # First reuse the exact table recorded by the completed ASL longitudinal
    # analysis. This avoids accidentally selecting a non-imaging
    # table whose identifiers cannot form id_event.
    manifest_candidates = sorted(DATA_DIR.rglob(
        "AABC2_longitudinal_regional_ASL_topography_final_repair_outputs/source_manifest.csv"
    ))
    search_roots = [DATA_DIR]
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
        raise FileNotFoundError("The prespecified fmri_node_parcel_manifest.csv is required")
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

# Eligibility uses only the prespecified parent opportunities and is completed before slopes.
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
    raise AssertionError(f"Only {len(ELIGIBLE_NETWORKS)} networks met the prespecified eligibility criteria")
if not FAST_MODE and len(ELIGIBLE_NETWORKS) != EXPECTED_ELIGIBLE_NETWORK_N:
    raise AssertionError(
        f"Analysis requires the existing ten-network set; recovered {len(ELIGIBLE_NETWORKS)}: {ELIGIBLE_NETWORKS}"
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
        f"Analysis requires 852 repeat participants; recovered {profile_data['id'].nunique()}"
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
    print(profile_identity_audit)
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

print(parcel_counts)
print(network_state_eligibility)
print(network_eligibility)
print(profile_identity_audit)
print("Eligible networks selected before slope estimation:", ELIGIBLE_NETWORKS)
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
        raise KeyError(f"{member} lacks {len(missing)} prespecified parcels; first {missing[:5]}")
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



# Repeated-measures model functions

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
        "pedigree_influence": influence,
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



# Direct FP-relative models and robustness analyses

if FP_NETWORK not in ELIGIBLE_NETWORKS:
    raise AssertionError("Frontoparietal network is absent from the eligible set")
if not FAST_MODE and K != 10:
    raise AssertionError(f"Expected ten eligible networks, recovered {K}")

SOURCE_SCENARIOS = {
    "RAW_FULL": {
        "frame": "FULL_PRIMARY", "joint_scenario": "FULL_PRIMARY",
        "role": "PRINCIPAL_NATIVE_FULL_COVARIANCE", "suffix": "",
        "source_role": "PRIMARY_FULL_COVARIANCE",
    },
    "AMPLITUDE_STANDARDIZED": {
        "frame": "AMPLITUDE_STANDARDIZED", "joint_scenario": "AMPLITUDE_STANDARDIZED",
        "role": "PRINCIPAL_FIXED_AMPLITUDE_STANDARDIZED_COVARIANCE", "suffix": "_STD",
        "source_role": "FIXED_AMPLITUDE_STANDARDIZATION",
    },
    "PARTIAL_MATCHED": {
        "frame": "PARTIAL_MATCHED", "joint_scenario": "PARTIAL_MATCHED",
        "role": "SECONDARY_POSITIVE_PARTIAL_COVARIANCE", "suffix": "",
        "source_role": "PARTIAL_COVARIANCE_SENSITIVITY",
    },
}


def add_fp_relative(frame, suffix=""):
    out = frame.copy()
    fp = pd.to_numeric(out[f"L{suffix}__{FP_NETWORK}"], errors="coerce").astype(np.float64)
    references = [f"L{suffix}__{network}" for network in ELIGIBLE_NETWORKS if network != FP_NETWORK]
    out["Y_FP_REL"] = fp - out[references].to_numpy(dtype=np.float64).mean(axis=1)
    out["Y_FP_REL_FROM_PROFILE"] = (K / (K - 1.0)) * pd.to_numeric(
        out[f"R{suffix}__{FP_NETWORK}"], errors="coerce"
    ).astype(np.float64)
    return out


SCALAR_FRAMES = {
    name: add_fp_relative(FRAME_REGISTRY[spec["frame"]], spec["suffix"])
    for name, spec in SOURCE_SCENARIOS.items()
}
if not FAST_MODE:
    principal_participant_counts = {
        name: int(SCALAR_FRAMES[name]["id"].nunique())
        for name in ["RAW_FULL", "AMPLITUDE_STANDARDIZED"]
    }
    if set(principal_participant_counts.values()) != {EXPECTED_CONFIRMATION_PARTICIPANTS}:
        raise AssertionError(
            "Both principal representations must retain the expected 852 participants; "
            f"found {principal_participant_counts}"
        )

identity_rows = []
for name, frame in SCALAR_FRAMES.items():
    error = (frame["Y_FP_REL"] - frame["Y_FP_REL_FROM_PROFILE"]).to_numpy(dtype=np.float64)
    identity_rows.append({
        "representation": name,
        "identity": "Y_FP_REL = L_FP - mean(other nine L) = (10/9) R_FP",
        "eligible_network_n": K,
        "participants": int(frame["id"].nunique()),
        "rows": int(len(frame)),
        "max_absolute_error": float(np.nanmax(np.abs(error))),
        "tolerance": 1e-10,
        "pass": bool(np.nanmax(np.abs(error)) <= 1e-10),
    })
fp_relative_identity_audit = pd.DataFrame(identity_rows)
if not fp_relative_identity_audit["pass"].all():
    print(fp_relative_identity_audit)
    raise AssertionError("FP-relative scalar/profile identity failed")


def source_spec(name, spec):
    return {
        "scenario": spec["joint_scenario"], "frame": spec["frame"],
        "role": spec["source_role"], "site_left_out": None,
    }


def profile_fingerprint(frame, spec):
    columns = [
        "id", "id_event", "state", "pedigree_group", "age0_c5", "t5",
        "site", "site_scanner", "sex", "study",
    ] + [f"profile__{network}" for network in ELIGIBLE_NETWORKS]
    payload = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=True).to_numpy().tobytes()
    marker = {**spec, "model": "profile", "networks": ELIGIBLE_NETWORKS}
    return hashlib.sha256(payload + json.dumps(marker, sort_keys=True).encode()).hexdigest()


source_manifest_path = SOURCE_PROFILE_OUTPUT_DIR / "run_manifest.json"
if not source_manifest_path.is_file():
    raise FileNotFoundError(
        f"Saved joint-profile run manifest not found: {source_manifest_path}. "
        "Run 02_primary_profile_and_measurement.py first."
    )
source_run_manifest = json.loads(source_manifest_path.read_text())
if source_run_manifest.get("eligible_networks") != ELIGIBLE_NETWORKS:
    raise AssertionError(
        "Eligible-network order differs from the saved joint fit: "
        f"saved={source_run_manifest.get('eligible_networks')}, rebuilt={ELIGIBLE_NETWORKS}"
    )
if not FAST_MODE:
    if int(source_run_manifest.get("frozen_repeat_participants", -1)) != 852:
        raise AssertionError("Script 02 does not report the expected 852-person cohort")
    if int(source_run_manifest.get("profile_participants", -1)) != 852:
        raise AssertionError("Saved joint profile does not contain exactly 852 participants")

JOINT_SAVED = {}
source_fit_audit_rows = []
for name, spec in SOURCE_SCENARIOS.items():
    checkpoint = SOURCE_PROFILE_CHECKPOINT_DIR / f"profile__{spec['joint_scenario'].lower()}.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required saved joint-profile checkpoint is missing: {checkpoint}")
    saved = json.loads(checkpoint.read_text())
    current_frame = FRAME_REGISTRY[spec["frame"]]
    current_fingerprint = profile_fingerprint(current_frame, source_spec(name, spec))
    payload = saved.get("result", {}).get("payload")
    if payload is None:
        raise RuntimeError(f"Saved joint-profile checkpoint lacks its coefficient payload: {checkpoint}")
    design = DirectStateDesign().fit(current_frame)
    names_match = payload.get("names") == design.names
    fingerprint_match = saved.get("fingerprint") == current_fingerprint
    source_fit_audit_rows.append({
        "representation": name, "checkpoint": str(checkpoint),
        "fingerprint_match": fingerprint_match, "design_names_match": names_match,
        "saved_network_n": len(source_run_manifest.get("eligible_networks", [])),
        "rebuilt_participants": int(current_frame["id"].nunique()),
        "rebuilt_rows": int(len(current_frame)),
    })
    if not fingerprint_match or not names_match:
        raise AssertionError(
            f"{name} does not match its saved joint fit (fingerprint={fingerprint_match}, names={names_match})"
        )
    JOINT_SAVED[name] = {"saved": saved, "design": design, "frame": current_frame}

source_fit_audit = pd.DataFrame(source_fit_audit_rows)
print(fp_relative_identity_audit)
print(source_fit_audit)

def scalar_fingerprint(frame, spec):
    columns = [
        "id", "id_event", "state", "pedigree_group", "age0_c5", "t5",
        "site", "site_scanner", "sex", "study", "Y_FP_REL",
    ]
    payload = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=True).to_numpy().tobytes()
    return hashlib.sha256(payload + json.dumps(spec, sort_keys=True).encode()).hexdigest()


def site_state_slope_vector(design, site, state):
    vector = design.state_slope_vector(state)
    for coded_site in design.site_levels[:-1]:
        site_weight = design._coded_weight(site, design.site_levels, coded_site)
        design._put(vector, f"t5:site_sum[{coded_site}]", site_weight)
        for coded_state in STATES[:-1]:
            state_weight = design._coded_weight(state, STATES, coded_state)
            design._put(
                vector,
                f"t5:site_sum[{coded_site}]:state_sum[{coded_state}]",
                site_weight * state_weight,
            )
    return vector


def fit_fp_scalar(frame, spec):
    required = ["Y_FP_REL", "age0_c5", "t5"]
    frame = frame.loc[np.isfinite(frame[required].to_numpy(float)).all(axis=1)].copy()
    state_n = frame.groupby("state")["id"].nunique().reindex(STATES, fill_value=0)
    if state_n.min() < MIN_STATE_PARTICIPANTS:
        return {
            "summary": {**spec, "fit_status": "INSUFFICIENT_STATE_COVERAGE", "status_detail": json.dumps(state_n.to_dict())},
            "tests": [], "state": [], "site_state": [], "payload": None,
        }
    fingerprint = scalar_fingerprint(frame, spec)
    checkpoint = CHECKPOINT_DIR / f"scalar__{spec['scenario'].lower()}.json"
    if checkpoint.exists() and not FORCE_REFIT:
        saved = json.loads(checkpoint.read_text())
        if saved.get("fingerprint") == fingerprint:
            return saved["result"]
    try:
        design = DirectStateDesign().fit(frame)
        x = design.transform(frame)
        fit = fit_direct_profile_covariance(
            x, frame["Y_FP_REL"], frame["id"], frame["state"], frame["pedigree_group"]
        )
        if fit is None:
            raise RuntimeError("Direct scalar covariance fit returned None")
        beta, cov = fit["coef"], fit["robust_cov"]
        c_state = design.state_slope_matrix()
        test_rows = []
        for test, matrix in [
            ("all_four_state_slopes_zero", c_state),
            ("state_heterogeneity", c_state[:3] - c_state[3]),
            ("site_x_time", design.site_time_matrix()),
            ("site_x_time_x_state", design.site_time_state_matrix()),
        ]:
            statistic, df, p = wald_matrix(beta, cov, matrix)
            test_rows.append({
                "scenario": spec["scenario"], "representation": spec["representation"],
                "test": test, "chi2": statistic, "df": df, "p": p,
            })
        estimates = c_state @ beta
        state_cov = c_state @ cov @ c_state.T
        se = np.sqrt(np.maximum(0, np.diag(state_cov)))
        p_values = 2 * norm.sf(np.abs(estimates / se))
        p_holm = holm_adjust(p_values)
        q4 = norm.ppf(1 - .05 / 8)
        state_rows = [{
            "scenario": spec["scenario"], "representation": spec["representation"], "state": state,
            "site_left_out": spec.get("site_left_out"),
            "slope_per_5y": float(estimates[j]), "se": float(se[j]),
            "z": float(estimates[j] / se[j]) if se[j] > 0 else np.nan,
            "p_unadjusted": float(p_values[j]), "p_holm_four_states": float(p_holm[j]),
            "simultaneous_ci_low": float(estimates[j] - q4 * se[j]),
            "simultaneous_ci_high": float(estimates[j] + q4 * se[j]),
        } for j, state in enumerate(STATES)]

        site_state_rows = []
        if spec.get("export_site_state", False):
            q_site = norm.ppf(1 - .05 / (2 * len(design.site_levels) * len(STATES)))
            for site in design.site_levels:
                for state in STATES:
                    vector = site_state_slope_vector(design, site, state)
                    estimate, se_one, z, p = scalar_contrast(beta, cov, vector)
                    site_state_rows.append({
                        "scenario": spec["scenario"], "representation": spec["representation"],
                        "site": site, "state": state, "slope_per_5y": estimate, "se": se_one,
                        "z": z, "p_unadjusted": p,
                        "simultaneous_ci_low": estimate - q_site * se_one,
                        "simultaneous_ci_high": estimate + q_site * se_one,
                        "interval_family": f"Bonferroni across {len(design.site_levels)} sites x 4 states",
                    })

        summary = {
            **spec, "fit_status": "PASS", "status_detail": "",
            "rows": fit["row_n"], "participants": fit["participant_n"], "pedigrees": fit["pedigree_n"],
            "design_columns": len(design.names), "design_rank": fit["rank"],
            "dropped_design_columns": json.dumps(design.dropped_names),
            "condition_number": float(np.linalg.cond(x)),
            "optimizer_success": fit["optimizer_success"], "optimizer_nfev": fit["optimizer_nfev"],
            "objective": fit["objective"], "sigma_e2": fit["sigma_e2"],
            "tau_participant": fit["tau_participant"],
            "tau_participant_state": fit["tau_participant_state"],
            "robust_cov_rank": int(np.linalg.matrix_rank(cov)),
            "robust_cov_min_eigenvalue": float(np.linalg.eigvalsh((cov + cov.T) / 2).min()),
        }
        result = {
            "summary": summary, "tests": test_rows, "state": state_rows, "site_state": site_state_rows,
            "payload": {"coef": beta, "cov": cov, "names": design.names},
        }
        checkpoint.write_text(json.dumps({"fingerprint": fingerprint, "result": result}, default=jsonable, indent=2) + "\n")
        return result
    except Exception as exc:
        return {
            "summary": {**spec, "fit_status": "FIT_ERROR", "status_detail": f"{type(exc).__name__}: {exc}"},
            "tests": [], "state": [], "site_state": [], "payload": None,
        }


DIRECT_SPECS = []
for name, source in SOURCE_SCENARIOS.items():
    DIRECT_SPECS.append({
        "scenario": name, "representation": name, "role": source["role"],
        "site_left_out": None, "export_site_state": name in {"RAW_FULL", "AMPLITUDE_STANDARDIZED"},
    })

principal_sites = sorted(SCALAR_FRAMES["RAW_FULL"]["site"].astype(str).unique())
if not FAST_MODE and len(principal_sites) != 4:
    raise AssertionError(f"Expected four AABC2 sites for the leave-one-site-out analysis; found {principal_sites}")

for name in ["RAW_FULL", "AMPLITUDE_STANDARDIZED"]:
    base_frame = SCALAR_FRAMES[name]
    for site in sorted(base_frame["site"].astype(str).unique()):
        site_key = re.sub(r"[^A-Za-z0-9]+", "_", site).strip("_").upper()
        scenario = f"{name}__LOO_{site_key}"
        SCALAR_FRAMES[scenario] = retain_two_sessions(base_frame.loc[base_frame["site"].astype(str).ne(site)])
        DIRECT_SPECS.append({
            "scenario": scenario, "representation": name, "role": "LEAVE_ONE_SITE_OUT",
            "site_left_out": site, "export_site_state": False,
        })

direct_results = []
for spec in tqdm(DIRECT_SPECS, desc="Direct FP-relative scalar fits"):
    direct_results.append(fit_fp_scalar(SCALAR_FRAMES[spec["scenario"]], spec))

direct_model_summary = pd.DataFrame([result["summary"] for result in direct_results])
direct_tests = pd.DataFrame([row for result in direct_results for row in result["tests"]])
direct_state_slopes = pd.DataFrame([row for result in direct_results for row in result["state"]])
site_state_slopes = pd.DataFrame([row for result in direct_results for row in result["site_state"]])

critical = direct_model_summary.loc[direct_model_summary["scenario"].isin(SOURCE_SCENARIOS)]
if len(critical) != 3 or not critical["fit_status"].eq("PASS").all():
    print(critical)
    raise AssertionError("At least one direct scalar analysis fit failed")
critical_tests = direct_tests.loc[direct_tests["scenario"].isin(SOURCE_SCENARIOS)]
critical_df = critical_tests.loc[
    critical_tests["test"].isin(["all_four_state_slopes_zero", "state_heterogeneity"]),
    ["scenario", "test", "df"],
]
expected_df = critical_df["test"].map({"all_four_state_slopes_zero": 4, "state_heterogeneity": 3})
if not critical_df["df"].eq(expected_df).all():
    print(critical_df)
    raise AssertionError("The required four-df omnibus or three-df heterogeneity test lost rank")
print(direct_model_summary)
print(direct_tests)
print(direct_state_slopes)

def weight_fp_reference(excluded_network=None, pair_network=None):
    weight = np.zeros(K)
    index = {network: j for j, network in enumerate(ELIGIBLE_NETWORKS)}
    weight[index[FP_NETWORK]] = 1.0
    if pair_network is not None:
        weight[index[pair_network]] = -1.0
        return weight
    reference = [network for network in ELIGIBLE_NETWORKS if network != FP_NETWORK and network != excluded_network]
    for network in reference:
        weight[index[network]] = -1.0 / len(reference)
    if abs(weight.sum()) > 1e-12:
        raise AssertionError("FP-reference weight does not sum to zero")
    return weight


def reconstruct_from_joint(name, weight, label, family):
    entry = JOINT_SAVED[name]
    payload = entry["saved"]["result"]["payload"]
    beta = np.asarray(payload["coef"], float)
    cov = np.asarray(payload["cov"], float)
    basis = np.asarray(payload["basis"], float)
    design = entry["design"]
    c_state = design.state_slope_matrix()
    vectors = np.vstack([np.kron(weight @ basis, row) for row in c_state])
    estimates = vectors @ beta
    covariance = vectors @ cov @ vectors.T
    se = np.sqrt(np.maximum(0, np.diag(covariance)))
    statistic, df, p_joint = wald_matrix(beta, cov, vectors)
    rows = [{
        "representation": name, "family": family, "reference_choice": label,
        "state": state, "slope_per_5y": float(estimates[j]), "se": float(se[j]),
        "z": float(estimates[j] / se[j]) if se[j] > 0 else np.nan,
        "joint_chi2": statistic, "joint_df": df, "joint_p": p_joint,
    } for j, state in enumerate(STATES)]
    return rows


joint_primary_rows, pairwise_rows, reference_rows = [], [], []
for name in SOURCE_SCENARIOS:
    joint_primary_rows.extend(reconstruct_from_joint(
        name, weight_fp_reference(), "ALL_NINE", "PRIMARY_FP_VS_OTHER_NINE_MEAN"
    ))
    for other in [network for network in ELIGIBLE_NETWORKS if network != FP_NETWORK]:
        pairwise_rows.extend(reconstruct_from_joint(
            name, weight_fp_reference(pair_network=other), f"FP_MINUS_{other}", "FP_VS_EACH_OTHER_NETWORK"
        ))
    reference_rows.extend(reconstruct_from_joint(
        name, weight_fp_reference(), "ALL_NINE", "REFERENCE_DELETION_SENSITIVITY"
    ))
    for excluded in [network for network in ELIGIBLE_NETWORKS if network != FP_NETWORK]:
        reference_rows.extend(reconstruct_from_joint(
            name, weight_fp_reference(excluded_network=excluded),
            f"LEAVE_OUT_{excluded}", "REFERENCE_DELETION_SENSITIVITY",
        ))

joint_primary_slopes = pd.DataFrame(joint_primary_rows)
fp_vs_each_network = pd.DataFrame(pairwise_rows)
leave_one_network_out_references = pd.DataFrame(reference_rows)

direct_primary = direct_state_slopes.loc[direct_state_slopes["scenario"].isin(SOURCE_SCENARIOS)].copy()
direct_vs_joint = direct_primary.merge(
    joint_primary_slopes[["representation", "state", "slope_per_5y", "se"]].rename(columns={
        "slope_per_5y": "joint_profile_slope_per_5y", "se": "joint_profile_se",
    }),
    on=["representation", "state"], how="inner", validate="one_to_one",
)
direct_vs_joint = direct_vs_joint.rename(columns={
    "slope_per_5y": "direct_scalar_slope_per_5y", "se": "direct_scalar_se",
})
direct_vs_joint["direct_minus_joint"] = (
    direct_vs_joint["direct_scalar_slope_per_5y"] - direct_vs_joint["joint_profile_slope_per_5y"]
)
direct_vs_joint["absolute_difference"] = direct_vs_joint["direct_minus_joint"].abs()
direct_vs_joint["formal_equivalence_test_performed"] = False

reference_summary_rows = []
for representation, group in leave_one_network_out_references.groupby("representation"):
    choice = group[["reference_choice", "joint_p"]].drop_duplicates()
    reference_summary_rows.append({
        "representation": representation,
        "reference_choice_n": int(choice["reference_choice"].nunique()),
        "leave_one_out_choice_n": int(choice["reference_choice"].str.startswith("LEAVE_OUT_").sum()),
        "minimum_state_estimate": float(group["slope_per_5y"].min()),
        "minimum_z": float(group["z"].min()),
        "maximum_joint_p": float(choice["joint_p"].max()),
        "any_reference_direction_reversal": bool((group["slope_per_5y"] < 0).any()),
        "any_reference_removed_joint_evidence": bool((choice["joint_p"] >= .05).any()),
        "main_answer": (
            "YES__A_REFERENCE_DELETION_REVERSED_DIRECTION_OR_REMOVED_JOINT_EVIDENCE"
            if (group["slope_per_5y"] < 0).any() or (choice["joint_p"] >= .05).any()
            else "NO__NO_REFERENCE_DELETION_REVERSED_DIRECTION_OR_REMOVED_JOINT_EVIDENCE"
        ),
        "interpretation": "Sensitivity variants, not independent replications.",
    })
reference_deletion_summary = pd.DataFrame(reference_summary_rows)
if not fp_vs_each_network["joint_df"].eq(4).all() or not leave_one_network_out_references["joint_df"].eq(4).all():
    raise AssertionError("A reconstructed saved-joint four-state omnibus lost rank")

print(direct_vs_joint)
print(fp_vs_each_network)
print(reference_deletion_summary)

principal_tests = direct_tests.loc[
    direct_tests["scenario"].isin(["RAW_FULL", "AMPLITUDE_STANDARDIZED"])
].copy()
site_tests = principal_tests.loc[principal_tests["test"].isin(["site_x_time", "site_x_time_x_state"])].copy()

loo_state_slopes = direct_state_slopes.loc[
    direct_state_slopes["scenario"].str.contains("__LOO_", regex=False)
].copy()

site_robustness_rows = []
for representation in ["RAW_FULL", "AMPLITUDE_STANDARDIZED"]:
    local = site_state_slopes.loc[site_state_slopes["representation"].eq(representation)]
    loo = loo_state_slopes.loc[loo_state_slopes["representation"].eq(representation)]
    tests = site_tests.loc[site_tests["representation"].eq(representation)].set_index("test")
    site_time_p = float(tests.loc["site_x_time", "p"]) if "site_x_time" in tests.index else np.nan
    site_state_p = float(tests.loc["site_x_time_x_state", "p"]) if "site_x_time_x_state" in tests.index else np.nan
    all_positive = bool((local["slope_per_5y"] > 0).all()) if len(local) else False
    loo_positive = bool((loo["slope_per_5y"] > 0).all()) if len(loo) else False
    if site_state_p < .05 and all_positive:
        interpretation = "DIRECTIONALLY_CONSISTENT_WITH_HETEROGENEOUS_SITE_STATE_MAGNITUDE"
    elif site_state_p >= .05 and all_positive:
        interpretation = "DIRECTIONALLY_CONSISTENT_WITHOUT_DETECTED_SITE_STATE_MODERATION"
    elif not all_positive:
        interpretation = "SITE_STATE_POINT_DIRECTION_NOT_UNIFORM"
    else:
        interpretation = "INCONCLUSIVE"
    site_robustness_rows.append({
        "representation": representation,
        "site_x_time_p": site_time_p, "site_x_time_x_state_p": site_state_p,
        "all_adjusted_site_state_estimates_positive": all_positive,
        "all_leave_one_site_out_state_estimates_positive": loo_positive,
        "minimum_site_state_estimate": float(local["slope_per_5y"].min()) if len(local) else np.nan,
        "minimum_loo_estimate": float(loo["slope_per_5y"].min()) if len(loo) else np.nan,
        "maximum_loo_estimate": float(loo["slope_per_5y"].max()) if len(loo) else np.nan,
        "interpretation": interpretation,
        "replication_claim_allowed": False,
    })
site_robustness_summary = pd.DataFrame(site_robustness_rows)

print(site_tests)
print(site_state_slopes)
print(loo_state_slopes)
print(site_robustness_summary)

saved_network_slopes_path = SOURCE_PROFILE_TABLE_DIR / "network_profile_state_slopes.csv"
saved_parcel_counts_path = SOURCE_PROFILE_TABLE_DIR / "parcel_counts.csv"
if not saved_network_slopes_path.is_file() or not saved_parcel_counts_path.is_file():
    raise FileNotFoundError("Saved joint-profile network slopes or parcel-count table is missing")
saved_network_slopes = pd.read_csv(saved_network_slopes_path)
saved_parcel_counts = pd.read_csv(saved_parcel_counts_path)
network_size_data = saved_network_slopes.loc[
    saved_network_slopes["scenario"].eq("FULL_PRIMARY")
    & saved_network_slopes["network"].isin(ELIGIBLE_NETWORKS)
].merge(saved_parcel_counts, on="network", how="left", validate="many_to_one")
if len(network_size_data) != K * len(STATES):
    raise AssertionError(f"Expected {K * len(STATES)} network-size rows, found {len(network_size_data)}")

expected_focal_counts = {"Frontoparietal": 50, "Default": 77, "Cingulo-Opercular": 56}
focal_count_audit = pd.DataFrame([{
    "network": network, "expected_parcel_n": expected,
    "observed_parcel_n": int(saved_parcel_counts.set_index("network").loc[network, "parcel_n"]),
} for network, expected in expected_focal_counts.items()])
focal_count_audit["matches_expected"] = focal_count_audit["expected_parcel_n"].eq(focal_count_audit["observed_parcel_n"])
if not FAST_MODE and not focal_count_audit["matches_expected"].all():
    print(focal_count_audit)
    raise AssertionError("FP/DMN/CO parcel counts differ from the prespecified values")

size_rows = []
for state in STATES:
    part = network_size_data.loc[network_size_data["state"].eq(state)].copy()
    if part["parcel_n"].nunique() < 2 or part["relative_profile_slope_per_5y"].nunique() < 2:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
    else:
        pearson_r, pearson_p = pearsonr(part["parcel_n"], part["relative_profile_slope_per_5y"])
        spearman_r, spearman_p = spearmanr(part["parcel_n"], part["relative_profile_slope_per_5y"])
    size_rows.append({
        "state": state, "network_n": len(part),
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "weighted_analysis_performed": False,
    })
network_size_correlations = pd.DataFrame(size_rows)
print(focal_count_audit)
print(network_size_correlations)


# Refit the amplitude-standardized joint profile once to recover pedigree
# influence functions. Coefficients and robust covariance must agree with the
# corresponding script-02 checkpoint.
gate_frame = FRAME_REGISTRY["AMPLITUDE_STANDARDIZED"].copy()
source_checkpoint_path = (
    SOURCE_PROFILE_CHECKPOINT_DIR / "profile__amplitude_standardized.json"
)
source_slopes_path = SOURCE_PROFILE_TABLE_DIR / "network_profile_state_slopes.csv"
source_tests_path = SOURCE_PROFILE_TABLE_DIR / "profile_omnibus_tests.csv"
for required_path in [source_checkpoint_path, source_slopes_path, source_tests_path]:
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Required script-02 output is missing: {required_path}"
        )

saved_checkpoint = JOINT_SAVED["AMPLITUDE_STANDARDIZED"]["saved"]
saved_payload = saved_checkpoint["result"]["payload"]
profile_columns = [f"profile__{network}" for network in ELIGIBLE_NETWORKS]
required_columns = [*profile_columns, "age0_c5", "t5"]
fit_frame = gate_frame.loc[
    np.isfinite(gate_frame[required_columns].to_numpy(float)).all(axis=1)
].copy()

design = DirectStateDesign().fit(fit_frame)
x = design.transform(fit_frame)
fit_with_influence = fit_multivariate_profile(
    x,
    fit_frame[profile_columns],
    fit_frame["id"],
    fit_frame["state"],
    fit_frame["pedigree_group"],
)
if fit_with_influence is None:
    raise RuntimeError("Amplitude-standardized joint profile refit failed")

beta_refit = np.asarray(fit_with_influence["coef"], float)
cov_refit = np.asarray(fit_with_influence["robust_cov"], float)
basis_refit = np.asarray(fit_with_influence["basis"], float)
influence_by_pedigree = fit_with_influence["pedigree_influence"]

beta_saved = np.asarray(saved_payload["coef"], float)
cov_saved = np.asarray(saved_payload["cov"], float)
basis_saved = np.asarray(saved_payload["basis"], float)
names_saved = list(saved_payload["names"])
if names_saved != design.names:
    raise AssertionError("Design-column names differ from the script-02 checkpoint")

beta_max_abs_diff = float(np.max(np.abs(beta_refit - beta_saved)))
cov_max_abs_diff = float(np.max(np.abs(cov_refit - cov_saved)))
basis_max_abs_diff = float(np.max(np.abs(basis_refit - basis_saved)))
if beta_max_abs_diff > SOURCE_ESTIMATE_TOL:
    raise AssertionError(
        f"Refit coefficient mismatch: {beta_max_abs_diff:.3e}"
    )
if cov_max_abs_diff > SOURCE_COV_TOL:
    raise AssertionError(
        f"Refit robust-covariance mismatch: {cov_max_abs_diff:.3e}"
    )
if basis_max_abs_diff > IDENTITY_TOL:
    raise AssertionError(f"Helmert-basis mismatch: {basis_max_abs_diff:.3e}")

if_vectors = {
    pedigree: np.asarray(value, float).ravel(order="F")
    for pedigree, value in influence_by_pedigree.items()
}
cov_from_influence = sum(np.outer(value, value) for value in if_vectors.values())
influence_cov_error = float(np.max(np.abs(cov_from_influence - cov_refit)))
if influence_cov_error > 1e-10:
    raise AssertionError(
        f"Pedigree influences do not reconstruct covariance: {influence_cov_error:.3e}"
    )

source_reconciliation = pd.DataFrame([{
    "participants": int(fit_with_influence["participant_n"]),
    "pedigrees": int(fit_with_influence["pedigree_n"]),
    "rows": int(fit_with_influence["row_n"]),
    "optimizer_success": bool(fit_with_influence["optimizer_success"]),
    "beta_max_abs_refit_minus_checkpoint": beta_max_abs_diff,
    "robust_cov_max_abs_refit_minus_checkpoint": cov_max_abs_diff,
    "basis_max_abs_refit_minus_checkpoint": basis_max_abs_diff,
    "influence_cov_reconstruction_error": influence_cov_error,
}])

profile_results = pd.read_csv(source_slopes_path)
amplitude_profile = profile_results.loc[
    profile_results["scenario"].eq("AMPLITUDE_STANDARDIZED")
    & profile_results["network"].isin(ELIGIBLE_NETWORKS)
].copy()
if len(amplitude_profile) != len(ELIGIBLE_NETWORKS) * len(STATES):
    raise AssertionError("Expected 40 amplitude-standardized network-state slopes")

profile_tests_all = pd.read_csv(source_tests_path)
amplitude_profile_tests = profile_tests_all.loc[
    profile_tests_all["scenario"].eq("AMPLITUDE_STANDARDIZED")
].copy()


# Nine-comparator familywise specificity analysis

network_index = {network: j for j, network in enumerate(ELIGIBLE_NETWORKS)}
q_coord = basis_saved.shape[1]
c_state = design.state_slope_matrix()


def network_vector(weight, state_contrast):
    return np.kron(np.asarray(weight, float) @ basis_saved, np.asarray(state_contrast, float))


# Verify that the script-02 checkpoint reconstructs the script-02 40-row table.
reconstructed_rows = []
for network in ELIGIBLE_NETWORKS:
    weight = np.zeros(K)
    weight[network_index[network]] = 1.0
    for state_i, state in enumerate(STATES):
        vector = network_vector(weight, c_state[state_i])
        estimate, se, z, p = scalar_contrast(beta_saved, cov_saved, vector)
        reconstructed_rows.append({
            "network": network,
            "state": state,
            "reconstructed_slope": estimate,
            "reconstructed_se": se,
        })
reconstructed_profile = pd.DataFrame(reconstructed_rows)

profile_compare = amplitude_profile.merge(
    reconstructed_profile,
    on=["network", "state"],
    how="inner",
    validate="one_to_one",
)
profile_compare["slope_abs_error"] = (
    profile_compare["relative_profile_slope_per_5y"]
    - profile_compare["reconstructed_slope"]
).abs()
profile_compare["se_abs_error"] = (
    profile_compare["se"] - profile_compare["reconstructed_se"]
).abs()

profile_table_reconstruction_error = float(
    profile_compare[["slope_abs_error", "se_abs_error"]].to_numpy(float).max()
)
if profile_table_reconstruction_error > 1e-10:
    raise AssertionError(
        f"Script-02 checkpoint does not reconstruct its 40-row slope table: "
        f"max error={profile_table_reconstruction_error:.3e}"
    )

full_profile = amplitude_profile.copy()
full_profile["within_state_rank_descending"] = (
    full_profile.groupby("state")["relative_profile_slope_per_5y"]
    .rank(method="min", ascending=False)
)
full_profile["within_state_rank_absolute"] = (
    full_profile.assign(
        _abs=full_profile["relative_profile_slope_per_5y"].abs()
    )
    .groupby("state")["_abs"]
    .rank(method="min", ascending=False)
)
equal_state_mean = (
    full_profile.groupby("network")["relative_profile_slope_per_5y"].mean()
)
full_profile["equal_state_mean_slope"] = (
    full_profile["network"].map(equal_state_mean)
)

# Whole-profile network-specificity omnibus: one pre-script-02 amplitude-standardized
# time×network test.
profile_time_row = amplitude_profile_tests.loc[
    amplitude_profile_tests["test"].eq("time_x_network")
]
if len(profile_time_row) != 1:
    raise AssertionError("Expected exactly one amplitude-standardized time_x_network test")
profile_time_p = float(profile_time_row["p"].iloc[0])

# Recompute from the saved payload as a provenance check.
pooled_network_matrix = block_expand(c_state.mean(axis=0)[None, :], q_coord)
profile_time_stat_reconstructed, profile_time_df_reconstructed, profile_time_p_reconstructed = (
    wald_matrix(beta_saved, cov_saved, pooled_network_matrix)
)
profile_time_p_error = abs(profile_time_p - profile_time_p_reconstructed)
if profile_time_p_error > 1e-10:
    raise AssertionError(
        f"Script-02 profile omnibus mismatch: |Δp|={profile_time_p_error:.3e}"
    )

print(full_profile)
print({
    "time_x_network_p": profile_time_p,
    "time_x_network_df": profile_time_df_reconstructed,
})

comparators = [network for network in ELIGIBLE_NETWORKS if network != FP_NETWORK]
if len(comparators) != 9:
    raise AssertionError(f"Expected nine FP comparators; found {comparators}")


def fp_minus_network_weight(other):
    weight = np.zeros(K)
    weight[network_index[FP_NETWORK]] = 1.0
    weight[network_index[other]] = -1.0
    if abs(weight.sum()) > 1e-12:
        raise AssertionError("FP-vs-network weight must sum to zero")
    return weight


contrast_matrices = []
maxwald_rows = []
contrast_cov_inverses = []
contrast_covariances = []

for other in comparators:
    weight = fp_minus_network_weight(other)
    C = np.vstack([
        network_vector(weight, c_state[state_i])
        for state_i in range(len(STATES))
    ])
    contrast_matrices.append(C)

    estimates = C @ beta_saved
    V = C @ cov_saved @ C.T
    V = (V + V.T) / 2
    rank = int(np.linalg.matrix_rank(V, tol=1e-12))
    if rank != 4:
        raise AssertionError(
            f"FP-vs-{other} four-state covariance lost rank: {rank}"
        )
    Vinv = np.linalg.pinv(V, rcond=1e-12)
    W = float(estimates @ Vinv @ estimates)
    p_chi2 = float(chi2.sf(W, rank))
    contrast_covariances.append(V)
    contrast_cov_inverses.append(Vinv)

    row = {
        "comparator": other,
        "wald_chi2_observed": W,
        "wald_df": rank,
        "p_chi2_unadjusted": p_chi2,
    }
    for state_i, state in enumerate(STATES):
        row[f"{state}_difference"] = float(estimates[state_i])
        row[f"{state}_se"] = float(np.sqrt(max(V[state_i, state_i], 0.0)))
    maxwald_rows.append(row)

contrast_matrices = np.stack(contrast_matrices, axis=0)          # 9 x 4 x P
contrast_cov_inverses = np.stack(contrast_cov_inverses, axis=0)  # 9 x 4 x 4

# Project pedigree coefficient influences once into the 9×4 family.
pedigrees = sorted(if_vectors)
coef_if_matrix = np.vstack([if_vectors[pg] for pg in pedigrees])  # G x P
family_if = np.stack(
    [coef_if_matrix @ contrast_matrices[j].T for j in range(len(comparators))],
    axis=1,
)  # G x 9 x 4
family_if_flat = family_if.reshape(len(pedigrees), -1)

# Sanity check: projected influences reproduce every saved contrast covariance
# within the source/refit covariance reconciliation tolerance.
projected_cov_errors = []
for j in range(len(comparators)):
    vif = family_if[:, j, :]
    projected = vif.T @ vif
    projected_cov_errors.append(
        float(np.max(np.abs(projected - contrast_covariances[j])))
    )
max_projected_cov_error = max(projected_cov_errors)
PROJECTED_COV_TOL = max(2e-5, 20.0 * SOURCE_COV_TOL)
if max_projected_cov_error > PROJECTED_COV_TOL:
    raise AssertionError(
        f"Projected influence covariance differs from script-02 contrast covariance: "
        f"{max_projected_cov_error:.3e} > {PROJECTED_COV_TOL:.3e}"
    )

# Equal-state-average SEs for the max-|t| family.
mean_vectors = contrast_matrices.mean(axis=1)  # 9 x P
mean_estimates = mean_vectors @ beta_saved
mean_variances = np.einsum("jp,pq,jq->j", mean_vectors, cov_saved, mean_vectors)
mean_ses = np.sqrt(np.maximum(mean_variances, 0.0))
if np.any(~np.isfinite(mean_ses)) or np.any(mean_ses <= 0):
    raise AssertionError("At least one equal-state-average comparator SE is invalid")

rng = np.random.default_rng(BOOT_SEED)
max_wald_draws = np.empty(BOOT_B, float)
max_abs_t_draws = np.empty(BOOT_B, float)

chunk = 500
cursor = 0
with tqdm(total=BOOT_B, desc="FP comparator multiplier family") as progress:
    while cursor < BOOT_B:
        n = min(chunk, BOOT_B - cursor)
        multipliers = rng.standard_normal((n, len(pedigrees)))
        perturb = (multipliers @ family_if_flat).reshape(
            n, len(comparators), len(STATES)
        )

        wald_star = np.einsum(
            "bji,jik,bjk->bj",
            perturb,
            contrast_cov_inverses,
            perturb,
            optimize=True,
        )
        max_wald_draws[cursor:cursor+n] = np.max(wald_star, axis=1)

        mean_perturb = perturb.mean(axis=2)
        t_star = mean_perturb / mean_ses[None, :]
        max_abs_t_draws[cursor:cursor+n] = np.max(np.abs(t_star), axis=1)

        cursor += n
        progress.update(n)

maxwald_critical_95 = float(np.quantile(max_wald_draws, 1 - ALPHA, method="higher"))
maxt_critical_95 = float(np.quantile(max_abs_t_draws, 1 - ALPHA, method="higher"))

fp_vs_network_maxwald = pd.DataFrame(maxwald_rows)
fp_vs_network_maxwald["p_fwer_maxwald"] = [
    float((1 + np.sum(max_wald_draws >= value)) / (BOOT_B + 1))
    for value in fp_vs_network_maxwald["wald_chi2_observed"]
]
fp_vs_network_maxwald["maxwald_critical_95"] = maxwald_critical_95
fp_vs_network_maxwald["reject_fwer_0_05"] = (
    fp_vs_network_maxwald["p_fwer_maxwald"] < ALPHA
)
fp_vs_network_maxwald["bootstrap_B"] = BOOT_B
fp_vs_network_maxwald["pedigrees"] = len(pedigrees)
fp_vs_network_maxwald["bootstrap_seed"] = BOOT_SEED

fp_vs_network_maxt_intervals = pd.DataFrame({
    "comparator": comparators,
    "equal_state_mean_difference": mean_estimates,
    "se": mean_ses,
    "simultaneous_ci_low": mean_estimates - maxt_critical_95 * mean_ses,
    "simultaneous_ci_high": mean_estimates + maxt_critical_95 * mean_ses,
    "maxt_critical_95": maxt_critical_95,
    "bootstrap_B": BOOT_B,
    "pedigrees": len(pedigrees),
    "bootstrap_seed": BOOT_SEED,
})
fp_vs_network_maxt_intervals["simultaneous_ci_excludes_zero"] = (
    (fp_vs_network_maxt_intervals["simultaneous_ci_low"] > 0)
    | (fp_vs_network_maxt_intervals["simultaneous_ci_high"] < 0)
)

multiplier_diagnostics = pd.DataFrame([{
    "bootstrap_B": BOOT_B,
    "pedigrees": len(pedigrees),
    "seed": BOOT_SEED,
    "maxwald_critical_95": maxwald_critical_95,
    "maxt_critical_95": maxt_critical_95,
    "max_projected_covariance_error_vs_script-02": max_projected_cov_error,
    "projected_covariance_tolerance": PROJECTED_COV_TOL,
    "same_pedigree_multipliers_for_all_nine_contrasts": True,
    "null_multiplier_statistics_centered_at_zero": True,
}])

print(fp_vs_network_maxwald)
print(fp_vs_network_maxt_intervals)
print(multiplier_diagnostics)


def fp_reference_weight(excluded_network=None):
    weight = np.zeros(K)
    weight[network_index[FP_NETWORK]] = 1.0
    reference = [
        network for network in ELIGIBLE_NETWORKS
        if network != FP_NETWORK and network != excluded_network
    ]
    for network in reference:
        weight[network_index[network]] = -1.0 / len(reference)
    if abs(weight.sum()) > 1e-12:
        raise AssertionError("FP-reference weight does not sum to zero")
    return weight


def reference_choice_row(label, excluded_network=None):
    weight = fp_reference_weight(excluded_network)
    matrix = np.vstack([
        network_vector(weight, c_state[state_index])
        for state_index in range(len(STATES))
    ])
    estimates = matrix @ beta_saved
    covariance = matrix @ cov_saved @ matrix.T
    statistic, df, p_value = wald_matrix(beta_saved, cov_saved, matrix)
    row = {
        "reference_choice": label,
        "omitted_network": excluded_network if excluded_network is not None else "",
        "mean_four_state_slope": float(estimates.mean()),
        "joint_chi2": statistic,
        "joint_df": df,
        "joint_p": p_value,
    }
    for state_index, state in enumerate(STATES):
        row[f"{state}_slope"] = float(estimates[state_index])
        row[f"{state}_se"] = float(
            np.sqrt(max(covariance[state_index, state_index], 0.0))
        )
    return row


reference_rows = [reference_choice_row("ALL_NINE")]
for comparator in comparators:
    reference_rows.append(
        reference_choice_row(f"LEAVE_OUT_{comparator}", comparator)
    )
reference_driver_sensitivity = pd.DataFrame(reference_rows)

all_nine_mean = float(
    reference_driver_sensitivity.loc[
        reference_driver_sensitivity["reference_choice"].eq("ALL_NINE"),
        "mean_four_state_slope",
    ].iloc[0]
)
reference_driver_sensitivity["change_from_all_nine_mean"] = (
    reference_driver_sensitivity["mean_four_state_slope"] - all_nine_mean
)
for state in STATES:
    baseline = float(
        reference_driver_sensitivity.loc[
            reference_driver_sensitivity["reference_choice"].eq("ALL_NINE"),
            f"{state}_slope",
        ].iloc[0]
    )
    reference_driver_sensitivity[f"{state}_change_from_all_nine"] = (
        reference_driver_sensitivity[f"{state}_slope"] - baseline
    )
reference_driver_sensitivity["max_abs_state_change_from_all_nine"] = (
    reference_driver_sensitivity[
        [f"{state}_change_from_all_nine" for state in STATES]
    ].abs().max(axis=1)
)
reference_driver_sensitivity["driver_rank_by_abs_mean_change"] = (
    reference_driver_sensitivity["change_from_all_nine_mean"].abs()
    .where(~reference_driver_sensitivity["reference_choice"].eq("ALL_NINE"))
    .rank(method="min", ascending=False)
)

# Confirm that the newly reconstructed amplitude-standardized reference
# sensitivities match the independently reconstructed direct-analysis table.
existing_reference = leave_one_network_out_references.loc[
    leave_one_network_out_references["representation"].eq("AMPLITUDE_STANDARDIZED")
].copy()
existing_long = existing_reference[
    ["reference_choice", "state", "slope_per_5y"]
]
reconstructed_long = reference_driver_sensitivity.melt(
    id_vars=["reference_choice"],
    value_vars=[f"{state}_slope" for state in STATES],
    var_name="state",
    value_name="reconstructed_slope",
)
reconstructed_long["state"] = reconstructed_long["state"].str.replace(
    "_slope", "", regex=False
)
reference_compare = existing_long.merge(
    reconstructed_long,
    on=["reference_choice", "state"],
    how="inner",
    validate="one_to_one",
)
reference_compare["absolute_error"] = (
    reference_compare["slope_per_5y"]
    - reference_compare["reconstructed_slope"]
).abs()
reference_reconstruction_error = float(reference_compare["absolute_error"].max())
if len(reference_compare) != 40 or reference_reconstruction_error > 1e-10:
    raise AssertionError("FP reference-sensitivity reconstructions disagree")

fp_profile = full_profile.loc[
    full_profile["network"].eq(FP_NETWORK)
].set_index("state").reindex(STATES)
fp_positive_all_four = bool(
    (fp_profile["relative_profile_slope_per_5y"] > 0).all()
)
profile_omnibus_supported = bool(
    np.isfinite(profile_time_p) and profile_time_p < ALPHA
)
n_distinguishable = int(fp_vs_network_maxwald["reject_fwer_0_05"].sum())
n_mean_intervals_excluding_zero = int(
    fp_vs_network_maxt_intervals["simultaneous_ci_excludes_zero"].sum()
)

if not profile_omnibus_supported:
    specificity_status = "NO_WHOLE_PROFILE_RECONFIGURATION"
    interpretation = "Whole-profile longitudinal reconfiguration was not detected."
elif fp_positive_all_four and n_distinguishable == 9:
    specificity_status = "FP_SPECIFIC"
    interpretation = "FP differed from every comparator trajectory."
elif fp_positive_all_four:
    specificity_status = "FP_PROMINENT_NOT_UNIQUE"
    interpretation = (
        "Network-relative reconfiguration was broad, with FP among its prominent features."
    )
else:
    specificity_status = "PROFILE_RECONFIGURATION_WITHOUT_STATE_GENERAL_POSITIVE_FP"
    interpretation = (
        "Network-relative reconfiguration was detected without positive FP change in every state."
    )

fp_specificity_summary = pd.DataFrame([{
    "status": specificity_status,
    "profile_time_x_network_p": profile_time_p,
    "profile_omnibus_supported": profile_omnibus_supported,
    "fp_positive_all_four_states": fp_positive_all_four,
    "fp_vs_network_trajectory_rejections_fwer": n_distinguishable,
    "fp_vs_network_trajectory_total": len(comparators),
    "equal_state_mean_intervals_excluding_zero": n_mean_intervals_excluding_zero,
    "equal_state_mean_total": len(comparators),
    "interpretation": interpretation,
    "state_generality_note": (
        "The four states assess recurrence across states, not independent replication."
    ),
}])


# Diagnostic figures

# Direct raw/amplitude state slopes.
plot_primary = direct_state_slopes.loc[
    direct_state_slopes["scenario"].isin(["RAW_FULL", "AMPLITUDE_STANDARDIZED", "PARTIAL_MATCHED"])
].copy()
fig, ax = plt.subplots(figsize=(9, 5.2))
for offset, (scenario, part) in zip([-0.18, 0, 0.18], plot_primary.groupby("scenario", sort=False)):
    part = part.set_index("state").reindex(STATES).reset_index()
    y = np.arange(len(STATES)) + offset
    ax.errorbar(
        part["slope_per_5y"], y,
        xerr=np.vstack([
            part["slope_per_5y"] - part["simultaneous_ci_low"],
            part["simultaneous_ci_high"] - part["slope_per_5y"],
        ]), fmt="o", capsize=3, label=scenario,
    )
ax.axvline(0, color="black", lw=1)
ax.set_yticks(np.arange(len(STATES))); ax.set_yticklabels(STATES)
ax.set_xlabel("FP relative-preservation slope per 5 years")
ax.set_title("Direct scalar FP-relative preservation")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "direct_fp_relative_state_slopes.png", dpi=220, bbox_inches="tight")
fig.savefig(FIG_DIR / "direct_fp_relative_state_slopes.pdf", bbox_inches="tight")
plt.close("all")

# Raw and amplitude leave-one-site-out forest.
if len(loo_state_slopes):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, representation in zip(axes, ["RAW_FULL", "AMPLITUDE_STANDARDIZED"]):
        part = loo_state_slopes.loc[loo_state_slopes["representation"].eq(representation)].copy()
        sites = sorted(part["site_left_out"].dropna().astype(str).unique())
        for j, site in enumerate(sites):
            one = part.loc[part["site_left_out"].astype(str).eq(site)].set_index("state").reindex(STATES).reset_index()
            y = np.arange(len(STATES)) + (j - (len(sites) - 1) / 2) * .10
            ax.errorbar(one["slope_per_5y"], y, xerr=1.96 * one["se"], fmt="o", capsize=2, label=f"leave out {site}")
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(np.arange(len(STATES))); ax.set_yticklabels(STATES)
        ax.set_title(representation); ax.set_xlabel("slope per 5 years")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Leave-one-site-out FP-relative preservation (sensitivity, not replication)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fp_relative_leave_one_site_out_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fp_relative_leave_one_site_out_forest.pdf", bbox_inches="tight")
    plt.close("all")

# Network-size scatterplots.
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
for ax, state in zip(axes.flat, STATES):
    part = network_size_data.loc[network_size_data["state"].eq(state)]
    ax.scatter(part["parcel_n"], part["relative_profile_slope_per_5y"], color="0.65", s=42)
    focus = part.loc[part["network"].isin(FOCAL_NETWORKS)]
    ax.scatter(focus["parcel_n"], focus["relative_profile_slope_per_5y"], color="#C44E52", s=64)
    for row in focus.itertuples():
        ax.annotate(row.network, (row.parcel_n, row.relative_profile_slope_per_5y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title(state); ax.set_xlabel("released parcel count"); ax.set_ylabel("relative-profile slope / 5y")
fig.suptitle("Network size is descriptive; networks remain the analysis units")
fig.tight_layout()
fig.savefig(FIG_DIR / "network_size_relative_slope_diagnostic.png", dpi=220, bbox_inches="tight")
fig.savefig(FIG_DIR / "network_size_relative_slope_diagnostic.pdf", bbox_inches="tight")
plt.close("all")


# Figure 1: full 10×4 prespecified profile with the existing simultaneous intervals.
fig, ax = plt.subplots(figsize=(9.5, 6.8))
network_order = (
    full_profile.groupby("network")["equal_state_mean_slope"].first()
    .sort_values()
    .index.tolist()
)
y_base = np.arange(len(network_order))
offsets = np.linspace(-0.24, 0.24, len(STATES))

for offset, state in zip(offsets, STATES):
    part = (
        full_profile.loc[full_profile["state"].eq(state)]
        .set_index("network").reindex(network_order).reset_index()
    )
    y = y_base + offset
    ax.errorbar(
        part["relative_profile_slope_per_5y"],
        y,
        xerr=np.vstack([
            part["relative_profile_slope_per_5y"] - part["simultaneous_ci_low"],
            part["simultaneous_ci_high"] - part["relative_profile_slope_per_5y"],
        ]),
        fmt="o",
        capsize=2,
        label=state,
    )
ax.axvline(0, lw=1, ls="--")
ax.set_yticks(y_base)
ax.set_yticklabels(network_order)
ax.set_xlabel("Amplitude-standardized relative-profile slope per 5 years")
ax.set_title("Amplitude-standardized ten-network profile")
ax.legend(frameon=False, ncol=4)
fig.tight_layout()
fig.savefig(FIG_DIR / "full_profile_10x4.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG_DIR / "full_profile_10x4.pdf", bbox_inches="tight")
plt.close("all")


# Figure 2: equal-state-average FP-minus-network effects with one max-|t| family.
forest = fp_vs_network_maxt_intervals.sort_values(
    "equal_state_mean_difference"
).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 6.2))
y = np.arange(len(forest))
ax.errorbar(
    forest["equal_state_mean_difference"],
    y,
    xerr=np.vstack([
        forest["equal_state_mean_difference"] - forest["simultaneous_ci_low"],
        forest["simultaneous_ci_high"] - forest["equal_state_mean_difference"],
    ]),
    fmt="o",
    capsize=3,
)
ax.axvline(0, lw=1, ls="--")
ax.set_yticks(y)
ax.set_yticklabels(forest["comparator"])
ax.set_xlabel("Equal-state-average FP minus comparator slope / 5 years")
ax.set_title("FP versus each network: simultaneous max-|t| 95% intervals")
fig.tight_layout()
fig.savefig(FIG_DIR / "fp_vs_each_network_maxt_forest.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG_DIR / "fp_vs_each_network_maxt_forest.pdf", bbox_inches="tight")
plt.close("all")


# Figure 3: leave-one-network-out reference sensitivity.
refplot = reference_driver_sensitivity.loc[
    ~reference_driver_sensitivity["reference_choice"].eq("ALL_NINE")
].sort_values("mean_four_state_slope").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 6.2))
y = np.arange(len(refplot))
ax.plot(refplot["mean_four_state_slope"], y, "o")
ax.axvline(all_nine_mean, lw=1, ls="--", label="All-nine reference")
ax.set_yticks(y)
ax.set_yticklabels(refplot["omitted_network"])
ax.set_xlabel("Mean four-state FP-relative slope / 5 years")
ax.set_ylabel("Network omitted from nine-network reference")
ax.set_title("Leave-one-network-out FP reference sensitivity")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fp_reference_deletion_sensitivity.png", dpi=240, bbox_inches="tight")
fig.savefig(FIG_DIR / "fp_reference_deletion_sensitivity.pdf", bbox_inches="tight")
plt.close("all")



# Analysis summaries and exports

def direct_p(representation, test):
    row = direct_tests.loc[
        direct_tests["scenario"].eq(representation) & direct_tests["test"].eq(test), "p"
    ]
    return float(row.iloc[0]) if len(row) else np.nan


def all_direct_positive(representation):
    rows = direct_state_slopes.loc[direct_state_slopes["scenario"].eq(representation)]
    return bool((rows["slope_per_5y"] > 0).all()) if len(rows) == 4 else False


raw_p = direct_p("RAW_FULL", "all_four_state_slopes_zero")
amp_p = direct_p("AMPLITUDE_STANDARDIZED", "all_four_state_slopes_zero")
partial_p = direct_p("PARTIAL_MATCHED", "all_four_state_slopes_zero")
raw_site_state_p = direct_p("RAW_FULL", "site_x_time_x_state")
raw_site_all_positive = bool(site_robustness_summary.loc[
    site_robustness_summary["representation"].eq("RAW_FULL"),
    "all_adjusted_site_state_estimates_positive",
].iloc[0])
all_loo_positive = bool(site_robustness_summary[
    "all_leave_one_site_out_state_estimates_positive"
].all())
reference_robust_by_representation = (
    ~reference_deletion_summary["any_reference_direction_reversal"]
    & ~reference_deletion_summary["any_reference_removed_joint_evidence"]
)
reference_robust_all = bool(reference_robust_by_representation.all())
reference_robust_principal = bool(reference_robust_by_representation.loc[
    reference_deletion_summary["representation"].isin(["RAW_FULL", "AMPLITUDE_STANDARDIZED"])
].all())

if raw_p < .05 and amp_p < .05 and all_direct_positive("RAW_FULL") and all_direct_positive("AMPLITUDE_STANDARDIZED") and all_loo_positive and reference_robust_principal:
    if raw_site_state_p < .05 and raw_site_all_positive:
        final_claim = "FP_RELATIVE_PRESERVATION_CONFIRMED__SITE_STATE_MAGNITUDE_HETEROGENEOUS"
    elif raw_site_state_p >= .05:
        final_claim = "FP_RELATIVE_PRESERVATION_CONFIRMED__NO_DETECTED_SITE_STATE_MODERATION"
    else:
        final_claim = "FP_RELATIVE_PRESERVATION_CONFIRMED__SITE_DIRECTION_NOT_UNIFORM"
elif raw_p < .05 and not (amp_p < .05):
    final_claim = "RAW_ONLY__NOT_CONFIRMED_AFTER_FIXED_AMPLITUDE_STANDARDIZATION"
else:
    final_claim = "FP_RELATIVE_PRESERVATION_NOT_CONFIRMED"

final_claim_decision = pd.DataFrame([
    {"dimension": "raw_full_covariance", "four_state_p": raw_p, "all_state_slopes_positive": all_direct_positive("RAW_FULL"), "role": "PRINCIPAL"},
    {"dimension": "amplitude_standardized_covariance", "four_state_p": amp_p, "all_state_slopes_positive": all_direct_positive("AMPLITUDE_STANDARDIZED"), "role": "PRINCIPAL_NOT_CORRELATION"},
    {"dimension": "partial_covariance", "four_state_p": partial_p, "all_state_slopes_positive": all_direct_positive("PARTIAL_MATCHED"), "role": "SECONDARY_MATCHED_ROWS"},
    {"dimension": "reference_deletion_principal", "four_state_p": np.nan, "all_state_slopes_positive": reference_robust_principal, "role": "RAW_AND_AMPLITUDE_SENSITIVITY_VARIANTS_NOT_REPLICATIONS"},
    {"dimension": "reference_deletion_partial", "four_state_p": np.nan, "all_state_slopes_positive": bool(reference_robust_by_representation.loc[reference_deletion_summary["representation"].eq("PARTIAL_MATCHED")].all()), "role": "SECONDARY_SENSITIVITY_VARIANTS_NOT_REPLICATIONS"},
    {"dimension": "site", "four_state_p": raw_site_state_p, "all_state_slopes_positive": raw_site_all_positive, "role": "DIRECTION_AND_MAGNITUDE_NOT_REPLICATION"},
])
final_claim_decision["final_claim"] = final_claim


outputs = {
    "source_fit_reconciliation.csv": source_fit_audit,
    "fp_relative_identity.csv": fp_relative_identity_audit,
    "cohort_summary.csv": cohort_summary,
    "network_eligibility.csv": network_eligibility,
    "parcel_counts.csv": parcel_counts,
    "fp_relative_model_summary.csv": direct_model_summary,
    "fp_relative_tests.csv": direct_tests,
    "fp_relative_state_slopes.csv": direct_state_slopes,
    "fp_relative_site_state_slopes.csv": site_state_slopes,
    "fp_relative_leave_one_site_out_slopes.csv": loo_state_slopes,
    "fp_relative_site_robustness.csv": site_robustness_summary,
    "direct_vs_joint_profile.csv": direct_vs_joint,
    "fp_vs_network_descriptive.csv": fp_vs_each_network,
    "fp_reference_deletion_all_representations.csv": leave_one_network_out_references,
    "fp_reference_deletion_summary.csv": reference_deletion_summary,
    "network_size_diagnostic.csv": network_size_data,
    "network_size_correlations.csv": network_size_correlations,
    "focal_network_parcel_counts.csv": focal_count_audit,
    "fp_relative_summary.csv": final_claim_decision,
    "joint_refit_reconciliation.csv": source_reconciliation,
    "amplitude_standardized_profile_10x4.csv": full_profile,
    "fp_vs_network_maxwald.csv": fp_vs_network_maxwald,
    "fp_vs_network_maxt_intervals.csv": fp_vs_network_maxt_intervals,
    "fp_reference_driver_sensitivity.csv": reference_driver_sensitivity,
    "fp_specificity_summary.csv": fp_specificity_summary,
    "multiplier_diagnostics.csv": multiplier_diagnostics,
}
for filename, table in outputs.items():
    table.to_csv(TABLE_DIR / filename, index=False)

run_manifest = {
    "script": "03_fp_specificity_and_robustness.py",
    "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "participants": int(profile_data["id"].nunique()),
    "eligible_networks": ELIGIBLE_NETWORKS,
    "profile_time_x_network_p": profile_time_p,
    "fp_positive_all_four_states": fp_positive_all_four,
    "fp_vs_network_trajectory_rejections_fwer": n_distinguishable,
    "equal_state_mean_intervals_excluding_zero": n_mean_intervals_excluding_zero,
    "specificity_status": specificity_status,
    "bootstrap_B": BOOT_B,
    "bootstrap_seed": BOOT_SEED,
    "tables": sorted(outputs),
}
(OUTPUT_DIR / "run_manifest.json").write_text(
    json.dumps(run_manifest, indent=2) + "\n"
)

print("\nFP SPECIFICITY AND ROBUSTNESS ANALYSIS COMPLETE")
print(f"Participants: {profile_data['id'].nunique():,}")
print(f"Eligible networks: {len(ELIGIBLE_NETWORKS)}")
print(f"FP-versus-network trajectory differences: {n_distinguishable}/9")
print(
    "Equal-state-average intervals excluding zero: "
    f"{n_mean_intervals_excluding_zero}/9"
)
print(f"Conclusion: {specificity_status}")
print(f"Outputs: {OUTPUT_DIR}")
