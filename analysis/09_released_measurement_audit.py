#!/usr/bin/env python3
"""Audit released covariance, amplitude, and correlation products and their alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable=None, **kwargs):
        return iterable


STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
CANONICAL_NETWORKS = [
    "Visual1",
    "Visual2",
    "Somatomotor",
    "Cingulo-Opercular",
    "Dorsal-attention",
    "Language",
    "Frontoparietal",
    "Auditory",
    "Default",
    "Posterior-Multimodal",
    "Ventral-Multimodal",
    "Orbito-Affective",
]
DEFAULT_ELIGIBLE_NETWORKS = CANONICAL_NETWORKS[:10]
EXPECTED_PARCELS = 360
DEFAULT_DIAGNOSTIC_SESSIONS_PER_STATE = 3
CORRELATION_BOUND = 1.01
AGGREGATION_RTOL = 1e-6
AGGREGATION_ATOL = 1e-10


def parse_args() -> argparse.Namespace:
    code_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Restricted multimodal data root.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Local results root.",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=code_default,
        help="Repository root (default: parent of analysis/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RESULTS_DIR/09_released_measurement_audit).",
    )
    parser.add_argument(
        "--correlation-semantics",
        choices=["UNVERIFIED", "RAW_R", "FISHER_Z"],
        default=os.environ.get("AABC2_CORRELATION_SEMANTICS", "UNVERIFIED").strip().upper() or "UNVERIFIED",
        help="Documented scale of released FullCorrelation values.",
    )
    parser.add_argument(
        "--correlation-semantics-source",
        type=Path,
        default=(
            Path(os.environ["AABC2_CORRELATION_SEMANTICS_SOURCE"])
            if os.environ.get("AABC2_CORRELATION_SEMANTICS_SOURCE") else None
        ),
        help="Authoritative release documentation or extraction code supporting the chosen correlation scale.",
    )
    parser.add_argument(
        "--task-evoked-treatment",
        choices=["UNVERIFIED", "RETAINED", "REGRESSED", "PARTIAL_OR_UNCLEAR"],
        default="UNVERIFIED",
        help="Documented treatment of task-evoked responses in released task-state products.",
    )
    parser.add_argument(
        "--task-semantics-source",
        type=Path,
        default=None,
        help="Authoritative source supporting --task-evoked-treatment.",
    )
    parser.add_argument(
        "--diagnostic-sessions-per-state",
        type=int,
        default=DEFAULT_DIAGNOSTIC_SESSIONS_PER_STATE,
        help="Fallback deterministic sample if analysis-02 diagnostic sessions are unavailable.",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Run inventory/alignment only; correlation sensitivity will remain not ready.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def classify_state(value: object) -> Optional[str]:
    token = compact_token(value)
    if "VISMOTOR" in token:
        return "VISMOTOR"
    if "FACENAME" in token:
        return "FACENAME"
    if "CARIT" in token:
        return "CARIT"
    if "REST" in token and ("FMRI" in token or "RFMRI" in token):
        return "REST"
    return None


def identify_id_column(columns) -> str:
    columns = list(columns)
    exact = [column for column in columns if compact_token(column) == "IDEVENT"]
    if len(exact) == 1:
        return exact[0]
    accepted = {
        "IDEVENT", "SUBJECTEVENT", "SUBJECTID", "SUBJECT",
        "ID", "PARTICIPANTID", "X",
    }
    candidates = [column for column in columns if compact_token(column) in accepted]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Could not identify one event-ID column; header begins {columns[:12]}")


def pick_column(frame: pd.DataFrame, aliases, required=True, label="column") -> Optional[str]:
    lookup = {compact_token(column): column for column in frame.columns}
    for alias in aliases:
        if compact_token(alias) in lookup:
            return lookup[compact_token(alias)]
    if required:
        raise KeyError(f"Missing {label}; accepted aliases: {aliases}")
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def resolve_unique_files(data_dir: Path, pattern: str, results_dir: Path) -> dict[str, Optional[Path]]:
    resolved: dict[str, Optional[Path]] = {}
    for state in STATES:
        hits = sorted({
            path.resolve()
            for path in data_dir.rglob(pattern)
            if path.is_file()
            and classify_state(path.name) == state
            and results_dir.resolve() not in path.resolve().parents
        })
        if len(hits) > 1:
            details = "\n".join(f"  {path}" for path in hits)
            raise RuntimeError(f"Ambiguous {state} {pattern} archives:\n{details}")
        resolved[state] = hits[0] if hits else None
    return resolved


def resolve_node_manifest(data_dir: Path, results_dir: Path) -> tuple[Path, pd.DataFrame]:
    candidates = sorted({
        path.resolve()
        for path in data_dir.rglob("fmri_node_parcel_manifest.csv")
        if path.is_file() and results_dir.resolve() not in path.resolve().parents
    })
    valid = []
    for path in candidates:
        table = pd.read_csv(path, low_memory=False)
        area = pick_column(table, ["Brain Area", "brain_area", "parcel", "area"], False)
        node = pick_column(table, ["node", "network"], False)
        index = pick_column(table, ["Real Index", "real_index", "index"], False)
        if area and node and index:
            part = table[[area, node, index]].rename(
                columns={area: "Brain Area", node: "node", index: "Real Index"}
            )
            part["Real Index"] = pd.to_numeric(part["Real Index"], errors="coerce")
            # Keep all manifest rows: selection and validation happen explicitly
            # below, so malformed or noncortical entries cannot disappear silently.
            part = part.sort_values("Real Index").reset_index(drop=True)
            valid.append((path, part))
    if not valid:
        raise FileNotFoundError("Could not resolve fmri_node_parcel_manifest.csv")
    signatures = {
        hashlib.sha256(table.to_csv(index=False).encode()).hexdigest()
        for _, table in valid
    }
    if len(signatures) != 1:
        raise RuntimeError("Nonidentical node-parcel manifests were found")
    return valid[0]


def select_cortical_parcels(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the twelve canonical cortical networks, as in analysis 02.

    A released manifest may also contain entries outside these networks. Do
    not infer cortex from row position or assume the last two rows are extras.
    Keep a full selection audit and require unique parcel names and indices.
    The caller checks the selected count after saving that audit.
    """
    table = manifest[["Brain Area", "node", "Real Index"]].copy()
    if table.isna().any().any():
        raise ValueError("Parcel manifest contains missing labels, networks, or numeric indices")
    for column in ["Brain Area", "node"]:
        table[column] = table[column].astype(str).str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Parcel manifest contains empty {column} values")
    indices = pd.to_numeric(table["Real Index"], errors="coerce")
    if not np.isfinite(indices).all() or not indices.eq(np.floor(indices)).all():
        raise ValueError("Parcel manifest Real Index values must be finite integers")
    table["Real Index"] = indices.astype("int64")
    tokens = table["Brain Area"].map(compact_token)
    if tokens.eq("").any():
        raise ValueError("Parcel manifest contains an empty normalized parcel label")
    duplicate_names = table.loc[tokens.duplicated(keep=False), "Brain Area"].tolist()
    if duplicate_names:
        raise ValueError(f"Duplicate normalized parcel labels in manifest: {duplicate_names[:12]}")
    duplicate_indices = table.loc[
        table["Real Index"].duplicated(keep=False), "Real Index"
    ].tolist()
    if duplicate_indices:
        raise ValueError(f"Duplicate parcel indices in manifest: {duplicate_indices[:12]}")

    canonical_by_token = {compact_token(name): name for name in CANONICAL_NETWORKS}
    canonical = table["node"].map(compact_token).map(canonical_by_token)
    included = canonical.notna()
    audit = table.copy()
    audit["canonical_cortical_network"] = canonical.fillna("")
    audit["included_in_cortical_audit"] = included
    audit["selection_reason"] = np.where(
        included,
        "assigned to one of the twelve canonical cortical networks",
        "network outside the twelve canonical cortical networks",
    )
    selected = table.loc[included].sort_values("Real Index").reset_index(drop=True)
    return selected, audit


def load_eligible_networks(results_dir: Path) -> list[str]:
    manifest_path = results_dir / "02_primary_profile_and_measurement/run_manifest.json"
    if not manifest_path.is_file():
        return DEFAULT_ELIGIBLE_NETWORKS.copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    networks = list(manifest.get("eligible_networks", []))
    if len(networks) != 10:
        raise AssertionError(f"Expected ten eligible networks; found {networks}")
    return networks


def amplitude_member(zip_path: Path) -> str:
    with ZipFile(zip_path) as archive:
        hits = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and compact_token(Path(name).stem).endswith("FULLAMPLITUDES")
        ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one FullAmplitudes CSV in {zip_path.name}; found {len(hits)}")
    return hits[0]


def csv_members(zip_path: Path) -> list[str]:
    with ZipFile(zip_path) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".csv")]


def zip_header(zip_path: Path, member: str) -> list[str]:
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        return list(pd.read_csv(stream, nrows=0).columns)


def zip_member_for_area(zip_path: Path, area: str) -> str:
    token = compact_token(area)
    members = csv_members(zip_path)
    exact = [name for name in members if compact_token(Path(name).stem) == token]
    if len(exact) == 1:
        return exact[0]
    suffix = [name for name in members if compact_token(Path(name).stem).endswith(token)]
    if len(suffix) == 1:
        return suffix[0]
    raise RuntimeError(f"Could not uniquely map parcel {area} in {zip_path.name}")


def destination_map(columns, parcel_areas: list[str]) -> dict[str, str]:
    area_lookup = {compact_token(area): area for area in parcel_areas}
    return {
        area_lookup[compact_token(column)]: column
        for column in columns
        if compact_token(column) in area_lookup
    }


def read_amplitudes(zip_path: Path, parcel_areas: list[str]) -> pd.DataFrame:
    member = amplitude_member(zip_path)
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        header = list(pd.read_csv(stream, nrows=0).columns)
    id_column = identify_id_column(header)
    destinations = destination_map(header, parcel_areas)
    missing = [area for area in parcel_areas if area not in destinations]
    if missing:
        raise KeyError(f"{zip_path.name}/{member} lacks {len(missing)} parcels; first {missing[:5]}")
    usecols = [id_column, *[destinations[area] for area in parcel_areas]]
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        table = pd.read_csv(stream, usecols=usecols, low_memory=False)
    table = table.rename(
        columns={id_column: "id_event", **{destinations[area]: area for area in parcel_areas}}
    )
    table["id_event"] = table["id_event"].astype(str).str.strip()
    if table["id_event"].duplicated().any():
        raise AssertionError(f"Duplicate event IDs in {zip_path.name}/{member}")
    table[parcel_areas] = table[parcel_areas].apply(pd.to_numeric, errors="coerce")
    return table


def archive_event_ids(zip_path: Path, first_area: str) -> pd.Series:
    member = zip_member_for_area(zip_path, first_area)
    header = zip_header(zip_path, member)
    id_column = identify_id_column(header)
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        ids = pd.read_csv(stream, usecols=[id_column], low_memory=False)[id_column]
    ids = ids.astype(str).str.strip()
    if ids.duplicated().any():
        raise AssertionError(f"Duplicate event IDs in {zip_path.name}/{member}")
    return ids


def read_selected_rows(
    zip_path: Path,
    member: str,
    usecols: list[str],
    target_ids: list[str],
) -> pd.DataFrame:
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
    result = pd.concat(pieces, ignore_index=True)
    id_column = identify_id_column(result.columns)
    result[id_column] = result[id_column].astype(str).str.strip()
    if result[id_column].duplicated().any():
        raise AssertionError(f"Duplicate selected event IDs in {zip_path.name}/{member}")
    return result.set_index(id_column)


def sample_numeric_diagnostic(zip_path: Optional[Path], max_members=3, nrows=100) -> dict:
    if zip_path is None:
        return {
            "sample_numeric_n": 0,
            "sample_min": np.nan,
            "sample_median": np.nan,
            "sample_max": np.nan,
            "sample_fraction_abs_le_1": np.nan,
        }
    values = []
    members = csv_members(zip_path)[:max_members]
    with ZipFile(zip_path) as archive:
        for member in members:
            with archive.open(member) as stream:
                sample = pd.read_csv(stream, nrows=nrows, low_memory=False)
            try:
                id_column = identify_id_column(sample.columns)
            except Exception:
                id_column = None
            numeric = sample.drop(columns=[id_column], errors="ignore").apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float).ravel()
            numeric = numeric[np.isfinite(numeric)]
            if len(numeric):
                values.append(numeric)
    joined = np.concatenate(values) if values else np.array([], float)
    return {
        "sample_numeric_n": int(len(joined)),
        "sample_min": float(np.min(joined)) if len(joined) else np.nan,
        "sample_median": float(np.median(joined)) if len(joined) else np.nan,
        "sample_max": float(np.max(joined)) if len(joined) else np.nan,
        "sample_fraction_abs_le_1": float(np.mean(np.abs(joined) <= 1)) if len(joined) else np.nan,
    }


def basename_provenance(path: Optional[Path], members: list[str]) -> dict:
    text = " ".join([path.name if path else "", *members[:3]]).lower()
    return {
        "msmall_token_present": "msmall" in text,
        "hp0_token_present": "hp0" in text,
        "clean_token_present": "clean" in text,
        "rclean_token_present": "rclean" in text,
        "tclean_token_present": "tclean" in text,
        "provenance_from_names_only": True,
    }


def verify_semantics_source(
    semantics: str,
    source_path: Optional[Path],
    kind: str,
) -> tuple[str, str]:
    if semantics == "UNVERIFIED":
        return "UNRESOLVED", ""
    if source_path is None:
        raise AssertionError(f"{kind} semantics require an authoritative source path")
    source_path = source_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    text = source_path.read_text(errors="replace").lower()
    if kind == "correlation" and semantics == "RAW_R":
        if not re.search(r"pearson|raw\s*r|correlation coefficient", text):
            raise AssertionError("RAW_R source lacks Pearson/raw-r/correlation-coefficient language")
    if kind == "correlation" and semantics == "FISHER_Z":
        if not re.search(r"fisher|atanh|arctanh", text):
            raise AssertionError("FISHER_Z source lacks Fisher/atanh/arctanh language")
    return "VERIFIED_FROM_SUPPLIED_SOURCE", sha256_file(source_path)


def load_existing_diagnostic_ids(results_dir: Path) -> dict[str, list[str]]:
    path = (
        results_dir
        / "02_primary_profile_and_measurement/tables/gate1_normalization_diagnostics.csv"
    )
    if not path.is_file():
        return {}
    table = pd.read_csv(path, low_memory=False)
    required = {"source_table", "state", "id_event"}
    if not required.issubset(table.columns):
        return {}
    selected = table.loc[
        table["source_table"].astype(str).eq("covariance_amplitude_pair_summary.csv")
        & table["state"].isin(STATES)
        & table["id_event"].notna(),
        ["state", "id_event"],
    ].drop_duplicates()
    mapping = {
        state: selected.loc[selected["state"].eq(state), "id_event"].astype(str).tolist()
        for state in STATES
    }
    return mapping if all(mapping[state] for state in STATES) else {}


def deterministic_diagnostic_ids(
    common_ids: list[str],
    n: int,
) -> list[str]:
    if not common_ids:
        return []
    ordered = sorted(set(map(str, common_ids)))
    positions = np.linspace(0, len(ordered) - 1, min(n, len(ordered))).round().astype(int)
    return [ordered[position] for position in positions]


def resolve_node_table(data_dir: Path, results_dir: Path, state: str) -> Optional[Path]:
    patterns = [
        f"full_covariance__{state.lower()}__node_connectivity.parquet",
        f"full_covariance__{state.lower()}__node_connectivity.csv",
    ]
    hits = []
    for pattern in patterns:
        hits.extend(
            path.resolve()
            for path in data_dir.rglob(pattern)
            if path.is_file() and results_dir.resolve() not in path.resolve().parents
        )
    hits = sorted(set(hits))
    if len(hits) > 1:
        signatures = defaultdict(list)
        for path in hits:
            signatures[sha256_file(path)].append(path)
        if len(signatures) > 1:
            raise RuntimeError(f"Nonidentical node-connectivity tables found for {state}")
    return hits[0] if hits else None


def load_released_network_covariance(
    path: Optional[Path],
    id_event: str,
    networks: list[str],
) -> dict[str, float]:
    if path is None:
        return {network: np.nan for network in networks}
    table = read_table(path)
    id_column = pick_column(table, ["id_event"], True, "id_event")
    ids = table[id_column].astype(str).str.strip()
    selected = table.loc[ids.eq(str(id_event))]
    if len(selected) != 1:
        return {network: np.nan for network in networks}
    row = selected.iloc[0]
    output = {}
    for network in networks:
        column = f"fc__{slug(network)}__{slug(network)}"
        output[network] = float(pd.to_numeric(row.get(column, np.nan), errors="coerce"))
    return output


def relative_error(observed: float, expected: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(expected):
        return np.nan
    scale = max(abs(observed), abs(expected), AGGREGATION_ATOL)
    return abs(observed - expected) / scale


def quantile_value(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else np.nan


def summarize_ratio(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    absolute = np.abs(values)
    return {
        "finite_pair_n": int(len(values)),
        "ratio_min": float(np.min(values)) if len(values) else np.nan,
        "ratio_p001": quantile_value(values, .001),
        "ratio_p01": quantile_value(values, .01),
        "ratio_median": quantile_value(values, .5),
        "ratio_p99": quantile_value(values, .99),
        "ratio_p999": quantile_value(values, .999),
        "ratio_max": float(np.max(values)) if len(values) else np.nan,
        "fraction_abs_gt_1_01": float(np.mean(absolute > 1.01)) if len(values) else np.nan,
        "fraction_abs_gt_2": float(np.mean(absolute > 2)) if len(values) else np.nan,
        "fraction_abs_gt_5": float(np.mean(absolute > 5)) if len(values) else np.nan,
        "fraction_abs_gt_10": float(np.mean(absolute > 10)) if len(values) else np.nan,
    }


def pairwise_diagnostic(
    states: list[str],
    covariance_archives: dict[str, Optional[Path]],
    amplitude_tables: dict[str, pd.DataFrame],
    correlation_archives: dict[str, Optional[Path]],
    correlation_semantics: str,
    parcel_map: pd.DataFrame,
    eligible_networks: list[str],
    selected_ids: dict[str, list[str]],
    node_tables: dict[str, Optional[Path]],
) -> dict[str, pd.DataFrame]:
    parcel_areas = parcel_map.sort_values("Real Index")["Brain Area"].astype(str).tolist()
    canonical_by_token = {
        compact_token(network): network for network in CANONICAL_NETWORKS
    }
    parcel_network = {
        str(row["Brain Area"]): canonical_by_token.get(
            compact_token(row["node"]), str(row["node"])
        )
        for _, row in parcel_map.iterrows()
    }
    network_areas = {
        network: [area for area in parcel_areas if parcel_network[area] == network]
        for network in eligible_networks
    }
    session_labels = {
        (state, id_event): f"{state}_{index:02d}"
        for state in states
        for index, id_event in enumerate(selected_ids[state], 1)
    }

    ratio_values = defaultdict(list)
    denominator_values = defaultdict(list)
    correlation_values = defaultdict(list)
    network_covariance = defaultdict(list)
    network_denominator = defaultdict(list)
    network_ratio = defaultdict(list)
    network_correlation = defaultdict(list)
    top_extremes = defaultdict(list)
    self_rows = []

    for state in states:
        covariance_archive = covariance_archives[state]
        if covariance_archive is None:
            raise FileNotFoundError(f"Missing {state} FullCovarianceConnectivity archive")
        correlation_archive = correlation_archives.get(state)
        use_correlation = (
            correlation_archive is not None
            and correlation_semantics in {"RAW_R", "FISHER_Z"}
        )
        amplitudes = amplitude_tables[state].set_index("id_event")

        for source_index, source_area in enumerate(
            tqdm(parcel_areas, desc=f"{state} released-product audit", leave=False)
        ):
            later = parcel_areas[source_index + 1:]
            covariance_member = zip_member_for_area(covariance_archive, source_area)
            covariance_header = zip_header(covariance_archive, covariance_member)
            covariance_id = identify_id_column(covariance_header)
            covariance_destinations = destination_map(covariance_header, parcel_areas)
            required_areas = [source_area, *later]
            missing = [area for area in required_areas if area not in covariance_destinations]
            if missing:
                raise KeyError(
                    f"{state}/{covariance_member} lacks {len(missing)} destinations; first {missing[:5]}"
                )
            covariance_usecols = [
                covariance_id,
                *[covariance_destinations[area] for area in required_areas],
            ]
            covariance_rows = read_selected_rows(
                covariance_archive,
                covariance_member,
                covariance_usecols,
                selected_ids[state],
            )

            correlation_rows = None
            correlation_destinations = None
            if use_correlation:
                correlation_member = zip_member_for_area(correlation_archive, source_area)
                correlation_header = zip_header(correlation_archive, correlation_member)
                correlation_id = identify_id_column(correlation_header)
                correlation_destinations = destination_map(correlation_header, parcel_areas)
                missing_corr = [area for area in required_areas if area not in correlation_destinations]
                if missing_corr:
                    raise KeyError(
                        f"{state}/{correlation_member} lacks {len(missing_corr)} destinations; first {missing_corr[:5]}"
                    )
                correlation_usecols = [
                    correlation_id,
                    *[correlation_destinations[area] for area in required_areas],
                ]
                correlation_rows = read_selected_rows(
                    correlation_archive,
                    correlation_member,
                    correlation_usecols,
                    selected_ids[state],
                )

            for id_event in selected_ids[state]:
                if id_event not in covariance_rows.index or id_event not in amplitudes.index:
                    raise RuntimeError(f"Missing selected observation {state}/{id_event}")
                covariance_row = covariance_rows.loc[id_event]
                if isinstance(covariance_row, pd.DataFrame):
                    raise AssertionError(f"Duplicate selected covariance row: {state}/{id_event}")
                amplitude_i = float(amplitudes.loc[id_event, source_area])
                self_covariance = float(
                    pd.to_numeric(
                        covariance_row[covariance_destinations[source_area]], errors="coerce"
                    )
                )
                self_rows.append({
                    "state": state,
                    "diagnostic_session": session_labels[(state, id_event)],
                    "parcel": source_area,
                    "covariance_self_entry": self_covariance,
                    "full_amplitude": amplitude_i,
                    "full_amplitude_squared": amplitude_i ** 2,
                    "relative_error_vs_amplitude": relative_error(self_covariance, amplitude_i),
                    "relative_error_vs_amplitude_squared": relative_error(
                        self_covariance, amplitude_i ** 2
                    ),
                })

                correlation_row = None
                if use_correlation:
                    if id_event not in correlation_rows.index:
                        raise RuntimeError(f"Missing selected correlation row {state}/{id_event}")
                    correlation_row = correlation_rows.loc[id_event]
                    if isinstance(correlation_row, pd.DataFrame):
                        raise AssertionError(f"Duplicate selected correlation row: {state}/{id_event}")

                label = session_labels[(state, id_event)]
                source_network = parcel_network[source_area]
                for destination_area in later:
                    covariance = float(
                        pd.to_numeric(
                            covariance_row[covariance_destinations[destination_area]],
                            errors="coerce",
                        )
                    )
                    amplitude_j = float(amplitudes.loc[id_event, destination_area])
                    denominator = amplitude_i * amplitude_j
                    ratio = (
                        covariance / denominator
                        if np.isfinite(covariance)
                        and np.isfinite(denominator)
                        and denominator != 0
                        else np.nan
                    )
                    session_key = (state, label)
                    if np.isfinite(ratio):
                        ratio_values[session_key].append(ratio)
                        denominator_values[session_key].append(denominator)
                        if abs(ratio) > CORRELATION_BOUND:
                            top_extremes[session_key].append(
                                {
                                    "state": state,
                                    "diagnostic_session": label,
                                    "source_parcel": source_area,
                                    "destination_parcel": destination_area,
                                    "amplitude_product": denominator,
                                    "ratio": ratio,
                                    "absolute_ratio": abs(ratio),
                                }
                            )

                    destination_network = parcel_network[destination_area]
                    if source_network == destination_network and source_network in eligible_networks:
                        network_key = (state, label, source_network)
                        if np.isfinite(covariance):
                            network_covariance[network_key].append(covariance)
                        if np.isfinite(denominator):
                            network_denominator[network_key].append(denominator)
                        if np.isfinite(ratio):
                            network_ratio[network_key].append(ratio)

                    if use_correlation:
                        released_value = float(
                            pd.to_numeric(
                                correlation_row[correlation_destinations[destination_area]],
                                errors="coerce",
                            )
                        )
                        released_r = (
                            math.tanh(released_value)
                            if correlation_semantics == "FISHER_Z" and np.isfinite(released_value)
                            else released_value
                        )
                        if np.isfinite(ratio) and np.isfinite(released_r):
                            correlation_values[session_key].append((ratio, released_r, denominator))
                        if (
                            source_network == destination_network
                            and source_network in eligible_networks
                            and np.isfinite(released_r)
                        ):
                            network_correlation[(state, label, source_network)].append(released_r)

    session_rows = []
    denominator_quartile_rows = []
    pairwise_agreement_rows = []
    for session_key, ratios_list in ratio_values.items():
        state, label = session_key
        ratios = np.asarray(ratios_list, float)
        denominators = np.asarray(denominator_values[session_key], float)
        row = {"state": state, "diagnostic_session": label, **summarize_ratio(ratios)}
        row.update({
            "denominator_min": float(np.min(denominators)),
            "denominator_median": float(np.median(denominators)),
            "denominator_max": float(np.max(denominators)),
            "spearman_abs_ratio_vs_inverse_denominator": pd.Series(np.abs(ratios)).corr(
                pd.Series(1.0 / denominators), method="spearman"
            ),
        })
        session_rows.append(row)

        quartiles = pd.qcut(
            pd.Series(denominators), q=4, labels=False, duplicates="drop"
        ).to_numpy()
        for quartile in sorted(set(quartiles[np.isfinite(quartiles)])):
            keep = quartiles == quartile
            denominator_quartile_rows.append({
                "state": state,
                "diagnostic_session": label,
                "denominator_quartile": int(quartile) + 1,
                "pair_n": int(keep.sum()),
                "denominator_median": float(np.median(denominators[keep])),
                "fraction_abs_ratio_gt_1_01": float(
                    np.mean(np.abs(ratios[keep]) > CORRELATION_BOUND)
                ),
            })

        if session_key in correlation_values:
            values = np.asarray(correlation_values[session_key], float)
            pairwise_agreement_rows.append({
                "state": state,
                "diagnostic_session": label,
                "pair_n": int(len(values)),
                "pearson_ratio_vs_released_r": pd.Series(values[:, 0]).corr(
                    pd.Series(values[:, 1]), method="pearson"
                ),
                "spearman_ratio_vs_released_r": pd.Series(values[:, 0]).corr(
                    pd.Series(values[:, 1]), method="spearman"
                ),
                "median_ratio_minus_released_r": float(
                    np.median(values[:, 0] - values[:, 1])
                ),
                "released_r_min": float(np.min(values[:, 1])),
                "released_r_max": float(np.max(values[:, 1])),
                "released_r_fraction_abs_gt_1_01": float(
                    np.mean(np.abs(values[:, 1]) > CORRELATION_BOUND)
                ),
            })

    network_rows = []
    reproduction_rows = []
    for state in states:
        state_node_table = node_tables[state]
        for id_event in selected_ids[state]:
            label = session_labels[(state, id_event)]
            released_network = load_released_network_covariance(
                state_node_table, id_event, eligible_networks
            )
            amplitude_row = amplitude_tables[state].set_index("id_event").loc[id_event]
            for network in eligible_networks:
                key = (state, label, network)
                covariance = np.asarray(network_covariance[key], float)
                denominator = np.asarray(network_denominator[key], float)
                ratios = np.asarray(network_ratio[key], float)
                correlations = np.asarray(network_correlation[key], float)
                covariance_mean = float(np.mean(covariance)) if len(covariance) else np.nan
                pair_amplitude_mean = float(np.mean(denominator)) if len(denominator) else np.nan
                standardized_ratio_of_means = (
                    covariance_mean / pair_amplitude_mean
                    if pair_amplitude_mean > 0 else np.nan
                )
                network_amplitudes = pd.to_numeric(
                    amplitude_row[network_areas[network]], errors="coerce"
                ).to_numpy(float)
                network_amplitudes = network_amplitudes[np.isfinite(network_amplitudes)]
                amplitude_n = len(network_amplitudes)
                pair_amplitude_formula = (
                    (
                        float(np.sum(network_amplitudes)) ** 2
                        - float(np.sum(network_amplitudes ** 2))
                    )
                    / (amplitude_n * (amplitude_n - 1))
                    if amplitude_n >= 2 else np.nan
                )
                network_rows.append({
                    "state": state,
                    "diagnostic_session": label,
                    "network": network,
                    "pair_n": int(len(covariance)),
                    "covariance_pair_mean": covariance_mean,
                    "pair_amplitude_mean": pair_amplitude_mean,
                    "pair_amplitude_formula": pair_amplitude_formula,
                    "pair_amplitude_formula_relative_difference": relative_error(
                        pair_amplitude_mean, pair_amplitude_formula
                    ),
                    "ratio_of_network_means": standardized_ratio_of_means,
                    "equal_pair_mean_of_ratios_diagnostic": (
                        float(np.mean(ratios)) if len(ratios) else np.nan
                    ),
                    "released_r_pair_mean": (
                        float(np.mean(correlations)) if len(correlations) else np.nan
                    ),
                })
                released_value = released_network[network]
                reproduction_rows.append({
                    "state": state,
                    "diagnostic_session": label,
                    "network": network,
                    "recomputed_covariance_pair_mean": covariance_mean,
                    "released_node_covariance": released_value,
                    "absolute_difference": (
                        abs(covariance_mean - released_value)
                        if np.isfinite(covariance_mean) and np.isfinite(released_value)
                        else np.nan
                    ),
                    "relative_difference": relative_error(covariance_mean, released_value),
                    "reproduction_available": bool(np.isfinite(released_value)),
                })

    network_summary = pd.DataFrame(network_rows)
    network_state_rows = []
    for (state, network), group in network_summary.groupby(["state", "network"]):
        network_state_rows.append({
            "state": state,
            "network": network,
            "diagnostic_session_n": int(group["diagnostic_session"].nunique()),
            "pair_n_min": int(group["pair_n"].min()),
            "ratio_of_network_means_min": float(group["ratio_of_network_means"].min()),
            "ratio_of_network_means_median": float(group["ratio_of_network_means"].median()),
            "ratio_of_network_means_max": float(group["ratio_of_network_means"].max()),
            "equal_pair_mean_of_ratios_median": float(
                group["equal_pair_mean_of_ratios_diagnostic"].median()
            ),
            "released_r_pair_mean_median": float(group["released_r_pair_mean"].median())
            if group["released_r_pair_mean"].notna().any() else np.nan,
        })

    self_table = pd.DataFrame(self_rows)
    self_summary = (
        self_table.groupby(["state", "diagnostic_session"], as_index=False)
        .agg(
            parcel_n=("parcel", "size"),
            median_relative_error_vs_amplitude=("relative_error_vs_amplitude", "median"),
            median_relative_error_vs_amplitude_squared=(
                "relative_error_vs_amplitude_squared", "median"
            ),
        )
    )
    self_summary["closer_literal_identity"] = np.where(
        self_summary["median_relative_error_vs_amplitude_squared"]
        < self_summary["median_relative_error_vs_amplitude"],
        "SELF_CLOSER_TO_AMPLITUDE_SQUARED",
        "SELF_CLOSER_TO_AMPLITUDE",
    )

    extreme_rows = []
    for records in top_extremes.values():
        extreme_rows.extend(
            sorted(records, key=lambda record: record["absolute_ratio"], reverse=True)[:10]
        )

    return {
        "normalization_by_session": pd.DataFrame(session_rows),
        "normalization_by_network_session": network_summary,
        "normalization_by_network_state": pd.DataFrame(network_state_rows),
        "normalization_by_denominator_quartile": pd.DataFrame(denominator_quartile_rows),
        "normalization_top_extremes": pd.DataFrame(extreme_rows),
        "covariance_self_summary": self_summary,
        "pairwise_ratio_vs_released_correlation": pd.DataFrame(pairwise_agreement_rows),
        "network_aggregate_reproduction": pd.DataFrame(reproduction_rows),
    }


def add_gate(rows: list[dict], name: str, status: str, severity: str, detail: str) -> None:
    rows.append({
        "gate": name,
        "status": status,
        "severity": severity,
        "detail": detail,
    })


def write_figures(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    written = []
    by_session = tables.get("normalization_by_session", pd.DataFrame())
    if len(by_session):
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        labels = by_session["diagnostic_session"].astype(str).tolist()
        values = 100 * by_session["fraction_abs_gt_1_01"].to_numpy(float)
        colors = ["#6E7781" if not label.startswith("REST") else "#3B5B92" for label in labels]
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Parcel pairs with |ratio| > 1.01 (%)")
        ax.set_xlabel("Fixed diagnostic session")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            path = figure_dir / f"normalization_violation_fraction.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    quartiles = tables.get("normalization_by_denominator_quartile", pd.DataFrame())
    if len(quartiles):
        summary = quartiles.groupby("denominator_quartile", as_index=False).agg(
            median_fraction=("fraction_abs_ratio_gt_1_01", "median"),
            q25=("fraction_abs_ratio_gt_1_01", lambda x: x.quantile(.25)),
            q75=("fraction_abs_ratio_gt_1_01", lambda x: x.quantile(.75)),
        )
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
        x = summary["denominator_quartile"].to_numpy(float)
        y = 100 * summary["median_fraction"].to_numpy(float)
        lower = 100 * (summary["median_fraction"] - summary["q25"]).to_numpy(float)
        upper = 100 * (summary["q75"] - summary["median_fraction"]).to_numpy(float)
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", color="#3B5B92")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Amplitude-product quartile (low to high)")
        ax.set_ylabel("Median |ratio| > 1.01 (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        for extension in ["pdf", "png"]:
            path = figure_dir / f"normalization_by_denominator_quartile.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if frame.empty:
        return "_No rows._"
    display_frame = frame.copy()
    for column in display_frame.columns:
        display_frame[column] = display_frame[column].map(
            lambda value: ""
            if pd.isna(value)
            else str(value).replace("|", "\\|").replace("\n", " ")
        )
    headers = [str(column).replace("|", "\\|") for column in display_frame.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for values in display_frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(rows)


def write_report(
    path: Path,
    summary: dict,
    gates: pd.DataFrame,
    product_inventory: pd.DataFrame,
    key_alignment: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# Released measurement audit",
        "",
        f"Created: {summary['created_utc']}",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        summary["decision_reason"],
        "",
        "## Product inventory",
        "",
        dataframe_markdown(product_inventory),
        "",
        "## Observation alignment",
        "",
        dataframe_markdown(key_alignment),
        "",
        "## Audit gates",
        "",
        dataframe_markdown(gates),
        "",
    ]
    diagnostic = tables.get("normalization_by_session", pd.DataFrame())
    if len(diagnostic):
        columns = [
            "state", "diagnostic_session", "finite_pair_n",
            "fraction_abs_gt_1_01", "ratio_min", "ratio_max",
            "spearman_abs_ratio_vs_inverse_denominator",
        ]
        lines.extend([
            "## Pairwise normalization diagnostic",
            "",
            "The values below describe a fixed diagnostic sample and are not cohort-wide prevalence estimates.",
            "",
            dataframe_markdown(diagnostic[columns]),
            "",
        ])
    agreement = tables.get("pairwise_ratio_vs_released_correlation", pd.DataFrame())
    if len(agreement):
        lines.extend([
            "## Ratio versus released correlation",
            "",
            dataframe_markdown(agreement),
            "",
        ])
    lines.extend([
        "## Interpretation",
        "",
        "- Archive names and member names provide provenance clues only; authoritative preprocessing and task-regression statements require release documentation.",
        "- The parcel-pair covariance/amplitude ratio is diagnostic and is not treated as Pearson correlation.",
        "- Released-correlation comparisons require documented scale and passing alignment checks.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    code_dir = args.code_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else results_dir / "09_released_measurement_audit"
    )
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in [output_dir, table_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Data directory not found: {data_dir}\n"
            "Pass --data-dir if multimodal is not beside four_state_fmri_code."
        )
    if not results_dir.is_dir():
        raise NotADirectoryError(
            f"Results directory not found: {results_dir}\n"
            "Pass --results-dir if four_state_fmri_results is not beside the code directory."
        )
    if not code_dir.is_dir():
        raise NotADirectoryError(code_dir)
    if args.diagnostic_sessions_per_state < 1:
        raise ValueError("--diagnostic-sessions-per-state must be positive")

    correlation_source_status, correlation_source_sha = verify_semantics_source(
        args.correlation_semantics,
        args.correlation_semantics_source,
        "correlation",
    )
    task_source_status, task_source_sha = verify_semantics_source(
        args.task_evoked_treatment,
        args.task_semantics_source,
        "task",
    )

    node_manifest_path, full_parcel_map = resolve_node_manifest(data_dir, results_dir)
    parcel_map, parcel_selection = select_cortical_parcels(full_parcel_map)
    selection_path = table_dir / "02a_manifest_parcel_selection.csv"
    parcel_selection.to_csv(selection_path, index=False)
    excluded = parcel_selection.loc[
        ~parcel_selection["included_in_cortical_audit"], ["Brain Area", "node"]
    ]
    print(
        f"Parcel manifest: {len(full_parcel_map)} rows; "
        f"{len(parcel_map)} cortical parcels selected; {len(excluded)} entries excluded."
    )
    if len(excluded):
        print(excluded.to_string(index=False))
    if len(parcel_map) != EXPECTED_PARCELS:
        raise AssertionError(
            f"Expected {EXPECTED_PARCELS} unique cortical parcels assigned to the twelve "
            f"canonical networks; found {len(parcel_map)} among {len(full_parcel_map)} "
            f"manifest rows. Inspect {selection_path}. Do not change EXPECTED_PARCELS "
            "or truncate the manifest; check the parcel labels and network assignments."
        )
    parcel_areas = parcel_map.sort_values("Real Index")["Brain Area"].astype(str).tolist()
    eligible_networks = load_eligible_networks(results_dir)

    amplitude_archives = resolve_unique_files(data_dir, "*fMRIAmplitudes.zip", results_dir)
    covariance_archives = resolve_unique_files(
        data_dir, "*FullCovarianceConnectivity.zip", results_dir
    )
    correlation_archives = resolve_unique_files(
        data_dir, "*FullCorrelationConnectivity.zip", results_dir
    )
    archive_groups = {
        "full_amplitude": amplitude_archives,
        "full_covariance": covariance_archives,
        "full_correlation": correlation_archives,
    }

    inventory_rows = []
    provenance_rows = []
    numeric_rows = []
    for product, mapping in archive_groups.items():
        for state in STATES:
            path = mapping[state]
            members = csv_members(path) if path else []
            inventory_rows.append({
                "product": product,
                "state": state,
                "archive_status": "PRESENT_UNIQUE" if path else "ABSENT",
                "archive_name": path.name if path else "",
                "csv_member_n": len(members),
                "archive_size_bytes": path.stat().st_size if path else np.nan,
            })
            provenance_rows.append({
                "product": product,
                "state": state,
                "archive_name": path.name if path else "",
                **basename_provenance(path, members),
            })
            numeric_rows.append({
                "product": product,
                "state": state,
                **sample_numeric_diagnostic(path),
            })
    product_inventory = pd.DataFrame(inventory_rows)
    preprocessing_provenance = pd.DataFrame(provenance_rows)
    numeric_properties = pd.DataFrame(numeric_rows)

    amplitude_tables: dict[str, pd.DataFrame] = {}
    key_rows = []
    parcel_rows = []
    node_tables = {state: resolve_node_table(data_dir, results_dir, state) for state in STATES}
    existing_ids = load_existing_diagnostic_ids(results_dir)
    selected_ids: dict[str, list[str]] = {}

    for state in STATES:
        amplitude_path = amplitude_archives[state]
        covariance_path = covariance_archives[state]
        correlation_path = correlation_archives[state]
        if amplitude_path is None or covariance_path is None:
            continue
        amplitude = read_amplitudes(amplitude_path, parcel_areas)
        amplitude_tables[state] = amplitude
        covariance_ids = archive_event_ids(covariance_path, parcel_areas[0])
        correlation_ids = (
            archive_event_ids(correlation_path, parcel_areas[0])
            if correlation_path else pd.Series(dtype="string")
        )
        amplitude_ids = amplitude["id_event"].astype(str)
        amp_set, cov_set, corr_set = set(amplitude_ids), set(covariance_ids), set(correlation_ids)
        key_rows.append({
            "state": state,
            "amplitude_event_n": len(amp_set),
            "covariance_event_n": len(cov_set),
            "correlation_event_n": len(corr_set),
            "amplitude_covariance_intersection_n": len(amp_set & cov_set),
            "amplitude_only_n": len(amp_set - cov_set),
            "covariance_only_n": len(cov_set - amp_set),
            "all_three_intersection_n": len(amp_set & cov_set & corr_set) if corr_set else 0,
        })

        for product, archive_path in [
            ("full_covariance", covariance_path),
            ("full_correlation", correlation_path),
        ]:
            if archive_path is None:
                parcel_rows.append({
                    "state": state,
                    "product": product,
                    "source_parcel_members_mapped": 0,
                    "all_360_sources_mapped": False,
                    "destination_headers_with_360_parcels": 0,
                    "status": "ABSENT",
                })
                continue
            mapped = 0
            full_headers = 0
            for area in parcel_areas:
                member = zip_member_for_area(archive_path, area)
                mapped += 1
                header = zip_header(archive_path, member)
                destinations = destination_map(header, parcel_areas)
                full_headers += int(len(destinations) == EXPECTED_PARCELS)
            parcel_rows.append({
                "state": state,
                "product": product,
                "source_parcel_members_mapped": mapped,
                "all_360_sources_mapped": mapped == EXPECTED_PARCELS,
                "destination_headers_with_360_parcels": full_headers,
                "status": "PASS" if mapped == EXPECTED_PARCELS and full_headers == EXPECTED_PARCELS else "FAIL",
            })

        common = sorted(amp_set & cov_set)
        if existing_ids and all(value in common for value in existing_ids[state]):
            selected_ids[state] = existing_ids[state]
        else:
            selected_ids[state] = deterministic_diagnostic_ids(
                common, args.diagnostic_sessions_per_state
            )

    key_alignment = pd.DataFrame(key_rows)
    parcel_alignment = pd.DataFrame(parcel_rows)

    documentation = pd.DataFrame([
        {
            "question": "correlation scale",
            "status": correlation_source_status,
            "value": args.correlation_semantics,
            "source_sha256": correlation_source_sha,
        },
        {
            "question": "task-evoked response treatment",
            "status": task_source_status,
            "value": args.task_evoked_treatment,
            "source_sha256": task_source_sha,
        },
        {
            "question": "preprocessing branch",
            "status": "NAMES_ONLY__AUTHORITATIVE_DOCUMENTATION_REQUIRED",
            "value": "MSMAll/hp0/clean/rclean/tclean tokens inventoried from archive/member names",
            "source_sha256": "",
        },
    ])

    audit_tables: dict[str, pd.DataFrame] = {}
    if not args.skip_pairwise:
        missing_selected = [state for state in STATES if not selected_ids.get(state)]
        if missing_selected:
            raise RuntimeError(f"No diagnostic sessions available for {missing_selected}")
        audit_tables = pairwise_diagnostic(
            STATES,
            covariance_archives,
            amplitude_tables,
            correlation_archives,
            args.correlation_semantics,
            parcel_map,
            eligible_networks,
            selected_ids,
            node_tables,
        )

    gates = []
    amp_cov_present = all(
        amplitude_archives[state] is not None and covariance_archives[state] is not None
        for state in STATES
    )
    add_gate(
        gates,
        "four amplitude and covariance archives present uniquely",
        "PASS" if amp_cov_present else "FAIL",
        "CRITICAL",
        f"states present: {sum(amplitude_archives[s] is not None and covariance_archives[s] is not None for s in STATES)}/4",
    )
    add_gate(
        gates,
        "360 unique cortical parcels selected from manifest",
        "PASS" if len(parcel_map) == EXPECTED_PARCELS else "FAIL",
        "CRITICAL",
        f"manifest rows: {len(full_parcel_map)}; cortical parcels: {len(parcel_map)}; "
        f"entries outside canonical cortical networks: {len(excluded)}",
    )
    parcel_pass = bool(
        len(parcel_alignment)
        and parcel_alignment.loc[parcel_alignment["product"].eq("full_covariance"), "status"].eq("PASS").all()
    )
    add_gate(
        gates,
        "covariance parcel members and destination headers align",
        "PASS" if parcel_pass else "FAIL",
        "CRITICAL",
        "all source members and destination headers must map to the frozen 360 parcels",
    )
    key_pass = bool(
        len(key_alignment)
        and key_alignment["amplitude_covariance_intersection_n"].gt(0).all()
    )
    add_gate(
        gates,
        "amplitude and covariance observation overlap",
        "PASS" if key_pass else "FAIL",
        "CRITICAL",
        "positive intersection required in every state",
    )

    correlation_present = all(correlation_archives[state] is not None for state in STATES)
    add_gate(
        gates,
        "four released correlation archives present uniquely",
        "PASS" if correlation_present else "UNRESOLVED",
        "REQUIRED_FOR_CORRELATION_SENSITIVITY",
        f"states present: {sum(correlation_archives[s] is not None for s in STATES)}/4",
    )
    add_gate(
        gates,
        "released correlation scale documented",
        "PASS" if correlation_source_status.startswith("VERIFIED") else "UNRESOLVED",
        "REQUIRED_FOR_CORRELATION_SENSITIVITY",
        args.correlation_semantics,
    )
    add_gate(
        gates,
        "task-evoked response treatment documented",
        "PASS" if task_source_status.startswith("VERIFIED") else "UNRESOLVED",
        "INTERPRETIVE",
        args.task_evoked_treatment,
    )

    if audit_tables:
        reproduction = audit_tables["network_aggregate_reproduction"]
        available = reproduction.loc[reproduction["reproduction_available"]]
        reproduction_max = (
            float(available["relative_difference"].max()) if len(available) else np.nan
        )
        reproduction_pass = bool(
            len(available)
            and np.isfinite(reproduction_max)
            and reproduction_max <= AGGREGATION_RTOL
        )
        add_gate(
            gates,
            "raw covariance reproduces released within-network covariance",
            "PASS" if reproduction_pass else "FAIL",
            "CRITICAL",
            f"maximum relative difference: {reproduction_max:.6g}" if np.isfinite(reproduction_max) else "comparison unavailable",
        )
        network_summary = audit_tables["normalization_by_network_session"]
        amplitude_formula_max = float(
            network_summary["pair_amplitude_formula_relative_difference"].max()
        )
        amplitude_formula_pass = bool(
            np.isfinite(amplitude_formula_max)
            and amplitude_formula_max <= AGGREGATION_RTOL
        )
        add_gate(
            gates,
            "explicit amplitude-product mean matches algebraic pair-amplitude formula",
            "PASS" if amplitude_formula_pass else "FAIL",
            "CRITICAL",
            f"maximum relative difference: {amplitude_formula_max:.6g}",
        )
        diagnostic = audit_tables["normalization_by_session"]
        violation_median = float(diagnostic["fraction_abs_gt_1_01"].median())
        add_gate(
            gates,
            "pairwise normalized values behave as bounded correlations",
            "WARN" if violation_median > .01 else "PASS",
            "DIAGNOSTIC",
            f"median diagnostic-session fraction outside [-1.01,1.01]: {violation_median:.4f}",
        )
        agreement = audit_tables["pairwise_ratio_vs_released_correlation"]
        add_gate(
            gates,
            "pairwise comparison with released correlation completed",
            "PASS" if len(agreement) else "UNRESOLVED",
            "DIAGNOSTIC",
            f"diagnostic sessions compared: {len(agreement)}",
        )
    else:
        reproduction_pass = False
        reproduction_max = np.nan
        amplitude_formula_pass = False
        amplitude_formula_max = np.nan
        violation_median = np.nan
        add_gate(
            gates,
            "pairwise and aggregation audit completed",
            "UNRESOLVED",
            "CRITICAL",
            "skipped by --skip-pairwise",
        )

    gate_table = pd.DataFrame(gates)
    critical_fail = bool(
        ((gate_table["severity"] == "CRITICAL") & (gate_table["status"] != "PASS")).any()
    )
    ready_for_correlation = bool(
        not critical_fail
        and correlation_present
        and correlation_source_status.startswith("VERIFIED")
        and not args.skip_pairwise
    )
    if critical_fail:
        decision = "NOT_READY__CRITICAL_AUDIT_FAILURE"
        decision_reason = "At least one critical alignment or reproduction gate failed."
    elif ready_for_correlation:
        decision = "READY_WITH_QUALIFICATIONS"
        decision_reason = (
            "The released-correlation longitudinal sensitivity is technically defined. "
            "Pairwise normalization remains a diagnostic and must not be called Pearson correlation."
        )
    else:
        decision = "NOT_READY_FOR_CORRELATION_SENSITIVITY"
        decision_reason = (
            "Primary covariance/amplitude products passed available critical checks, but released-correlation "
            "availability, semantics, or the full pairwise audit remains unresolved."
        )

    tables = {
        "01_product_inventory.csv": product_inventory,
        "02_key_alignment.csv": key_alignment,
        "03_parcel_alignment.csv": parcel_alignment,
        "02a_manifest_parcel_selection.csv": parcel_selection,
        "04_numeric_properties.csv": numeric_properties,
        "05_preprocessing_provenance.csv": preprocessing_provenance,
        "06_documentation_status.csv": documentation,
        **{
            f"{index:02d}_{name}.csv": table
            for index, (name, table) in enumerate(audit_tables.items(), start=7)
        },
        "99_audit_gates.csv": gate_table,
    }
    written_paths = []
    for filename, table in tables.items():
        path = table_dir / filename
        table.to_csv(path, index=False)
        written_paths.append(path)

    figure_paths = write_figures(audit_tables, figure_dir)
    written_paths.extend(figure_paths)

    contract = {
        "created_utc": utc_now(),
        "script": Path(__file__).name,
        "analysis": "released fMRI measurement product audit",
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "states": STATES,
        "eligible_networks": eligible_networks,
        "expected_parcels": EXPECTED_PARCELS,
        "manifest_rows": len(full_parcel_map),
        "selected_cortical_parcels": len(parcel_map),
        "excluded_manifest_entries": len(excluded),
        "cortical_parcel_selection": "membership in all twelve canonical cortical networks",
        "correlation_semantics": args.correlation_semantics,
        "task_evoked_treatment": args.task_evoked_treatment,
        "diagnostic_sessions_per_state": {
            state: len(selected_ids.get(state, [])) for state in STATES
        },
        "diagnostic_selection": (
            "reused analysis-02 fixed diagnostic sessions"
            if existing_ids else "deterministic evenly spaced overlap IDs"
        ),
        "identifiers_exported": False,
        "longitudinal_models_fitted": False,
        "source_files": {
            "node_manifest": {
                "name": node_manifest_path.name,
                "sha256": sha256_file(node_manifest_path),
            },
            "correlation_semantics_source_sha256": correlation_source_sha,
            "task_semantics_source_sha256": task_source_sha,
        },
    }
    contract_path = output_dir / "00_analysis_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    written_paths.append(contract_path)

    summary = {
        "created_utc": utc_now(),
        "decision": decision,
        "decision_reason": decision_reason,
        "critical_failure": critical_fail,
        "ready_for_correlation_sensitivity": ready_for_correlation,
        "correlation_archives_present_all_states": correlation_present,
        "correlation_semantics": args.correlation_semantics,
        "correlation_semantics_status": correlation_source_status,
        "task_evoked_treatment": args.task_evoked_treatment,
        "task_semantics_status": task_source_status,
        "network_aggregate_reproduction_max_relative_difference": reproduction_max,
        "pair_amplitude_formula_max_relative_difference": amplitude_formula_max,
        "median_diagnostic_fraction_abs_ratio_gt_1_01": violation_median,
        "diagnostic_sample_is_cohort_prevalence_estimate": False,
        "pairwise_ratio_called_correlation": False,
    }
    summary_path = output_dir / "measurement_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written_paths.append(summary_path)

    report_path = output_dir / "measurement_audit_report.md"
    write_report(
        report_path,
        summary,
        gate_table,
        product_inventory,
        key_alignment,
        audit_tables,
    )
    written_paths.append(report_path)

    manifest = {
        "created_utc": utc_now(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "decision": decision,
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in sorted(written_paths)
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(gate_table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
