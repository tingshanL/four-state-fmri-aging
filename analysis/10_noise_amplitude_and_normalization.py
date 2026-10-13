#!/usr/bin/env python3
"""Reconstruct candidate total amplitude and compare matched longitudinal normalizations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
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
import scipy
from scipy.linalg import helmert, qr
from scipy.optimize import minimize
from scipy.stats import chi2, norm, spearmanr

STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
CANONICAL_NETWORKS = ["Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
 "Dorsal-attention", "Language", "Frontoparietal", "Auditory", "Default",
 "Posterior-Multimodal", "Ventral-Multimodal", "Orbito-Affective"]
DEFAULT_ELIGIBLE_NETWORKS = CANONICAL_NETWORKS[:10]
EXPECTED_PARCELS = 360
COMPONENTS = ["SIGNAL", "ATTENUATION_H", "TOTAL"]
VERSION = "1.0.0"
IDENTITY_TOL = 1e-9
REPRO_TOL = 1e-6
COLORS = {"SIGNAL":"#0072B2", "ATTENUATION_H":"#D55E00", "TOTAL":"#009E73"}


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
    return pd.read_csv(path, dtype=str, low_memory=False)


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


def csv_members(zip_path: Path) -> list[str]:
    with ZipFile(zip_path) as archive:
        return [name for name in archive.namelist() if name.lower().endswith(".csv")
                and "__MACOSX" not in Path(name).parts and not Path(name).name.startswith("._")]


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


def archive_event_ids(zip_path: Path, first_area: str) -> pd.Series:
    member = zip_member_for_area(zip_path, first_area)
    header = zip_header(zip_path, member)
    id_column = identify_id_column(header)
    with ZipFile(zip_path) as archive, archive.open(member) as stream:
        ids = pd.read_csv(stream, usecols=[id_column], dtype={id_column:str}, low_memory=False)[id_column]
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
        for chunk in pd.read_csv(stream, usecols=usecols, dtype=str, chunksize=512, low_memory=False):
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

# Reference model definitions from analysis 02 (no top-level execution).
def retain_two_sessions(frame):
    counts = frame[["id", "id_event"]].drop_duplicates().groupby("id")["id_event"].nunique()
    return frame.loc[frame["id"].isin(counts[counts.ge(2)].index)].copy()


def cohort_row(label, frame):
    visits = frame[["id", "id_event"]].drop_duplicates()
    counts = visits.groupby("id")["id_event"].nunique() if len(visits) else pd.Series(dtype=float)
    return {"cohort": label, "participants": int(frame["id"].nunique()), "sessions": int(frame["id_event"].nunique()),
            "session_state_rows": len(frame), "participants_2plus": int((counts >= 2).sum()),
            "participants_3plus": int((counts >= 3).sum())}


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

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--stage', choices=['all', 'diagnostic', 'longitudinal'], default='all')
    p.add_argument('--subject-table', type=Path, help='Normally reuse the exact subject table recorded by analysis 02.')
    p.add_argument('--diagnostic-sessions-per-state', type=int, default=3,
                   help='Fallback sample size only; frozen analysis-02 diagnostic IDs take precedence.')
    p.add_argument('--force-diagnostic', action='store_true', help='Ignore private diagnostic matrix checkpoints.')
    p.add_argument('--force-refit', action='store_true', help='Ignore private REML reference-fit checkpoints.')
    p.add_argument('--amplitude-units', choices=['UNVERIFIED','SD'], default='UNVERIFIED',
                   help='SD requires an authoritative source. Reconstruction still runs conditionally when unverified.')
    p.add_argument('--noise-decomposition', choices=['UNVERIFIED','ADDITIVE_ORTHOGONAL'], default='UNVERIFIED')
    p.add_argument('--measurement-source', type=Path, help='Local authoritative documentation/extraction code supporting units/decomposition.')
    p.add_argument('--correlation-semantics', choices=['UNVERIFIED','RAW_R','FISHER_Z'], default='UNVERIFIED')
    p.add_argument('--correlation-source', type=Path,
                   help='Required for optional diagnostic comparison with released correlations. No suffix-based inference.')
    p.add_argument('--psd-rtol', type=float, default=1e-6,
                   help='Relative eigenvalue tolerance; report raw spectrum and rounding-aware tolerance separately.')
    p.add_argument('--csv-entry-rtol', type=float, default=1e-6,
                   help='Assumed entry-wise relative CSV rounding uncertainty; not inferred release precision.')
    p.add_argument('--csv-entry-atol', type=float, default=0.0)
    p.add_argument('--symmetry-rtol', type=float, default=1e-6)
    p.add_argument('--symmetry-atol', type=float, default=1e-10)
    p.add_argument('--reproduction-tol', type=float, default=REPRO_TOL)
    p.add_argument('--identity-tol', type=float, default=IDENTITY_TOL)
    p.add_argument('--hash-large-archives', action='store_true', help='Optional complete SHA256 of large archives; default records size/mtime and ZIP CRCs.')
    args = p.parse_args()
    for name in ['data_dir','results_dir','subject_table','measurement_source','correlation_source']:
        value = getattr(args,name)
        if value is not None: setattr(args,name,value.expanduser().resolve())
    args.output_dir = (args.output_dir or args.results_dir/'10_noise_amplitude_and_normalization').expanduser().resolve()
    if args.output_dir == args.results_dir or args.output_dir == args.data_dir:
        p.error('--output-dir must be a dedicated analysis-10 subdirectory')
    if args.output_dir == args.results_dir/'02_primary_profile_and_measurement':
        p.error('Cannot overwrite analysis-02 outputs')
    if args.diagnostic_sessions_per_state < 1: p.error('Diagnostic sample size must be positive')
    for name in ['psd_rtol','csv_entry_rtol','csv_entry_atol','symmetry_rtol','symmetry_atol']:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<0: p.error(name+' must be finite and nonnegative')
    for name in ['reproduction_tol','identity_tol']:
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<=0: p.error(name+' must be finite and positive')
    if args.amplitude_units!='UNVERIFIED' or args.noise_decomposition!='UNVERIFIED':
        if not args.measurement_source or not args.measurement_source.is_file():
            p.error('Documented measurement semantics require --measurement-source')
    if args.correlation_semantics!='UNVERIFIED':
        if not args.correlation_source or not args.correlation_source.is_file():
            p.error('Correlation numeric scale requires --correlation-source')
    return args


def quantile_value(values, q):
    return float(np.quantile(values,q)) if len(values) else np.nan


def save_json(path, value):
    # Scientific unknowns are null, never nonstandard JSON NaN/Infinity.
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
        if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
        if isinstance(v,(np.integer,)): return int(v)
        if isinstance(v,np.bool_): return bool(v)
        if isinstance(v,Path): return str(v)
        return v
    path.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_table(out, name, rows, columns=None):
    table = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    table.to_csv(out/'tables'/name,index=False)
    return table


def canonical_ids(table, source):
    table = table.copy()
    col = identify_id_column(table.columns)
    table = table.rename(columns={col:'id_event'})
    if table.id_event.isna().any(): raise ValueError('Missing event ID in '+str(source))
    table['id_event'] = table.id_event.astype(str).str.strip()
    if table.id_event.eq('').any() or table.id_event.duplicated().any():
        raise ValueError('Empty or duplicate event IDs in '+str(source))
    return table


def column_map(columns, areas):
    # Fail on normalized duplicates rather than letting dict construction hide them.
    tokens = [compact_token(c) for c in columns]
    if len(set(tokens))!=len(tokens): raise ValueError('Duplicate normalized CSV header labels')
    lookup = dict(zip(tokens,columns))
    missing = [area for area in areas if compact_token(area) not in lookup]
    if missing: raise KeyError(f'Missing {len(missing)} parcels; first {missing[:4]}')
    return {area:lookup[compact_token(area)] for area in areas}


def load_amplitude_product(zip_path, areas, kind):
    names = [name for name in csv_members(zip_path) if compact_token(Path(name).stem).endswith(compact_token(kind))]
    if len(names)!=1: raise ValueError(f'{zip_path.name}: expected one {kind}.csv; found {len(names)}')
    member=names[0]; header=zip_header(zip_path,member)
    mapping=column_map(header,areas); idcol=identify_id_column(header)
    with ZipFile(zip_path) as z, z.open(member) as f:
        table=pd.read_csv(f,usecols=[idcol,*mapping.values()],dtype={idcol:str},low_memory=False)
    table=canonical_ids(table,zip_path.name+'/'+member)
    table=table.rename(columns={column:area for area,column in mapping.items()})
    for area in areas:
        original=table[area]; numeric=pd.to_numeric(original,errors='coerce')
        if (original.notna() & numeric.isna()).any(): raise ValueError('Nonnumeric amplitude in '+member)
        if (numeric[np.isfinite(numeric)]<0).any(): raise ValueError('Negative SD amplitude in '+member)
        table[area]=numeric.astype(float)
    return table.set_index('id_event')[areas],member


def pair_products(values, minimum_count=None):
    values=np.asarray(values,float)
    good=np.isfinite(values)
    count=good.sum(axis=1)
    clean=np.where(good,values,0.)
    sums=clean.sum(axis=1); squares=(clean*clean).sum(axis=1)
    need=values.shape[1] if minimum_count is None else minimum_count
    result=np.full(len(values),np.nan)
    valid=count>=max(2,need)
    result[valid]=(sums[valid]**2-squares[valid])/(count[valid]*(count[valid]-1))
    return result,count


def archive_signature(path, hash_large=False):
    stat=path.stat()
    with ZipFile(path) as z:
        members=sorted((i.filename,i.CRC,i.file_size,i.compress_size) for i in z.infolist())
    signature={'path':str(path),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
               'zip_members_crc_sha256':hashlib.sha256(json.dumps(members).encode()).hexdigest()}
    if hash_large: signature['sha256']=sha256_file(path)
    return signature


def load_diagnostic_matrices(path, areas, ids, private, force=False):
    signature=archive_signature(path)
    payload={'archive':signature,'areas':areas,'event_ids':ids,'version':VERSION}
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    cache=private/('diagnostic_'+key+'.npz')
    if cache.is_file() and not force:
        with np.load(cache,allow_pickle=False) as z:
            return z['matrices'].copy(), 'REUSED_CHECKPOINT'
    matrices=np.full((len(ids),len(areas),len(areas)),np.nan)
    for i,area in enumerate(areas):
        member=zip_member_for_area(path,area); header=zip_header(path,member)
        mapping=column_map(header,areas); idcol=identify_id_column(header)
        selected=read_selected_rows(path,member,[idcol,*mapping.values()],ids)
        # read_selected_rows validates duplicates across all selected chunks.
        if set(selected.index)!=set(ids): raise ValueError(f'{path.name}/{member}: selected event alignment failed')
        raw=selected.reindex(ids)[[mapping[a] for a in areas]]
        converted=raw.apply(pd.to_numeric,errors='coerce')
        if (raw.notna() & converted.isna()).any().any(): raise ValueError('Nonnumeric matrix entries')
        matrices[:,i,:]=converted.to_numpy(float)
        if (i+1)%60==0: print(f'  {path.name}: {i+1}/{len(areas)} source parcels',flush=True)
    np.savez_compressed(cache,matrices=matrices,event_ids=np.asarray(ids,dtype=str),areas=np.asarray(areas,dtype=str))
    return matrices,'BUILT_CHECKPOINT'


def symmetry_and_spectrum(matrix,args):
    c=np.asarray(matrix,float)
    finite=np.isfinite(c).all()
    result={'all_entries_finite':bool(finite),'entry_n':int(c.size),
            'nonfinite_entry_n':int((~np.isfinite(c)).sum())}
    if not finite: return result,None
    delta=np.abs(c-c.T)
    tolerance=args.symmetry_atol+args.symmetry_rtol*np.maximum(np.abs(c),np.abs(c.T))
    compatible=bool(np.all(delta<=tolerance))
    result.update(symmetry_max_abs_error=float(delta.max()),
                  symmetry_relative_frobenius=float(np.linalg.norm(c-c.T)/max(np.linalg.norm(c),np.finfo(float).tiny)),
                  symmetry_compatible_with_tolerance=compatible,
                  symmetry_bad_entry_n=int((delta>tolerance).sum()),
                  symmetrization_applied=bool(compatible and delta.max()>0))
    if not compatible: return result,None
    s=(c+c.T)/2
    eig=np.linalg.eigvalsh(s)
    scale=max(float(np.max(np.abs(eig))),np.finfo(float).tiny)
    # Bound spectral perturbation by infinity norm of symmetric entry-error bounds.
    round_bound=float(np.max(np.sum(args.csv_entry_atol+args.csv_entry_rtol*np.abs(s),axis=1)))
    numeric_tol=100*len(s)*np.finfo(float).eps*scale
    relative_tol=args.psd_rtol*scale
    tol=max(numeric_tol,relative_tol,round_bound)
    result.update(eigenvalue_min=float(eig[0]),eigenvalue_max=float(eig[-1]),
                  eigenvalue_min_over_spectral_scale=float(eig[0]/scale),
                  negative_eigenvalue_n=int((eig<0).sum()),
                  negative_beyond_tolerance_n=int((eig < -tol).sum()),
                  psd_tolerance_numeric=numeric_tol,psd_tolerance_relative=relative_tol,
                  psd_tolerance_assumed_csv_rounding=round_bound,psd_tolerance_used=tol,
                  psd_within_tolerance=bool(eig[0]>=-tol))
    return result,s


def pair_ratio(c,sd):
    iu=np.triu_indices(len(sd),1); denominator=np.outer(sd,sd)[iu]
    numerator=c[iu]; ratio=np.full(len(denominator),np.nan)
    valid=np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0)
    ratio[valid]=numerator[valid]/denominator[valid]
    return ratio,denominator,valid


def diagnostic_stage(args,parcel_map,networks,amplitudes,archives,out):
    areas=parcel_map['Brain Area'].tolist(); previous=load_existing_diagnostic_ids(args.results_dir)
    matrix_rows=[]; ratio_rows=[]; pair_rows=[]; diagonal_rows=[]; network_rows=[]
    bin_rows=[]; alignment_rows=[]; comparison_rows=[]; extreme_rows=[]; failures=[]
    for state in STATES:
        print('Diagnostic reconstruction: '+state,flush=True)
        if archives['covariance'][state] is None: raise FileNotFoundError('Missing covariance ZIP for '+state)
        a,n=amplitudes[state]['signal'],amplitudes[state]['noise']
        cov_ids=set(archive_event_ids(archives['covariance'][state],areas[0]))
        common=sorted(set(a.index)&set(n.index)&cov_ids)
        frozen=previous.get(state)
        if frozen:
            ids=list(frozen)
            missing=set(ids)-set(common)
            if missing: raise ValueError(f'{state}: {len(missing)} frozen diagnostic IDs lack aligned products')
            selection='ANALYSIS_02_FROZEN'
        else:
            ids=deterministic_diagnostic_ids(common,args.diagnostic_sessions_per_state)
            selection='DETERMINISTIC_FALLBACK'
        if not ids: raise ValueError('No aligned diagnostic sessions for '+state)
        matrices,checkpoint=load_diagnostic_matrices(archives['covariance'][state],areas,ids,out/'private',args.force_diagnostic)
        corr=None
        if args.correlation_semantics!='UNVERIFIED' and archives['correlation'][state] is not None:
            corr,_=load_diagnostic_matrices(archives['correlation'][state],areas,ids,out/'private',args.force_diagnostic)
            if args.correlation_semantics=='FISHER_Z': corr=np.tanh(corr)
        nodepath=resolve_node_table(args.data_dir,args.results_dir,state)
        cached=canonical_ids(read_table(nodepath),nodepath).set_index('id_event') if nodepath else None
        for j,event in enumerate(ids):
            label=f'{state}_{j+1:02d}'; c=matrices[j]
            av=a.loc[event,areas].to_numpy(float); nv=n.loc[event,areas].to_numpy(float)
            if not np.isfinite(av).all() or not np.isfinite(nv).all():
                raise ValueError(f'{label}: nonfinite diagnostic SD amplitudes')
            t=np.hypot(av,nv)
            stats,symmetric=symmetry_and_spectrum(c,args)
            matrix_rows.append({'state':state,'diagnostic_session':label,'matrix':'RELEASED_SIGNAL_DIAGONAL',**stats})
            alignment_rows.append({'state':state,'diagnostic_session':label,'selection':selection,
                                   'cache_status':checkpoint,'parcel_n':len(areas),
                                   'signal_and_noise_event_sets_identical':set(a.index)==set(n.index),
                                   'signal_event_n':len(a),'noise_event_n':len(n),'shared_event_n':len(common)})
            if symmetric is None:
                failures.append(label+': nonfinite matrix or material asymmetry; PSD/normalization inference withheld')
                continue
            total=c.copy(); total[np.diag_indices(len(c))]+=nv**2
            totalstats,totalsymmetric=symmetry_and_spectrum(total,args)
            matrix_rows.append({'state':state,'diagnostic_session':label,'matrix':'CANDIDATE_TOTAL',**totalstats})
            rs,ds,vs=pair_ratio(symmetric,av); rt,dt,vt=pair_ratio(symmetric,t)
            diagonal=np.diag(c); expected=av**2
            rel=np.abs(diagonal-expected)/np.maximum(np.abs(expected),np.finfo(float).tiny)
            totaldiag=np.diag(total); normdiag=np.divide(totaldiag,t**2,out=np.full_like(t,np.nan),where=t>0)
            diagonal_rows.append({'state':state,'diagnostic_session':label,
                'signal_diagonal_relative_error_median':float(np.median(rel)),
                'signal_diagonal_relative_error_max':float(np.max(rel)),
                'signal_diagonal_absolute_error_max':float(np.max(np.abs(diagonal-expected))),
                'total_normalized_diagonal_max_abs_error_from_1':float(np.nanmax(np.abs(normdiag-1))) if np.isfinite(normdiag).any() else np.nan,
                'zero_signal_sd_n':int((av==0).sum()),'zero_total_sd_n':int((t==0).sum())})
            iu=np.triu_indices(len(av),1)
            attenuation=np.divide(ds,dt,out=np.full_like(ds,np.nan),where=dt>0)
            valid=vs&vt
            finite_pairs=np.flatnonzero(valid)
            top=finite_pairs[np.argsort(-np.abs(rs[finite_pairs]),kind='stable')[:10]]
            for rank,index in enumerate(top,1):
                ppos,qpos=iu[0][index],iu[1][index]
                extreme_rows.append({'state':state,'diagnostic_session':label,'rank_abs_signal_ratio':rank,
                    'parcel_p':areas[ppos],'parcel_q':areas[qpos],'C_pq':symmetric[ppos,qpos],
                    'signal_sd_p':av[ppos],'signal_sd_q':av[qpos],'noise_sd_p':nv[ppos],'noise_sd_q':nv[qpos],
                    'total_sd_p':t[ppos],'total_sd_q':t[qpos],'ratio_signal':rs[index],'ratio_total':rt[index],
                    'pair_sd_fraction':attenuation[index],
                    'diagonal_p_minus_signal_sd_squared':diagonal[ppos]-av[ppos]**2,
                    'diagonal_q_minus_signal_sd_squared':diagonal[qpos]-av[qpos]**2})
            for kind,r in [('SIGNAL_NORMALIZED',rs),('CANDIDATE_TOTAL_NORMALIZED',rt)]:
                ratio_rows.append({'state':state,'diagnostic_session':label,'representation':kind,
                    'all_pair_n':len(r),'undefined_pair_n':int((~np.isfinite(r)).sum()),**summarize_ratio(r)})
            pair_rows.append({'state':state,'diagnostic_session':label,
                'median_signal_amplitude_product':float(np.median(ds)),
                'median_total_amplitude_product':float(np.median(dt)),
                'median_pair_sd_fraction':float(np.nanmedian(attenuation)),
                'ratio_identity_max_abs_error':float(np.max(np.abs(rt[valid]-rs[valid]*attenuation[valid]))) if valid.any() else np.nan,
                'spearman_abs_signal_ratio_vs_pair_sd_fraction':spearmanr(np.abs(rs[valid]),attenuation[valid])[0] if valid.sum()>2 and np.ptp(attenuation[valid])>0 else np.nan})
            # Tied quartiles use rank groups deterministically; report tie convention.
            for variable,values in [('SIGNAL_PRODUCT',ds),('PAIR_SD_FRACTION',attenuation)]:
                valid_bin=np.isfinite(values)&np.isfinite(rs)&np.isfinite(rt)
                indices=np.flatnonzero(valid_bin)
                ordered=indices[np.argsort(values[indices],kind='stable')]
                for b,ix in enumerate(np.array_split(ordered,4),1):
                    bin_rows.append({'state':state,'diagnostic_session':label,'bin_variable':variable,
                        'quartile':b,'pair_n':len(ix),'bin_min':float(np.min(values[ix])) if len(ix) else np.nan,
                        'bin_max':float(np.max(values[ix])) if len(ix) else np.nan,
                        'signal_fraction_abs_gt_1_01':float(np.mean(np.abs(rs[ix])>1.01)) if len(ix) else np.nan,
                        'total_fraction_abs_gt_1_01':float(np.mean(np.abs(rt[ix])>1.01)) if len(ix) else np.nan,
                        'ties':'stable rank split; tied values may span adjacent bins'})
            for network in networks:
                positions=np.flatnonzero(parcel_map['node'].map(compact_token).eq(compact_token(network)).to_numpy())
                block=symmetric[np.ix_(positions,positions)]
                cov=float(block[np.triu_indices(len(positions),1)].mean())
                ps=pair_products(av[positions][None,:])[0][0]
                pt=pair_products(t[positions][None,:])[0][0]
                released=float(cached.loc[event,f'fc__{slug(network)}__{slug(network)}']) if cached is not None and event in cached.index else np.nan
                difference=abs(cov-released)/max(abs(released),np.finfo(float).tiny) if np.isfinite(released) else np.nan
                network_rows.append({'state':state,'diagnostic_session':label,'network':network,
                    'parcel_n':len(positions),'covariance_reconstructed':cov,'covariance_cached':released,
                    'covariance_relative_difference':difference,'P_signal':ps,'P_total':pt,
                    'C_over_P_signal':cov/ps if ps>0 else np.nan,'C_over_P_total':cov/pt if pt>0 else np.nan,
                    'amplitude_product_attenuation_factor':ps/pt if pt>0 else np.nan})
                if np.isfinite(difference) and difference>args.reproduction_tol:
                    failures.append(label+': covariance aggregate reproduction failed for '+network)
            if corr is not None:
                rv=corr[j][iu]; good=np.isfinite(rv)&np.isfinite(rt)
                comparison_rows.append({'state':state,'diagnostic_session':label,'numeric_scale':args.correlation_semantics,
                    'pair_n':int(good.sum()),'released_fraction_abs_gt_1_01':float(np.mean(np.abs(rv[good])>1.01)) if good.any() else np.nan,
                    'candidate_total_vs_released_rmse':float(np.sqrt(np.mean((rt[good]-rv[good])**2))) if good.any() else np.nan,
                    'candidate_total_vs_released_median_abs_error':float(np.median(np.abs(rt[good]-rv[good]))) if good.any() else np.nan,
                    'candidate_total_vs_released_max_abs_error':float(np.max(np.abs(rt[good]-rv[good]))) if good.any() else np.nan})
    for name,rows in [('01_diagnostic_alignment.csv',alignment_rows),('02_matrix_structure.csv',matrix_rows),
        ('03_diagonal_reconstruction.csv',diagonal_rows),('04_pairwise_normalization.csv',ratio_rows),
        ('05_pairwise_attenuation.csv',pair_rows),('06_pairwise_quartiles.csv',bin_rows),
        ('07_diagnostic_network_reproduction.csv',network_rows),('09_extreme_pair_reconstruction.csv',extreme_rows)]: write_table(out,name,rows)
    write_table(out,'08_released_correlation_comparison.csv',comparison_rows,
        columns=['state','diagnostic_session','numeric_scale','pair_n','released_fraction_abs_gt_1_01',
                 'candidate_total_vs_released_rmse','candidate_total_vs_released_median_abs_error','candidate_total_vs_released_max_abs_error'])
    result={'stage':'diagnostic','technical_failures':failures,'diagnostic_sessions':len(alignment_rows),
            'matrix_indefiniteness_is_technical_failure':False,
            'construction_interpretation':'CONDITIONAL; structural support does not uniquely identify release algorithm',
            'full_correlation_comparison':'RUN' if comparison_rows else 'NOT_RUN_UNVERIFIED_OR_UNAVAILABLE'}
    save_json(out/'diagnostic_summary.json',result)
    diagnostic_figures(out,pd.DataFrame(matrix_rows),pd.DataFrame(ratio_rows))
    if failures: raise ValueError('Diagnostic technical failures; see diagnostic_summary.json')
    return result
def recorded_file(raw_path,data_dir,expected_sha=None):
    direct=Path(str(raw_path)).expanduser()
    candidates=[direct.resolve()] if direct.is_file() else sorted({p.resolve() for p in data_dir.rglob(direct.name) if p.is_file()})
    if not candidates: raise FileNotFoundError('Recorded analysis-02 source not found: '+direct.name)
    if expected_sha and str(expected_sha) not in ['nan','NOT_COMPUTED_LARGE_ARCHIVE','']:
        candidates=[p for p in candidates if sha256_file(p)==expected_sha]
        if not candidates: raise ValueError('Recorded source SHA256 mismatch: '+direct.name)
    if len(candidates)>1 and len({sha256_file(p) for p in candidates})>1:
        raise ValueError('Ambiguous recorded source: '+direct.name)
    return candidates[0]


def load_original_sources(args):
    base=args.results_dir/'02_primary_profile_and_measurement'
    required=[base/'run_manifest.json',base/'gate1_run_manifest.json',base/'tables/source_manifest.csv',
              base/'tables/common_level_state_slopes.csv',base/'tables/network_profile_state_slopes.csv']
    absent=[str(p) for p in required if not p.is_file()]
    if absent: raise FileNotFoundError('Analysis-02 outputs required for frozen reproduction:\n'+'\n'.join(absent))
    source=read_table(base/'tables/source_manifest.csv')
    source.columns=source.columns.astype(str)
    if not {'role','path'}.issubset(source.columns): raise ValueError('Invalid analysis-02 source manifest')
    resolved={}
    for _,row in source.iterrows():
        role=str(row.role)
        if role=='longitudinal_subject_event_table' or role=='statewise_metric_input' or role in ['raw_'+s for s in STATES]:
            path=recorded_file(row.path,args.data_dir,row.get('sha256'))
            resolved[role]=path
    if args.subject_table:
        if not args.subject_table.is_file(): raise FileNotFoundError(str(args.subject_table))
        original=resolved.get('longitudinal_subject_event_table')
        if original and sha256_file(original)!=sha256_file(args.subject_table):
            raise ValueError('--subject-table differs from recorded analysis-02 input')
        resolved['longitudinal_subject_event_table']=args.subject_table
    if 'longitudinal_subject_event_table' not in resolved: raise ValueError('No recorded longitudinal subject table')
    return base,resolved,json.loads((base/'run_manifest.json').read_text()),json.loads((base/'gate1_run_manifest.json').read_text())


def load_subjects(path):
    raw=read_table(path)
    event=pick_column(raw,['id_event','subject_event','participant_event'],False)
    person=pick_column(raw,['id','subject_id','participant_id','src_subject_id'],False)
    visit=pick_column(raw,['eventname','event_name','event','visit','session'],False)
    if event is None:
        if person is None or visit is None: raise ValueError('Cannot construct subject-event IDs')
        raw=raw.copy(); raw['__event_id']=raw[person].astype(str).str.strip()+'_'+raw[visit].astype(str).str.strip(); event='__event_id'
    mappings={event:'id_event'}
    specs=[('event_age',['age_open','age','event_age','interview_age'],True),
           ('sex',['sex','gender'],True),('site',['site','site_name','site_id','collection_site','institution'],True),
           ('scanner',['scanner','scanner_id','scanner_name','acquisition_scanner','mri_scanner','scanner_model','scanner_serial_number'],False),
           ('study',['study','project','cohort'],False),('pedigree_group',['pedid','pedigree','pedigree_id','family_id'],False),
           ('days_from_v1',['days_from_V1','daysfromv1'],False)]
    for target,aliases,required in specs:
        column=pick_column(raw,aliases,required,target)
        if column is not None: mappings[column]=target
    if person is not None: mappings[person]='id'
    subjects=raw[list(mappings)].rename(columns=mappings).copy()
    subjects=canonical_ids(subjects,path)
    if 'id' not in subjects: subjects['id']=subjects.id_event.str.replace(r'_[^_]+$','',regex=True)
    if subjects.id.isna().any(): raise ValueError('Missing participant IDs')
    subjects['id']=subjects.id.astype(str).str.strip()
    if subjects.id.eq('').any(): raise ValueError('Empty participant IDs')
    for column in ['sex','site','scanner','study']:
        if column not in subjects: subjects[column]='MISSING'
        subjects[column]=subjects[column].astype('string').str.strip().fillna('MISSING')
    if 'pedigree_group' not in subjects: subjects['pedigree_group']='UNRELATED__'+subjects.id
    subjects['pedigree_group']=subjects.pedigree_group.astype('string').str.strip()
    missing=subjects.pedigree_group.isna()|subjects.pedigree_group.str.upper().isin(['','<NA>','NAN','NA'])
    subjects.loc[missing,'pedigree_group']='UNRELATED__'+subjects.loc[missing,'id']
    for column in ['event_age','days_from_v1']:
        if column not in subjects: subjects[column]=np.nan
        subjects[column]=pd.to_numeric(subjects[column],errors='coerce')
    return subjects


def reconstruct_timing(sources,subjects):
    metrics=['within_default','within_frontoparietal']
    if all('raw_'+state in sources for state in STATES):
        combined=None
        for state in STATES:
            table=canonical_ids(read_table(sources['raw_'+state]),sources['raw_'+state])
            part=table[['id_event',*metrics]].rename(columns={m:f'raw__{state.lower()}__{m}' for m in metrics})
            combined=part if combined is None else combined.merge(part,on='id_event',how='outer',validate='one_to_one')
    elif 'statewise_metric_input' in sources:
        path=sources['statewise_metric_input']; table=canonical_ids(read_table(path),path)
        combined=table[['id_event']].copy()
        for state in STATES:
            for metric in metrics:
                aliases=[f'{prefix}{state.lower()}__{metric}' for prefix in ['raw__','resid__','residual__','oof__','','fmri__']]
                col=pick_column(table,aliases,True,f'{state}/{metric}')
                combined[f'raw__{state.lower()}__{metric}']=pd.to_numeric(table[col],errors='coerce')
    else: raise ValueError('Cannot reconstruct frozen event timing from recorded metric input')
    for column in combined.columns:
        if column!='id_event': combined[column]=pd.to_numeric(combined[column],errors='coerce')
    usable=combined.drop(columns='id_event').notna().any(axis=1)
    events=combined.loc[usable,['id_event']].merge(subjects,on='id_event',how='inner',validate='one_to_one')
    events['sort_time']=events.days_from_v1.where(events.days_from_v1.notna(),events.event_age*365.25)
    events=events.sort_values(['id','sort_time','id_event'])
    events['baseline_sort_time']=events.groupby('id').sort_time.transform('min')
    baseline_age=events.dropna(subset=['event_age']).groupby('id').event_age.first()
    center=float(baseline_age.mean())
    events['baseline_age']=events.id.map(baseline_age)
    events['time_years']=(events.sort_time-events.baseline_sort_time)/365.25
    events['t5']=events.time_years/5.; events['age0_c5']=(events.baseline_age-center)/5.
    events['age0_c10']=(events.baseline_age-center)/10.
    events['visit_number']=events.groupby('id').cumcount()+1
    events['site_scanner']=events.site.astype(str)+' | '+events.scanner.astype(str)
    return events,center


def load_network_rows(args,events,networks,base):
    sourcepath=base/'tables/full_node_sources.csv'
    records=read_table(sourcepath) if sourcepath.is_file() else pd.DataFrame()
    rows=[]
    for state in STATES:
        found=records.loc[records.state.eq(state)] if 'state' in records else pd.DataFrame()
        if len(found)==1:
            row=found.iloc[0]; path=recorded_file(row.path,args.data_dir,row.get('sha256'))
        else:
            path=resolve_node_table(args.data_dir,args.results_dir,state)
        if path is None: raise FileNotFoundError('Missing within-network covariance cache for '+state)
        table=canonical_ids(read_table(path),path)
        columns={f'fc__{slug(n)}__{slug(n)}':'C__'+n for n in networks}
        missing=set(columns)-set(table)
        if missing: raise KeyError('Missing network covariance columns: '+str(sorted(missing)))
        table=table[['id_event',*columns]].rename(columns=columns)
        for col in columns.values(): table[col]=pd.to_numeric(table[col],errors='coerce')
        part=events.merge(table,on='id_event',how='left',validate='one_to_one'); part['state']=state; rows.append(part)
    full=pd.concat(rows,ignore_index=True)
    valid=np.isfinite(full[['age0_c5','t5']].to_numpy(float)).all(axis=1)&(full['C__Default']>0)&(full['C__Frontoparietal']>0)
    parent=retain_two_sessions(full.loc[valid].copy())
    cov=parent[['C__'+n for n in networks]].to_numpy(float)
    profile=retain_two_sessions(parent.loc[np.isfinite(cov).all(axis=1)&(cov>0).all(axis=1)].copy())
    return parent,profile


def key_hash(frame):
    keys=frame[['id','id_event','state']].astype(str).sort_values(['id','id_event','state'])
    return hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()


def prepare_outcomes(args,parcel_map,networks,amplitudes,out):
    base,sources,manifest,gate=load_original_sources(args)
    frozen_networks=list(manifest.get('eligible_networks',[]))
    if frozen_networks!=networks: raise ValueError('Frozen network order mismatch')
    subjects=load_subjects(sources['longitudinal_subject_event_table'])
    events,center=reconstruct_timing(sources,subjects)
    parent,profile=load_network_rows(args,events,networks,base)
    if subjects.groupby('id').pedigree_group.nunique().max()>1:
        raise ValueError('Participant assigned to multiple pedigree groups across visits')
    aggregate=[]; coverage=[]; formula_rows=[]
    for state in STATES:
        a=amplitudes[state]['signal']; noise=amplitudes[state]['noise']; n=noise.reindex(a.index)
        rows={'id_event':a.index.to_numpy(),'state':state}
        for network in networks:
            areas=parcel_map.loc[parcel_map.node.map(compact_token).eq(compact_token(network)),'Brain Area'].tolist()
            av=a[areas].to_numpy(float); nv=n[areas].to_numpy(float)
            old,count=pair_products(av,max(2,math.ceil(.75*len(areas))))
            ps,strict_count=pair_products(av)
            total=np.hypot(av,nv); pt,total_count=pair_products(total)
            rows['P_old__'+network]=old; rows['P_signal__'+network]=ps; rows['P_total__'+network]=pt
            coverage.append({'state':state,'network':network,'parcel_n':len(areas),'signal_event_n':len(a),
                'noise_event_n':len(noise),'noise_only_event_n':len(set(noise.index)-set(a.index)),
                'original_75pct_positive_P_n':int((np.isfinite(old)&(old>0)).sum()),
                'signal_full_parcel_support_n':int((strict_count==len(areas)).sum()),
                'paired_full_parcel_support_n':int((total_count==len(areas)).sum()),
                'matched_positive_products_n':int((np.isfinite(ps)&np.isfinite(pt)&(ps>0)&(pt>0)).sum())})
            # Direct upper-triangle reproduction on the first finite complete amplitude row.
            complete=np.flatnonzero(np.isfinite(av).all(axis=1))
            if len(complete):
                v=av[complete[0]]; direct=float(np.outer(v,v)[np.triu_indices(len(v),1)].mean())
                formula_rows.append({'state':state,'network':network,'P_formula':ps[complete[0]],'P_direct_pairs':direct,
                                     'relative_difference':abs(ps[complete[0]]-direct)/max(abs(direct),np.finfo(float).tiny)})
        aggregate.append(pd.DataFrame(rows))
    amp=pd.concat(aggregate,ignore_index=True)
    joined=profile.merge(amp,on=['id_event','state'],how='left',validate='one_to_one')
    old_values=joined[['P_old__'+n for n in networks]].to_numpy(float)
    original=retain_two_sessions(joined.loc[np.isfinite(old_values).all(axis=1)&(old_values>0).all(axis=1)].copy())
    matched_values=original[[prefix+n for prefix in ['P_signal__','P_total__'] for n in networks]].to_numpy(float)
    matched=retain_two_sessions(original.loc[np.isfinite(matched_values).all(axis=1)&(matched_values>0).all(axis=1)].copy())
    original=original.sort_values(['id','id_event','state']).reset_index(drop=True)
    matched=matched.sort_values(['id','id_event','state']).reset_index(drop=True)
    for label,frame in [('ORIGINAL_SIGNAL',original),('NOISE_MATCHED',matched)]:
        write_table(out,'10_'+label.lower()+'_cohort_by_state.csv',[
            {'state':s,**cohort_row(label,frame.loc[frame.state.eq(s)])} for s in STATES])
    cohort=write_table(out,'10_cohort_summary.csv',[cohort_row('FROZEN_PARENT',parent),cohort_row('PROFILE',profile),
        cohort_row('ORIGINAL_SIGNAL',original),cohort_row('NOISE_MATCHED',matched)])
    write_table(out,'11_parcel_support_coverage.csv',coverage)
    write_table(out,'12_pair_amplitude_formula.csv',formula_rows)
    observed_hash=key_hash(original)
    expected_hash=gate.get('amplitude_matched_row_sha256')
    if not expected_hash: raise ValueError('Frozen amplitude-matched row fingerprint absent from analysis 02')
    audits=[{'item':'original_signal_row_hash','observed':observed_hash,'expected':expected_hash,'pass':observed_hash==expected_hash},
            {'item':'eligible_network_order','observed':json.dumps(networks),'expected':json.dumps(frozen_networks),'pass':networks==frozen_networks}]
    for field,observed in [('amplitude_matched_rows',len(original)),('amplitude_matched_participants',original.id.nunique())]:
        if field in gate: audits.append({'item':field,'observed':observed,'expected':gate[field],'pass':observed==gate[field]})
    write_table(out,'13_frozen_cohort_reproduction.csv',audits)
    if not all(r['pass'] for r in audits): raise ValueError('Frozen cohort reproduction failed; see table 13. No longitudinal fits run.')
    if len(matched)==0: raise ValueError('No complete signal/noise matched observations')
    for frame,old in [(original,True),(matched,False)]:
        frame.attrs['Z_SIGNAL']=np.log(frame[['C__'+n for n in networks]].to_numpy(float))-np.log(frame[[('P_old__' if old else 'P_signal__')+n for n in networks]].to_numpy(float))
    zs=matched.attrs['Z_SIGNAL']
    h=np.log(matched[['P_signal__'+n for n in networks]].to_numpy(float))-np.log(matched[['P_total__'+n for n in networks]].to_numpy(float))
    zt=np.log(matched[['C__'+n for n in networks]].to_numpy(float))-np.log(matched[['P_total__'+n for n in networks]].to_numpy(float))
    if (h>args.identity_tol).any(): raise AssertionError('H must be nonpositive under candidate noise reconstruction')
    errors=[]
    fp=np.full(len(networks),-1/(len(networks)-1)); fp[networks.index('Frontoparietal')]=1.
    for kind,transform in [('NETWORK_LOG',lambda z:z),('COMMON',lambda z:z.mean(axis=1)),
                           ('CENTERED_PROFILE',lambda z:z-z.mean(axis=1,keepdims=True)),('FP_RELATIVE',lambda z:z@fp)]:
        error=float(np.max(np.abs(transform(zt)-transform(zs)-transform(h))))
        errors.append({'identity':kind,'max_absolute_error':error,'pass':error<args.identity_tol})
    oldmatch=original.set_index(['id_event','state'])
    old_p=oldmatch.loc[pd.MultiIndex.from_frame(matched[['id_event','state']]),['P_old__'+n for n in networks]].to_numpy(float)
    ps=matched[['P_signal__'+n for n in networks]].to_numpy(float)
    p_error=float(np.max(np.abs(old_p-ps)/np.maximum(np.abs(old_p),np.finfo(float).tiny)))
    formula_max=max((r['relative_difference'] for r in formula_rows),default=np.inf)
    errors.extend([{'identity':'ORIGINAL_P_ON_MATCHED_COMPLETE_SUPPORT','max_absolute_error':p_error,'pass':p_error<args.reproduction_tol},
                   {'identity':'PAIR_PRODUCT_FORMULA_RELATIVE_ERROR','max_absolute_error':formula_max,'pass':formula_max<args.reproduction_tol}])
    write_table(out,'14_observation_identities.csv',errors)
    if not all(r['pass'] for r in errors): raise AssertionError('Outcome algebra or pair-amplitude reproduction failed')
    return base,original,matched,{'SIGNAL':zs,'ATTENUATION_H':h,'TOTAL':zt},center,cohort,observed_hash


def fixed_joint_gls(x,y,participant,state,pedigree,tau_p,tau_s):
    """One fixed row-covariance operator, joint outcome-cluster influences.

    y is n x D. Coefficients are p x D, flattened coordinate-major (Fortran).
    No rows are dropped inside the function. Numerical scaling matches script 02.
    """
    x=np.asarray(x,float); y=np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    participant=np.asarray(participant).astype(str); state=np.asarray(state).astype(str); pedigree=np.asarray(pedigree).astype(str)
    n,p=x.shape; d=y.shape[1]
    if y.shape[0]!=n or not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Fixed GLS requires all matched finite rows')
    if n<=p+12 or len(np.unique(pedigree))<2: raise ValueError('Insufficient observations or pedigree clusters')
    if min(tau_p,tau_s)<0 or not np.isfinite([tau_p,tau_s]).all(): raise ValueError('Invalid working covariance ratios')
    scale=np.sqrt(np.mean(x*x,axis=0)); scale=np.where(scale>1e-10,scale,1.)
    xs=x/scale; xtx=np.zeros((p,p)); xty=np.zeros((p,d)); blocks=[]
    for pid in np.unique(participant):
        ix=np.flatnonzero(participant==pid)
        if len(set(pedigree[ix]))!=1: raise ValueError('Inconsistent participant pedigree membership')
        a=np.eye(len(ix))+tau_p*np.ones((len(ix),len(ix)))+tau_s*(state[ix,None]==state[None,ix])
        ax=np.linalg.solve(a,xs[ix]); ay=np.linalg.solve(a,y[ix])
        xtx+=xs[ix].T@ax; xty+=xs[ix].T@ay
        blocks.append((ix,a,pedigree[ix[0]]))
    if np.linalg.matrix_rank(xtx)!=p: raise ValueError('GLS design is rank deficient')
    bread=np.linalg.pinv(xtx,rcond=1e-11); b=bread@xty
    residual=y-xs@b; scores={}
    for ix,a,pg in blocks:
        score=xs[ix].T@np.linalg.solve(a,residual[ix])
        scores[pg]=scores.get(pg,np.zeros_like(score))+score
    g=len(scores); correction=g/(g-1)*(n-1)/(n-p)
    influences=np.stack([(math.sqrt(correction)*(bread@scores[pg])/scale[:,None]).ravel(order='F') for pg in sorted(scores)])
    coef=(b/scale[:,None]).ravel(order='F')
    return {'coef':coef,'robust_cov':influences.T@influences,'influence':influences,
            'coefficient_matrix':b/scale[:,None],'row_n':n,'pedigree_n':g,'design_rank':p}


def reference_fit(frame,z,family,out,force=False):
    design=DirectStateDesign().fit(frame); x=design.transform(frame)
    y=z.mean(axis=1) if family=='COMMON' else z-z.mean(axis=1,keepdims=True)
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(frame[['id','id_event','state','pedigree_group','age0_c5','age0_c10','t5','site','site_scanner','sex','study']],index=False).to_numpy().tobytes()+
        np.asarray(y,float).tobytes()+json.dumps(design.names).encode()+VERSION.encode()).hexdigest()
    cache=out/'private'/('reference_'+family.lower()+'_'+fingerprint+'.json')
    if cache.is_file() and not force:
        fit=json.loads(cache.read_text()); fit['coef']=np.asarray(fit['coef']); fit['robust_cov']=np.asarray(fit['robust_cov'])
        if 'basis' in fit: fit['basis']=np.asarray(fit['basis'])
        fit['checkpoint_status']='REUSED'
    else:
        func=fit_direct_profile_covariance if family=='COMMON' else fit_multivariate_profile
        fit=func(x,y,frame.id,frame.state,frame.pedigree_group)
        if fit is None: raise RuntimeError(family+' reference REML fit failed')
        # Keep coefficients/uncertainty; private pedigree IDs are not exported.
        fit.pop('influences',None)
        save_json(cache,fit); fit['checkpoint_status']='BUILT'
    if not fit['optimizer_success']:
        raise RuntimeError(family+' reference optimizer did not converge; checkpoint retained for inspection')
    return design,x,fit
def legacy_reproduction(args,base,original,networks,out):
    z=original.attrs['Z_SIGNAL']; records=[]; refs={}
    common=read_table(base/'tables/common_level_state_slopes.csv')
    profile=read_table(base/'tables/network_profile_state_slopes.csv')
    common['slope_per_5y']=pd.to_numeric(common['slope_per_5y'],errors='raise')
    profile['relative_profile_slope_per_5y']=pd.to_numeric(profile['relative_profile_slope_per_5y'],errors='raise')
    for family,table in [('COMMON',common),('PROFILE',profile)]:
        print('Reproduce original SIGNAL '+family+' model',flush=True)
        design,x,fit=reference_fit(original,z,family,out,args.force_refit)
        refs[family]=(design,x,fit)
        old=table.loc[table.scenario.eq('AMPLITUDE_STANDARDIZED')]
        if old.empty: raise ValueError('No original AMPLITUDE_STANDARDIZED rows for '+family)
        cs=design.state_slope_matrix()
        if family=='COMMON':
            if len(old)!=4: raise ValueError('Expected four original common-level slopes')
            for j,state in enumerate(STATES):
                values=old.loc[old.state.eq(state),'slope_per_5y']
                if len(values)!=1: raise ValueError('Ambiguous original state slopes')
                new=float(cs[j]@fit['coef']); previous=float(values.iloc[0]); difference=abs(new-previous)
                records.append({'family':family,'network':'COMMON','state':state,'original_slope_per_5y':previous,
                                'reconstructed_slope_per_5y':new,'absolute_difference':difference,'pass':difference<args.reproduction_tol})
        else:
            basis=np.asarray(fit['basis'])
            if len(old)!=4*len(networks): raise ValueError('Expected original ten-network x four-state slopes')
            for k,network in enumerate(networks):
                for j,state in enumerate(STATES):
                    values=old.loc[old.network.eq(network)&old.state.eq(state),'relative_profile_slope_per_5y']
                    if len(values)!=1: raise ValueError('Ambiguous original network slopes')
                    vector=np.kron(basis[k],cs[j]); new=float(vector@fit['coef']); previous=float(values.iloc[0])
                    difference=abs(new-previous)
                    records.append({'family':family,'network':network,'state':state,'original_slope_per_5y':previous,
                                    'reconstructed_slope_per_5y':new,'absolute_difference':difference,'pass':difference<args.reproduction_tol})
    write_table(out,'15_original_signal_fit_reproduction.csv',records)
    if not all(r['pass'] for r in records): raise ValueError('Original signal slopes failed reproduction; new sensitivity inference withheld')
    return refs


def fit_family(args,matched,outcomes,networks,family,reference,out):
    design,x,anchor=reference; k=len(networks); p=x.shape[1]
    coordinate_n=1 if family=='COMMON' else k
    ys=[outcomes[c].mean(axis=1,keepdims=True) if family=='COMMON' else outcomes[c] for c in COMPONENTS]
    joint=fixed_joint_gls(x,np.hstack(ys),matched.id,matched.state,matched.pedigree_group,
                          anchor['tau_participant'],anchor['tau_participant_state'])
    coef,cov=joint['coef'],joint['robust_cov']; size=p*coordinate_n
    slices=[slice(i*size,(i+1)*size) for i in range(3)]
    error=float(np.max(np.abs(coef[slices[2]]-coef[slices[0]]-coef[slices[1]])))
    influence_error=float(np.max(np.abs(joint['influence'][:,slices[2]]-joint['influence'][:,slices[0]]-joint['influence'][:,slices[1]])))
    # Includes cross-component uncertainty; variances alone are not additive.
    predicted_cov=cov[slices[0],slices[0]]+cov[slices[1],slices[1]]+cov[slices[0],slices[1]]+cov[slices[1],slices[0]]
    covariance_error=float(np.max(np.abs(predicted_cov-cov[slices[2],slices[2]])))
    if family=='COMMON': signal_prediction=coef[slices[0]]
    else:
        b=coef[slices[0]].reshape((p,k),order='F')
        signal_prediction=(b@np.asarray(anchor['basis'])).ravel(order='F')
    anchor_error=float(np.max(np.abs(signal_prediction-np.asarray(anchor['coef']))))
    identities=[{'family':family,'identity':'COEFFICIENT_TOTAL_EQUALS_SIGNAL_PLUS_H','max_absolute_error':error,'pass':error<args.identity_tol},
                {'family':family,'identity':'JOINT_INFLUENCE_TOTAL_EQUALS_SIGNAL_PLUS_H','max_absolute_error':influence_error,'pass':influence_error<args.identity_tol},
                {'family':family,'identity':'COV_TOTAL_EQUALS_COV_SIGNAL_PLUS_COV_H_PLUS_CROSS_TERMS','max_absolute_error':covariance_error,'pass':covariance_error<args.identity_tol},
                {'family':family,'identity':'MATCHED_SIGNAL_REFERENCE_COEFFICIENTS','max_absolute_error':anchor_error,'pass':anchor_error<args.reproduction_tol}]
    if not all(r['pass'] for r in identities): raise AssertionError('Fixed-weight fit identity/reproduction failed: '+str(identities))
    def full_vector(vector,component):
        v=np.zeros(len(coef)); v[slices[component]]=vector; return v
    def contrast_row(vector,component,target,state,network=''):
        v=full_vector(vector,component)
        estimate,se,z,pvalue=scalar_contrast(coef,cov,v)
        cells=4*k if target in ['NETWORK_RELATIVE','NETWORK_LOG_LEVEL'] else 4
        critical=float(norm.ppf(1-.05/(2*cells)))
        return {'family':family,'component':COMPONENTS[component],'target':target,'network':network,'state':state,
                'slope_per_5y':estimate,'se':se,'ci95_low':estimate-1.96*se,'ci95_high':estimate+1.96*se,
                'simultaneous_ci_low':estimate-critical*se,'simultaneous_ci_high':estimate+critical*se,
                'interval_family':f'Bonferroni across {cells} slopes within this component/target; FP network selection not adjusted',
                'p_unadjusted':pvalue,'inference_role':'SELECTED_FP_SENSITIVITY' if target=='FP_RELATIVE' else 'NORMALIZATION_SENSITIVITY'}
    cs=design.state_slope_matrix(); slopes=[]; tests=[]; differences=[]
    basis=helmert(k,full=False).T
    fp=np.full(k,-1/(k-1)); fp[networks.index('Frontoparietal')]=1.
    for component in range(3):
        if family=='COMMON':
            targets=[('COMMON_LEVEL','',None)]
            test_matrices=[('all_four_common_level_slopes_zero',cs),('common_level_state_heterogeneity',cs[:3]-cs[3]),
                           ('common_level_site_x_time',design.site_time_matrix()),('common_level_site_x_time_x_state',design.site_time_state_matrix())]
        else:
            targets=[('NETWORK_RELATIVE',network,np.eye(k)[index]-np.ones(k)/k) for index,network in enumerate(networks)]
            targets += [('FP_RELATIVE','Frontoparietal',fp)]
            targets += [('NETWORK_LOG_LEVEL',network,np.eye(k)[index]) for index,network in enumerate(networks)]
            def profile_matrix(cmat):
                if len(cmat)==0: return np.zeros((0,size))
                return np.vstack([np.kron(basis[:,q],row) for q in range(k-1) for row in cmat])
            fp_matrix=np.vstack([np.kron(fp,row) for row in cs])
            test_matrices=[('time_x_network',profile_matrix(cs.mean(axis=0)[None,:])),
                           ('time_x_network_x_state',profile_matrix(cs[:3]-cs[3])),
                           ('site_x_time_x_network',profile_matrix(design.site_time_matrix())),
                           ('site_x_time_x_state_x_network',profile_matrix(design.site_time_state_matrix())),
                           ('FP_all_four_state_slopes_zero',fp_matrix),('FP_state_heterogeneity',fp_matrix[:3]-fp_matrix[3])]
        for test,matrix in test_matrices:
            big=np.zeros((len(matrix),len(coef))); big[:,slices[component]]=matrix
            if len(matrix): stat,df,pvalue=wald_matrix(coef,cov,big)
            else: stat,df,pvalue=np.nan,0,np.nan
            tests.append({'family':family,'component':COMPONENTS[component],'test':test,'chi2':stat,'df':df,'p_unadjusted':pvalue,
                          'inference_role':'SELECTED_FP_SENSITIVITY' if test.startswith('FP') else 'ALL_NETWORK_SENSITIVITY' if family=='PROFILE' else 'COMMON_LEVEL_SENSITIVITY'})
        for target,network,weight in targets:
            for j,state in enumerate(STATES):
                vector=cs[j] if family=='COMMON' else np.kron(weight,cs[j])
                slopes.append(contrast_row(vector,component,target,state,network))
                if component==0:
                    difference_vector=full_vector(vector,2)-full_vector(vector,0)
                    est,se,z,pvalue=scalar_contrast(coef,cov,difference_vector)
                    independent_variance=float(vector@cov[slices[2],slices[2]]@vector+vector@cov[slices[0],slices[0]]@vector)
                    differences.append({'family':family,'target':target,'network':network,'state':state,
                        'difference':'TOTAL_MINUS_SIGNAL_EQUALS_H','slope_difference_per_5y':est,'paired_se':se,
                        'ci95_low':est-1.96*se,'ci95_high':est+1.96*se,'p_unadjusted':pvalue,
                        'se_if_incorrectly_assumed_independent':math.sqrt(max(0,independent_variance)),
                        'difference_vs_H_estimate_error':abs(est-float(full_vector(vector,1)@coef)),
                        'difference_vs_H_variance_error':abs(se**2-float(full_vector(vector,1)@cov@full_vector(vector,1)))})
    metadata={'family':family,'working_covariance_reference':'MATCHED_SIGNAL_'+family,
              'tau_participant':anchor['tau_participant'],'tau_participant_state':anchor['tau_participant_state'],
              'optimizer_success':anchor['optimizer_success'],'checkpoint_status':anchor['checkpoint_status'],
              'rows':len(matched),'participants':matched.id.nunique(),'pedigrees':joint['pedigree_n'],
              'design_names':design.names,'dropped_nuisance_names':design.dropped_names,
              'component_order':COMPONENTS,'outcome_coordinate_order':['COMMON'] if family=='COMMON' else networks,
              'vectorization':'Fortran: design coefficients within outcome coordinate within component',
              'site_slope_estimand':'Equal-weight mean over modeled sites, matching analysis-02 sum coding',
              'uncertainty':'Joint pedigree-cluster sandwich, CR1 finite-sample correction, frozen GLS weights'}
    save_json(out/'private'/('model_'+family.lower()+'.json'),metadata)
    np.savez_compressed(out/'private'/('joint_'+family.lower()+'.npz'),coef=coef,robust_cov=cov,
                        design_names=np.asarray(design.names,dtype=str),components=np.asarray(COMPONENTS,dtype=str),
                        outcome_coordinates=np.asarray(['COMMON'] if family=='COMMON' else networks,dtype=str))
    return slopes,tests,differences,identities,metadata


def longitudinal_stage(args,parcel_map,networks,amplitudes,out):
    base,original,matched,outcomes,center,cohort,original_hash=prepare_outcomes(args,parcel_map,networks,amplitudes,out)
    originals=legacy_reproduction(args,base,original,networks,out)
    matched_hash=key_hash(matched); same=original_hash==matched_hash
    slope_rows=[]; test_rows=[]; difference_rows=[]; identity_rows=[]; metadata=[]
    for family in ['COMMON','PROFILE']:
        print('Fit fixed-weight normalization decomposition: '+family,flush=True)
        reference=originals[family] if same else reference_fit(matched,outcomes['SIGNAL'],family,out,args.force_refit)
        slopes,tests,differences,identities,meta=fit_family(args,matched,outcomes,networks,family,reference,out)
        slope_rows+=slopes; test_rows+=tests; difference_rows+=differences; identity_rows+=identities; metadata.append(meta)
    slopes=write_table(out,'20_longitudinal_slopes.csv',slope_rows)
    tests=pd.DataFrame(test_rows); tests['p_holm_sensitivity_family']=np.nan
    tests['multiplicity_family']='Secondary/descriptive; nominal test'
    profile_primary=tests.family.eq('PROFILE')&tests.test.isin(['time_x_network','time_x_network_x_state'])
    tests.loc[profile_primary,'p_holm_sensitivity_family']=holm_adjust(tests.loc[profile_primary,'p_unadjusted'])
    tests.loc[profile_primary,'multiplicity_family']='Six all-network normalization sensitivity tests (three representations x two profile tests)'
    common_primary=tests.family.eq('COMMON')&tests.test.eq('all_four_common_level_slopes_zero')
    tests.loc[common_primary,'p_holm_sensitivity_family']=holm_adjust(tests.loc[common_primary,'p_unadjusted'])
    tests.loc[common_primary,'multiplicity_family']='Three common-level normalization sensitivity omnibus tests'
    write_table(out,'21_omnibus_tests.csv',tests)
    write_table(out,'22_paired_representation_differences.csv',difference_rows)
    write_table(out,'23_fixed_weight_identities.csv',identity_rows)
    # Signal on matched rows can differ due to sample restriction alone.
    original_reference=read_table(out/'tables/15_original_signal_fit_reproduction.csv')
    original_reference['reconstructed_slope_per_5y']=pd.to_numeric(original_reference['reconstructed_slope_per_5y'],errors='raise')
    matched_compare=[]
    for _,row in slopes.loc[slopes.component.eq('SIGNAL')&slopes.target.isin(['COMMON_LEVEL','NETWORK_RELATIVE'])].iterrows():
        network='COMMON' if row.family=='COMMON' else row.network
        old=original_reference.loc[original_reference.family.eq(row.family)&original_reference.state.eq(row.state)&original_reference.network.eq(network)].iloc[0]
        matched_compare.append({'family':row.family,'network':network,'state':row.state,'original_signal_slope_per_5y':old.reconstructed_slope_per_5y,
            'matched_signal_slope_per_5y':row.slope_per_5y,'sample_restriction_slope_difference':row.slope_per_5y-old.reconstructed_slope_per_5y,
            'same_observation_keys':same,'inference':'Sample-change descriptive comparison; not an independent-samples contrast'})
    write_table(out,'24_sample_restriction_comparison.csv',matched_compare)
    metadata_public=[{key:value for key,value in m.items() if key!='design_names'} for m in metadata]
    save_json(out/'model_specifications.json',metadata_public)
    # 40 cells use uncentered network Z, each estimated with the PROFILE row weights.
    cell=slopes.loc[slopes.target.eq('NETWORK_LOG_LEVEL')].pivot(index=['state','network'],columns='component',values='slope_per_5y').reset_index()
    cell.columns.name=None
    cell['identity_error']=cell['TOTAL']-cell['SIGNAL']-cell['ATTENUATION_H']
    cell['interpretation']='Descriptive; shared-denominator algebra and dependent cells prevent a bias test'
    write_table(out,'25_network_state_slope_scatter.csv',cell)
    correlations=[]
    for state in ['ALL',*STATES]:
        subset=cell if state=='ALL' else cell.loc[cell.state.eq(state)]
        h=subset.ATTENUATION_H.to_numpy(float); z=subset.SIGNAL.to_numpy(float)
        rho=float(spearmanr(h,z)[0]) if len(subset)>2 and np.ptp(h)>0 and np.ptp(z)>0 else np.nan
        correlations.append({'state':state,'cell_n':len(subset),'spearman_descriptive_only':rho,
                             'warning':'No p-value: algebraically coupled and correlated estimates; this association cannot establish bias'})
    write_table(out,'26_descriptive_slope_associations.csv',correlations)
    longitudinal_figures(out,slopes,cell,networks)
    result={'stage':'longitudinal','original_signal_rows':len(original),'matched_rows':len(matched),
            'original_signal_participants':original.id.nunique(),'matched_participants':matched.id.nunique(),
            'same_original_and_matched_rows':same,'original_row_sha256':original_hash,'matched_row_sha256':matched_hash,
            'baseline_age_center':center,'all_frozen_cohort_and_signal_fit_reproduction_passed':True,
            'all_observation_and_fixed_weight_identities_passed':True,'construction_interpretation':'CONDITIONAL',
            'simulation_run':False,'forty_cell_scatter_is_bias_test':False,
            'FP_inference_role':'Previously selected normalization sensitivity; not new independent confirmation'}
    save_json(out/'longitudinal_summary.json',result)
    return result
def finish_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def diagnostic_figures(out,matrices,ratios):
    if not ratios.empty:
        labels=ratios.diagnostic_session.drop_duplicates().tolist()
        fig,ax=plt.subplots(figsize=(9,3.6))
        for kind,color,offset in [('SIGNAL_NORMALIZED',COLORS['SIGNAL'],-.16),('CANDIDATE_TOTAL_NORMALIZED',COLORS['TOTAL'],.16)]:
            sub=ratios.loc[ratios.representation.eq(kind)].set_index('diagnostic_session').reindex(labels)
            ax.bar(np.arange(len(labels))+offset,100*sub.fraction_abs_gt_1_01.to_numpy(float),width=.3,color=color,
                   label='Signal denominator' if kind=='SIGNAL_NORMALIZED' else 'Candidate total denominator')
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels,rotation=45,ha='right')
        ax.set_ylabel('Parcel pairs outside ±1.01 (%)'); ax.legend(frameon=False)
        ax.set_title('Fixed diagnostic sample; not cohort prevalence',fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); finish_figure(fig,out/'figures/01_pairwise_bounds')
    if not matrices.empty and 'eigenvalue_min_over_spectral_scale' in matrices:
        valid=matrices.loc[matrices.eigenvalue_min_over_spectral_scale.notna()]
        if valid.empty: return
        labels=valid.diagnostic_session.drop_duplicates().tolist()
        fig,ax=plt.subplots(figsize=(9,3.6))
        for kind,color in [('RELEASED_SIGNAL_DIAGONAL',COLORS['SIGNAL']),('CANDIDATE_TOTAL',COLORS['TOTAL'])]:
            sub=valid.loc[valid.matrix.eq(kind)].set_index('diagnostic_session').reindex(labels)
            ax.plot(np.arange(len(labels)),sub.eigenvalue_min_over_spectral_scale,marker='o',lw=1,color=color,label=kind.replace('_',' ').title())
        ax.axhline(0,color='.4',lw=.8); ax.set_ylabel('Minimum eigenvalue / spectral scale')
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels,rotation=45,ha='right')
        ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); finish_figure(fig,out/'figures/02_matrix_spectrum')


def longitudinal_figures(out,slopes,cell,networks):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,target in zip(axes,['COMMON_LEVEL','FP_RELATIVE']):
        for component,offset in zip(COMPONENTS,[-.18,0,.18]):
            sub=slopes.loc[slopes.target.eq(target)&slopes.component.eq(component)].set_index('state').reindex(STATES)
            estimates=sub.slope_per_5y.to_numpy(float); se=sub.se.to_numpy(float)
            ax.errorbar(np.arange(4)+offset,estimates,yerr=1.96*se,fmt='o',color=COLORS[component],capsize=2,
                        label={'SIGNAL':'Signal-normalized','ATTENUATION_H':'Attenuation H','TOTAL':'Candidate total-normalized'}[component])
        ax.axhline(0,color='.4',lw=.8); ax.set_xticks(np.arange(4)); ax.set_xticklabels(STATES,rotation=20)
        ax.set_title('Common level' if target=='COMMON_LEVEL' else 'FP − mean of other nine')
        ax.set_ylabel('Log slope per 5 years (pointwise 95% CI)'); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8); fig.tight_layout()
    finish_figure(fig,out/'figures/03_common_and_FP_decomposition')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for ax,state in zip(axes.flat,STATES):
        for component,offset in zip(['SIGNAL','TOTAL'],[-.12,.12]):
            sub=slopes.loc[slopes.target.eq('NETWORK_RELATIVE')&slopes.state.eq(state)&slopes.component.eq(component)].set_index('network').reindex(networks)
            ax.errorbar(np.arange(len(networks))+offset,sub.slope_per_5y,yerr=1.96*sub.se,fmt='o',ms=4,color=COLORS[component],capsize=2,label=component.title())
        ax.axhline(0,color='.4',lw=.8); ax.set_title(state); ax.set_xticks(np.arange(len(networks)))
        ax.set_xticklabels(networks,rotation=65,ha='right',fontsize=8); ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Centered log slope per 5 years')
    axes[0,0].legend(frameon=False); fig.suptitle('Same observations and profile weights; pointwise 95% CI')
    fig.tight_layout(); finish_figure(fig,out/'figures/04_network_relative_profiles')
    fig,axes=plt.subplots(1,4,figsize=(12,3),sharex=True,sharey=True)
    for ax,state in zip(axes,STATES):
        sub=cell.loc[cell.state.eq(state)]
        colors=['#882D9C' if n=='Frontoparietal' else '.5' for n in sub.network]
        ax.scatter(sub.ATTENUATION_H,sub.SIGNAL,c=colors,s=28)
        ax.axhline(0,color='.7',lw=.7); ax.axvline(0,color='.7',lw=.7)
        ax.set_title(state); ax.set_xlabel('H slope / 5 years')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Signal-normalized log slope / 5 years')
    fig.suptitle('Descriptive association; algebraic coupling prevents a bias test',fontsize=11)
    fig.tight_layout(); finish_figure(fig,out/'figures/05_descriptive_slope_scatter')


def write_report(args,out,summaries):
    parts=['# Noise-amplitude and normalization analysis',
        '', 'The reconstructed total quantities are candidates under matched SD units and an additive/orthogonal noise model. Passing bounds and PSD checks supports this interpretation without proving the release algorithm.',
        '', 'Measurement source: '+('document supplied; see analysis_contract.json' if args.measurement_source else 'not verified; all total/noise interpretations remain conditional.'),
        '', '## Diagnostic checks']
    diagnostic=summaries.get('diagnostic')
    if diagnostic:
        matrix=pd.read_csv(out/'tables/02_matrix_structure.csv'); ratio=pd.read_csv(out/'tables/04_pairwise_normalization.csv')
        parts.append(f"Fixed diagnostic sample: {diagnostic['diagnostic_sessions']} sessions. These labels are examples, not visit order, and percentages are not cohort prevalence.")
        if 'psd_within_tolerance' in matrix:
            for kind in ['RELEASED_SIGNAL_DIAGONAL','CANDIDATE_TOTAL']:
                sub=matrix.loc[matrix.matrix.eq(kind)]
                checked=sub.loc[sub.psd_within_tolerance.notna()]
                passed=checked.psd_within_tolerance.astype(str).str.lower().eq('true').sum()
                parts.append(f'- {kind}: {passed}/{len(checked)} checked matrices PSD within the stated tolerance; {len(sub)-len(checked)} could not be assessed.')
        parts.extend(['','Eigenvalues, symmetry errors, and separate numerical/assumed CSV-rounding tolerances are reported. Material asymmetry is not silently repaired. No entries are clipped or projected onto the PSD cone.',
            '', 'Optional released-correlation comparison: '+diagnostic['full_correlation_comparison']+'.'])
    else: parts.append('Diagnostic stage was not run in this invocation.')
    parts.extend(['','## Longitudinal sensitivity'])
    longitudinal=summaries.get('longitudinal')
    if longitudinal:
        parts.append(f"Original signal cohort: {longitudinal['original_signal_participants']} participants / {longitudinal['original_signal_rows']} session-state rows. Noise-matched cohort: {longitudinal['matched_participants']} participants / {longitudinal['matched_rows']} rows.")
        parts.extend(['','Frozen original cohort and common/profile slopes reproduced. All observation, coefficient, influence, and covariance identities passed.',
            '', 'Within each family, all three components use identical observations, design, and working covariance frozen from its matched signal fit. FP is derived from the same profile fit. Joint pedigree influences provide paired uncertainty.',
            '', 'Table 20 reports log slopes, pointwise and simultaneous CIs. Table 21 retains all-network tests as the main sensitivity hierarchy. FP tests are selected sensitivities, with no adjustment for its prior selection from ten networks. Table 24 separates sample restriction from representation changes.',
            '', 'The 40-cell scatter uses uncentered network log slopes under the profile working covariance. It is descriptive: beta_SIGNAL = beta_TOTAL − beta_H creates algebraic dependence, and cells share participants, networks and states. A negative relationship cannot establish estimation bias.',
            '', 'H is log(P_signal/P_total), a ratio of mean pair-amplitude products. It is not the mean parcel signal-variance fraction. Non-significant H does not demonstrate negligible contribution; assess its interval and the paired representation differences.'])
    else: parts.append('Longitudinal stage was not run in this invocation.')
    (out/'noise_normalization_report.md').write_text('\n'.join(parts)+'\n',encoding='utf-8')


def main():
    args=parse_args(); out=args.output_dir
    for sub in ['tables','figures','private']: (out/sub).mkdir(parents=True,exist_ok=True)
    # Clear only this script's requested-stage products; retain reusable checkpoints.
    # A failed rerun must not leave older slope tables looking current.
    prefixes = ([f'{i:02d}_' for i in range(1,10)] if args.stage in ['all','diagnostic'] else [])
    prefixes += ([f'{i:02d}_' for i in list(range(10,16))+list(range(20,27))] if args.stage in ['all','longitudinal'] else [])
    for file in (out/'tables').glob('*.csv'):
        if any(file.name.startswith(prefix) for prefix in prefixes): file.unlink()
    figprefixes = (['01_','02_'] if args.stage in ['all','diagnostic'] else [])
    figprefixes += (['03_','04_','05_'] if args.stage in ['all','longitudinal'] else [])
    for file in (out/'figures').iterdir():
        if file.is_file() and any(file.name.startswith(prefix) for prefix in figprefixes): file.unlink()
    owned = ['noise_normalization_summary.json','noise_normalization_report.md','run_manifest.json']
    if args.stage in ['all','diagnostic']: owned += ['diagnostic_summary.json']
    if args.stage in ['all','longitudinal']: owned += ['longitudinal_summary.json','model_specifications.json']
    for name in owned: (out/name).unlink(missing_ok=True)
    # Cache directory can contain restricted identifiers; summaries/figures do not.
    try: os.chmod(out/'private',0o700)
    except OSError: pass
    source_path,manifest=resolve_node_manifest(args.data_dir,args.results_dir)
    parcel_map,selection=select_cortical_parcels(manifest)
    write_table(out,'00_parcel_selection.csv',selection)
    if len(parcel_map)!=EXPECTED_PARCELS:
        raise ValueError(f'Expected {EXPECTED_PARCELS} cortical parcels; found {len(parcel_map)} among {len(manifest)} manifest entries')
    networks=load_eligible_networks(args.results_dir)
    if len(set(networks))!=10 or any(n not in CANONICAL_NETWORKS for n in networks): raise ValueError('Invalid frozen ten-network list')
    if 'Frontoparietal' not in networks or 'Default' not in networks: raise ValueError('Required continuity network missing')
    contract={'version':VERSION,'timestamp_utc':utc_now(),'data_dir':args.data_dir,'results_dir':args.results_dir,'output_dir':out,
        'stage':args.stage,'states':STATES,'network_order':networks,'selected_cortical_parcels':len(parcel_map),
        'amplitude_units':args.amplitude_units,'noise_decomposition':args.noise_decomposition,
        'measurement_source':args.measurement_source,'measurement_source_sha256':sha256_file(args.measurement_source) if args.measurement_source else None,
        'correlation_semantics':args.correlation_semantics,'correlation_source':args.correlation_source,
        'correlation_source_sha256':sha256_file(args.correlation_source) if args.correlation_source else None,
        'construction_is_uniquely_identified':False,'candidate_t':'sqrt(a^2+n^2)','candidate_C_total':'C+diag(n^2)',
        'common_level':'mean(log C - log P) across fixed ten networks',
        'relative_profile':'network log quantity minus its ten-network mean',
        'normalization_components':COMPONENTS,'H':'log P_signal - log P_total; ratio of means, not mean signal fraction',
        'matched_support':'All parcels in each eligible network finite in both amplitude products; no changing subset',
        'reference_weights':'Matched SIGNAL separately for common and profile families; predetermined, frozen across components',
        'no_clipping_or_PSD_projection':True,'scatter_is_bias_test':False,'FP_selected_sensitivity':True,
        'psd_rtol':args.psd_rtol,'csv_entry_rtol_assumed':args.csv_entry_rtol,'csv_entry_atol_assumed':args.csv_entry_atol,
        'symmetry_rtol':args.symmetry_rtol,'symmetry_atol':args.symmetry_atol,
        'reproduction_tol':args.reproduction_tol,'identity_tol':args.identity_tol,
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__}}
    save_json(out/'analysis_contract.json',contract)
    archives={'amplitude':resolve_unique_files(args.data_dir,'*fMRIAmplitudes.zip',args.results_dir),
              'covariance':resolve_unique_files(args.data_dir,'*FullCovarianceConnectivity.zip',args.results_dir),
              'correlation':resolve_unique_files(args.data_dir,'*FullCorrelationConnectivity.zip',args.results_dir)}
    areas=parcel_map['Brain Area'].tolist(); amplitudes={}; inventory=[]
    for state in STATES:
        path=archives['amplitude'][state]
        if path is None: raise FileNotFoundError('Missing amplitude ZIP for '+state)
        print('Load aligned signal/noise amplitudes: '+state,flush=True)
        a,amember=load_amplitude_product(path,areas,'FullAmplitudes')
        n,nmember=load_amplitude_product(path,areas,'NoiseAmplitudes')
        amplitudes[state]={'signal':a,'noise':n}
        for product,mapping in archives.items():
            p=mapping[state]
            inventory.append({'state':state,'product':product,'available':p is not None,
                              **(archive_signature(p,args.hash_large_archives) if p else {}),
                              'signal_member':amember if product=='amplitude' else '',
                              'noise_member':nmember if product=='amplitude' else ''})
    write_table(out,'00_product_inventory.csv',inventory)
    summaries={}
    try:
        if args.stage in ['all','diagnostic']: summaries['diagnostic']=diagnostic_stage(args,parcel_map,networks,amplitudes,archives,out)
        if args.stage in ['all','longitudinal']: summaries['longitudinal']=longitudinal_stage(args,parcel_map,networks,amplitudes,out)
        write_report(args,out,summaries)
        save_json(out/'noise_normalization_summary.json',{'status':'COMPLETED','stages':summaries,
                  'construction_interpretation':'Conditional candidate reconstruction; structural tests do not prove algorithm',
                  'all_network_inference_primary':True,'FP_is_selected_sensitivity':True,'simulation_run':False})
    except Exception as exc:
        save_json(out/'noise_normalization_summary.json',{'status':'TECHNICAL_OR_REPRODUCTION_FAILURE',
                  'error_type':type(exc).__name__,'message':str(exc),'completed_stages':list(summaries),
                  'matrix_indefiniteness_alone_is_failure':False})
        raise
    finally:
        generated=[p for p in out.rglob('*') if p.is_file() and 'private' not in p.relative_to(out).parts and p.name!='run_manifest.json']
        save_json(out/'run_manifest.json',{'version':VERSION,'timestamp_utc':utc_now(),'script':str(Path(__file__).resolve()),
            'script_sha256':sha256_file(Path(__file__)),'command_arguments':vars(args),
            'files':[{'path':str(p.relative_to(out)),'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(generated)],
            'private_directory':'Contains local-only diagnostic/reference caches; exclude from shared manuscript exports'})
    print('Completed. Results: '+str(out),flush=True)
    print('Completed. Results: '+str(out), flush=True)


if __name__=='__main__':
    main()
