#!/usr/bin/env python3
"""Characterize site-specific shape of the four-state network profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
SITES = ["MGH", "UCLA", "UMinn", "WashU"]
NETWORKS = [
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
]
FP_NETWORK = "Frontoparietal"
EXPECTED_PARTICIPANTS = 852
EXPECTED_SESSIONS = 2_220
EXPECTED_ROWS = 8_727
EXPECTED_COGNITION_INCLUDED = 746
BOOTSTRAP_SEED = 20_260_915
ALPHA = 0.05
TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Local multimodal project root containing fMRI caches and archives.",
    )
    parser.add_argument(
        "--subject-table",
        type=Path,
        required=True,
        help="Longitudinal subject-event CSV, Parquet, XLS, or XLSX file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Local results root containing outputs from analyses 02, 03, and 05.",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of analysis/).",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=4_999,
        help="Gaussian pedigree-multiplier draws (paper run: 4999).",
    )
    parser.add_argument(
        "--bootstrap-chunk",
        type=int,
        default=500,
        help="Number of coefficient draws processed per chunk.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Development run with 999 draws; not for manuscript results.",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Pass --force-refit to analysis 03 instead of using compatible checkpoints.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def display(value) -> None:
    print(value)


def counts_json(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def normalize_category(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper().fillna("MISSING")
    return out.replace({"": "MISSING", "<NA>": "MISSING", "NAN": "MISSING"})


def participant_origin(identifier: object) -> str:
    value = str(identifier).strip().upper()
    if value.startswith("HCA"):
        return "HCA"
    if value.startswith("AABC"):
        return "AABC"
    return "OTHER_OR_UNKNOWN"


def prefix_label_agreement(origin: str, study: str) -> str:
    if origin in {"HCA", "AABC"} and study in {"HCA", "AABC"}:
        return "MATCH" if origin == study else "MISMATCH"
    return "NOT_COMPARABLE"


def append_count_rows(
    rows: list[dict],
    frame: pd.DataFrame,
    population: str,
    definition: str,
    column: str,
    unit: str,
) -> None:
    for category, count in frame[column].value_counts(dropna=False).sort_index().items():
        rows.append(
            {
                "population": population,
                "definition": definition,
                "category": str(category),
                "unit": unit,
                "count": int(count),
            }
        )


def row_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt(np.sum(xc**2, axis=-1) * np.sum(yc**2, axis=-1))
    return np.divide(
        np.sum(xc * yc, axis=-1),
        denominator,
        out=np.full(np.broadcast_shapes(x.shape[:-1], y.shape[:-1]), np.nan),
        where=denominator > 1e-14,
    )


def row_cosine(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    denominator = np.sqrt(np.sum(x**2, axis=-1) * np.sum(y**2, axis=-1))
    return np.divide(
        np.sum(x * y, axis=-1),
        denominator,
        out=np.full(np.broadcast_shapes(x.shape[:-1], y.shape[:-1]), np.nan),
        where=denominator > 1e-14,
    )


def simultaneous_summary(labels, point, draws, transform="identity") -> pd.DataFrame:
    point = np.asarray(point, float)
    draws = np.asarray(draws, float)
    if draws.ndim != 2 or draws.shape[1] != len(point):
        raise ValueError((draws.shape, point.shape))

    if transform == "fisher_z":
        forward = lambda value: np.arctanh(np.clip(value, -1 + 1e-12, 1 - 1e-12))
        inverse = np.tanh
    elif transform == "log":
        if np.any(point <= 0) or np.any(draws <= 0):
            raise ValueError("Log-scale family contains nonpositive values")
        forward = np.log
        inverse = np.exp
    elif transform == "identity":
        forward = lambda value: value
        inverse = lambda value: value
    else:
        raise ValueError(transform)

    transformed_point = forward(point)
    transformed_draws = forward(draws)
    finite = np.isfinite(transformed_draws).all(axis=1)
    transformed_draws = transformed_draws[finite]
    if len(transformed_draws) < 0.99 * len(draws):
        raise AssertionError(f"Too many invalid draws: {len(draws) - len(transformed_draws)}")

    standard_error = transformed_draws.std(axis=0, ddof=1)
    if np.any(~np.isfinite(standard_error)) or np.any(standard_error <= 0):
        raise AssertionError("Invalid transformed standard error")

    max_abs = np.max(
        np.abs((transformed_draws - transformed_point[None, :]) / standard_error[None, :]),
        axis=1,
    )
    critical = float(np.quantile(max_abs, 1 - ALPHA, method="higher"))
    marginal = np.quantile(draws[finite], [ALPHA / 2, 1 - ALPHA / 2], axis=0)
    simultaneous_low = inverse(transformed_point - critical * standard_error)
    simultaneous_high = inverse(transformed_point + critical * standard_error)
    return pd.DataFrame(
        {
            "label": labels,
            "estimate": point,
            "marginal_95_low": marginal[0],
            "marginal_95_high": marginal[1],
            "simultaneous_95_low": simultaneous_low,
            "simultaneous_95_high": simultaneous_high,
            "family_critical_value": critical,
            "valid_draws": int(finite.sum()),
            "transform": transform,
        }
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    subject_table = args.subject_table.expanduser().resolve()
    code_dir = args.code_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    bootstrap_b = 999 if args.fast else args.bootstrap_b
    if bootstrap_b < 100:
        raise ValueError("--bootstrap-b must be at least 100")
    if args.bootstrap_chunk < 1:
        raise ValueError("--bootstrap-chunk must be positive")

    source_script = code_dir / "analysis/03_fp_specificity_and_robustness.py"
    source_checkpoint = (
        results_dir
        / "02_primary_profile_and_measurement/checkpoints/profile__amplitude_standardized.json"
    )
    source_slopes = (
        results_dir
        / "02_primary_profile_and_measurement/tables/network_profile_state_slopes.csv"
    )
    source_tests = (
        results_dir / "02_primary_profile_and_measurement/tables/profile_omnibus_tests.csv"
    )
    cognition_flow_path = results_dir / "05_cognition/tables/cognition_selection_flow.csv"
    scalar_site_slopes_path = (
        results_dir
        / "03_fp_specificity_and_robustness/tables/fp_relative_site_state_slopes.csv"
    )
    output_dir = results_dir / "07_site_profile_shape_diagnostic"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in [output_dir, table_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    required_paths = [
        data_dir,
        subject_table,
        code_dir,
        results_dir,
        source_script,
        source_checkpoint,
        source_slopes,
        source_tests,
        cognition_flow_path,
        scalar_site_slopes_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing))

    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "analysis": "site-specific shape of the amplitude-standardized ten-network longitudinal profile",
        "primary_estimand": "four site-specific equal-state ten-network slope vectors extracted from the existing pooled interaction model",
        "secondary_estimand": "site-by-state ten-network slope vectors extracted from the same model",
        "provenance_estimand": "aggregate participant-origin and event-study definitions in the profile and cognition rosters",
        "expected_cohort": {
            "participants": EXPECTED_PARTICIPANTS,
            "sessions": EXPECTED_SESSIONS,
            "session_state_rows": EXPECTED_ROWS,
            "cognition_included": EXPECTED_COGNITION_INCLUDED,
        },
        "states": STATES,
        "sites": SITES,
        "networks": NETWORKS,
        "representation": "pair-FullAmplitude-standardized covariance, centered across ten networks within visit and state",
        "time_unit": "5 years",
        "site_weighting": "site-specific contrasts from sum-coded pooled model; pooled contrast is the model's unweighted site average",
        "state_weighting": "equal 1/4 average",
        "uncertainty": {
            "method": "shared Gaussian pedigree multiplier draws from recovered coefficient influence functions",
            "draws": bootstrap_b,
            "seed": BOOTSTRAP_SEED,
            "correlation_scale": "Fisher z for interval construction only",
            "families": [
                "10 equal-state correlations: 4 site-to-other-three plus 6 site-pair",
                "4 equal-state site magnitude ratios",
                "4 equal-state site FP-relative slopes",
                "16 state-resolved site-to-other-three correlations",
                "16 state-resolved magnitude ratios",
                "16 state-resolved FP-relative slopes",
            ],
        },
        "source_files": {
            str(path): sha256_file(path)
            for path in [
                subject_table,
                source_script,
                source_checkpoint,
                source_slopes,
                source_tests,
                cognition_flow_path,
                scalar_site_slopes_path,
            ]
        },
    }
    contract_path = output_dir / "00_analysis_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    # Recover the analysis frame and pedigree influence contributions.
    source_argv = [
        str(source_script),
        "--data-dir",
        str(data_dir),
        "--subject-table",
        str(subject_table),
        "--results-dir",
        str(results_dir),
        "--bootstrap-b",
        str(bootstrap_b),
    ]
    if args.fast:
        source_argv.append("--fast")
    if args.force_refit:
        source_argv.append("--force-refit")

    old_argv = sys.argv[:]
    try:
        sys.argv = source_argv
        source = runpy.run_path(str(source_script), run_name="__site_shape_source__")
    finally:
        sys.argv = old_argv

    fit_frame = source["fit_frame"].copy()
    design = source["design"]
    beta = np.asarray(source["beta_saved"], float)
    covariance = np.asarray(source["cov_saved"], float)
    influence_vectors = source["if_vectors"]
    network_vector = source["network_vector"]
    site_state_slope_vector = source["site_state_slope_vector"]
    wald_matrix = source["wald_matrix"]

    if source["ELIGIBLE_NETWORKS"] != NETWORKS:
        raise AssertionError("Eligible-network order differs from the frozen analysis")
    if source["STATES"] != STATES:
        raise AssertionError("State order differs from the frozen analysis")
    if design.site_levels != SITES:
        raise AssertionError((design.site_levels, SITES))
    if fit_frame["id"].nunique() != EXPECTED_PARTICIPANTS:
        raise AssertionError("Frozen participant count changed")
    if fit_frame["id_event"].nunique() != EXPECTED_SESSIONS:
        raise AssertionError("Frozen session count changed")
    if len(fit_frame) != EXPECTED_ROWS:
        raise AssertionError("Frozen session-state row count changed")
    if int(source["fit_with_influence"]["participant_n"]) != EXPECTED_PARTICIPANTS:
        raise AssertionError("Recovered influence fit has the wrong participant count")
    if float(source["beta_max_abs_diff"]) > source["SOURCE_ESTIMATE_TOL"]:
        raise AssertionError("Recovered coefficients do not match the frozen checkpoint")
    if float(source["cov_max_abs_diff"]) > source["SOURCE_COV_TOL"]:
        raise AssertionError("Recovered covariance does not match the frozen checkpoint")
    if float(source["influence_cov_error"]) > 1e-10:
        raise AssertionError("Recovered influences do not reconstruct the covariance")

    profile_tests = pd.read_csv(source_tests)
    tests = profile_tests.loc[
        profile_tests["scenario"].eq("AMPLITUDE_STANDARDIZED")
    ].set_index("test")
    expected_p = {
        "time_x_network": 1.4728429519401009e-06,
        "time_x_network_x_state": 2.5476996273962662e-12,
        "site_x_time_x_network": 8.275523542268913e-04,
        "site_x_time_x_state_x_network": 7.545736829063459e-13,
    }
    for name, target in expected_p.items():
        observed = float(tests.loc[name, "p"])
        if not np.isclose(observed, target, rtol=1e-8, atol=1e-15):
            raise AssertionError(
                f"{name}: expected {target:.16g}, observed {observed:.16g}"
            )

    gate_rows = [
        {
            "gate": "participants",
            "observed": fit_frame["id"].nunique(),
            "expected": EXPECTED_PARTICIPANTS,
        },
        {
            "gate": "sessions",
            "observed": fit_frame["id_event"].nunique(),
            "expected": EXPECTED_SESSIONS,
        },
        {
            "gate": "session_state_rows",
            "observed": len(fit_frame),
            "expected": EXPECTED_ROWS,
        },
        *[
            {
                "gate": name,
                "observed": float(tests.loc[name, "p"]),
                "expected": target,
            }
            for name, target in expected_p.items()
        ],
    ]

    # Site composition and study provenance on the exact frozen profile rows.
    event_columns = [
        "id",
        "id_event",
        "pedigree_group",
        "site",
        "scanner",
        "site_scanner",
        "sex",
        "study",
        "baseline_age",
        "time_years",
    ]
    for column in event_columns:
        within_event_n = fit_frame.groupby("id_event", observed=True)[column].nunique(
            dropna=False
        )
        if within_event_n.gt(1).any():
            raise AssertionError(f"{column} differs across states within an event")

    events = fit_frame[event_columns].drop_duplicates("id_event").copy()
    events["id"] = events["id"].astype(str).str.strip()
    events["study_event"] = normalize_category(events["study"])
    events["id_prefix_origin"] = events["id"].map(participant_origin)
    events["prefix_event_study_agreement"] = [
        prefix_label_agreement(origin, study)
        for origin, study in zip(events["id_prefix_origin"], events["study_event"])
    ]
    events = events.sort_values(["id", "time_years", "id_event"]).reset_index(drop=True)

    person_core = events.groupby("id", as_index=False).agg(
        site=("site", "first"),
        site_n=("site", "nunique"),
        pedigree_group=("pedigree_group", "first"),
        sessions=("id_event", "nunique"),
        scanner_n=("scanner", "nunique"),
        baseline_age=("baseline_age", "first"),
        followup_years=("time_years", "max"),
        sex=("sex", "first"),
        id_prefix_origin=("id_prefix_origin", "first"),
        first_event_study=("study_event", "first"),
        last_event_study=("study_event", "last"),
        study_label_n=("study_event", "nunique"),
        ever_study_set=(
            "study_event",
            lambda values: "|".join(sorted(set(values.astype(str)))),
        ),
    )
    person_core["study_label_changed"] = np.where(
        person_core["study_label_n"].gt(1), "CHANGED", "STABLE"
    )
    person_core["prefix_first_study_agreement"] = [
        prefix_label_agreement(origin, study)
        for origin, study in zip(
            person_core["id_prefix_origin"], person_core["first_event_study"]
        )
    ]
    if not person_core["site_n"].eq(1).all():
        raise AssertionError("A participant spans more than one site")

    pedigree_site_n = events.groupby("pedigree_group", observed=True)["site"].nunique()
    cross_site_pedigrees = set(pedigree_site_n.index[pedigree_site_n.gt(1)])

    composition_rows = []
    for site in SITES:
        people = person_core.loc[person_core["site"].astype(str).eq(site)].copy()
        site_events = events.loc[events["site"].astype(str).eq(site)].copy()
        site_frame = fit_frame.loc[fit_frame["site"].astype(str).eq(site)].copy()
        composition_rows.append(
            {
                "site": site,
                "participants": people["id"].nunique(),
                "pedigree_clusters": people["pedigree_group"].nunique(),
                "cross_site_pedigree_clusters": people.loc[
                    people["pedigree_group"].isin(cross_site_pedigrees),
                    "pedigree_group",
                ].nunique(),
                "sessions": site_events["id_event"].nunique(),
                "session_state_rows": len(site_frame),
                "participants_3plus_sessions": int(people["sessions"].ge(3).sum()),
                "baseline_age_mean": people["baseline_age"].mean(),
                "baseline_age_sd": people["baseline_age"].std(ddof=1),
                "baseline_age_min": people["baseline_age"].min(),
                "baseline_age_max": people["baseline_age"].max(),
                "followup_median": people["followup_years"].median(),
                "followup_q1": people["followup_years"].quantile(0.25),
                "followup_q3": people["followup_years"].quantile(0.75),
                "same_scanner_participants": int(people["scanner_n"].eq(1).sum()),
                "scanner_changed_participants": int(people["scanner_n"].gt(1).sum()),
                "sex_counts": counts_json(people["sex"]),
                "id_prefix_origin_counts": counts_json(people["id_prefix_origin"]),
                "first_event_study_counts": counts_json(people["first_event_study"]),
                "ever_study_set_counts": counts_json(people["ever_study_set"]),
            }
        )
    site_composition = pd.DataFrame(composition_rows)
    site_state_coverage = (
        fit_frame.groupby(["site", "state"], observed=True)
        .agg(
            participants=("id", "nunique"),
            sessions=("id_event", "nunique"),
            rows=("id_event", "size"),
        )
        .reset_index()
    )
    site_scanner_composition = (
        events.groupby(["site", "scanner", "site_scanner"], observed=True)
        .agg(participants=("id", "nunique"), sessions=("id_event", "nunique"))
        .reset_index()
    )

    provenance_rows: list[dict] = []
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "participant ID prefix",
        "id_prefix_origin",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "study label at first eligible event",
        "first_event_study",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "study label at last eligible event",
        "last_event_study",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "set of study labels across eligible events",
        "ever_study_set",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "whether event-level study label changes within participant",
        "study_label_changed",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        person_core,
        "FROZEN_PROFILE_852",
        "participant ID prefix versus first event study label",
        "prefix_first_study_agreement",
        "participants",
    )
    append_count_rows(
        provenance_rows,
        events,
        "FROZEN_PROFILE_2220_SESSIONS",
        "event-level study label",
        "study_event",
        "sessions",
    )
    append_count_rows(
        provenance_rows,
        events,
        "FROZEN_PROFILE_2220_SESSIONS",
        "participant ID prefix versus event-level study label",
        "prefix_event_study_agreement",
        "sessions",
    )
    study_provenance = pd.DataFrame(provenance_rows)

    cognition_flow = pd.read_csv(cognition_flow_path)
    required_cognition_columns = {"id", "selection_category"}
    if not required_cognition_columns.issubset(cognition_flow.columns):
        raise KeyError(
            f"{cognition_flow_path} lacks {sorted(required_cognition_columns - set(cognition_flow.columns))}"
        )
    cognition_flow = cognition_flow[["id", "selection_category"]].copy()
    cognition_flow["id"] = cognition_flow["id"].astype(str).str.strip()
    if cognition_flow["id"].duplicated().any():
        raise AssertionError("Cognition selection flow is not one row per participant")
    if set(cognition_flow["id"]) != set(person_core["id"]):
        raise AssertionError("Cognition selection roster differs from the frozen profile roster")
    cognition_flow["cognition_group"] = np.where(
        cognition_flow["selection_category"].eq("INCLUDED"),
        "INCLUDED_2PLUS_PAIRED",
        "EXCLUDED_LT2_PAIRED",
    )
    cognition_people = person_core.merge(
        cognition_flow, on="id", how="left", validate="one_to_one"
    )
    cognition_provenance_rows: list[dict] = []
    for cognition_group, group_frame in cognition_people.groupby(
        "cognition_group", observed=True
    ):
        for definition, column in [
            ("participant ID prefix", "id_prefix_origin"),
            ("study label at first eligible event", "first_event_study"),
            ("study label at last eligible event", "last_event_study"),
            ("set of study labels across eligible events", "ever_study_set"),
            (
                "whether event-level study label changes within participant",
                "study_label_changed",
            ),
            (
                "participant ID prefix versus first event study label",
                "prefix_first_study_agreement",
            ),
        ]:
            append_count_rows(
                cognition_provenance_rows,
                group_frame,
                str(cognition_group),
                definition,
                column,
                "participants",
            )
    cognition_study_provenance = pd.DataFrame(cognition_provenance_rows)

    cognition_included = int(
        cognition_flow["selection_category"].eq("INCLUDED").sum()
    )
    gate_rows.extend(
        [
            {
                "gate": "site_participants_sum",
                "observed": int(site_composition["participants"].sum()),
                "expected": EXPECTED_PARTICIPANTS,
            },
            {
                "gate": "study_provenance_participants",
                "observed": person_core["id"].nunique(),
                "expected": EXPECTED_PARTICIPANTS,
            },
            {
                "gate": "cognition_roster_participants",
                "observed": cognition_flow["id"].nunique(),
                "expected": EXPECTED_PARTICIPANTS,
            },
            {
                "gate": "cognition_included_participants",
                "observed": cognition_included,
                "expected": EXPECTED_COGNITION_INCLUDED,
            },
            {
                "gate": "cognition_excluded_participants",
                "observed": len(cognition_flow) - cognition_included,
                "expected": EXPECTED_PARTICIPANTS - EXPECTED_COGNITION_INCLUDED,
            },
        ]
    )
    gate_table = pd.DataFrame(gate_rows)
    numeric_observed = pd.to_numeric(gate_table["observed"], errors="coerce")
    numeric_expected = pd.to_numeric(gate_table["expected"], errors="coerce")
    gate_table["pass"] = np.isclose(
        numeric_observed, numeric_expected, rtol=1e-8, atol=1e-15
    )
    if not gate_table["pass"].all():
        display(gate_table.loc[~gate_table["pass"]])
        raise AssertionError("One or more reproduction gates failed")

    if "id" in study_provenance.columns or "id" in cognition_study_provenance.columns:
        raise AssertionError("A participant identifier would be exported")

    # Extract site-by-state and equal-state profile contrasts.
    coefficient_n = len(beta)
    network_n = len(NETWORKS)
    network_index = {network: index for index, network in enumerate(NETWORKS)}
    site_state_contrasts = np.empty(
        (len(SITES), len(STATES), network_n, coefficient_n), float
    )
    for site_index, site in enumerate(SITES):
        for state_index, state in enumerate(STATES):
            design_vector = site_state_slope_vector(design, site, state)
            for index, _network in enumerate(NETWORKS):
                weight = np.zeros(network_n)
                weight[index] = 1.0
                site_state_contrasts[site_index, state_index, index] = network_vector(
                    weight, design_vector
                )

    pooled_state_contrasts = np.empty(
        (len(STATES), network_n, coefficient_n), float
    )
    for state_index, state in enumerate(STATES):
        design_vector = design.state_slope_vector(state)
        for index, _network in enumerate(NETWORKS):
            weight = np.zeros(network_n)
            weight[index] = 1.0
            pooled_state_contrasts[state_index, index] = network_vector(
                weight, design_vector
            )

    site_equal_contrasts = site_state_contrasts.mean(axis=1)
    pooled_equal_contrasts = pooled_state_contrasts.mean(axis=0)
    site_state_point = np.einsum(
        "jskp,p->jsk", site_state_contrasts, beta, optimize=True
    )
    pooled_state_point = np.einsum(
        "skp,p->sk", pooled_state_contrasts, beta, optimize=True
    )
    site_equal_point = np.einsum(
        "jkp,p->jk", site_equal_contrasts, beta, optimize=True
    )
    pooled_equal_point = pooled_equal_contrasts @ beta

    sum_zero_error = max(
        float(np.abs(site_state_point.sum(axis=2)).max()),
        float(np.abs(site_equal_point.sum(axis=1)).max()),
        float(np.abs(pooled_state_point.sum(axis=1)).max()),
        float(abs(pooled_equal_point.sum())),
    )
    pooled_site_average_error = float(
        np.abs(site_equal_point.mean(axis=0) - pooled_equal_point).max()
    )
    if sum_zero_error > TOL:
        raise AssertionError(
            f"Profile sum-to-zero error {sum_zero_error:.3e} exceeds {TOL:.1e}"
        )
    if pooled_site_average_error > TOL:
        raise AssertionError(
            "Pooled profile is not the expected unweighted site average: "
            f"{pooled_site_average_error:.3e}"
        )

    site_state_rows = []
    for site_index, site in enumerate(SITES):
        for state_index, state in enumerate(STATES):
            for network_index_value, network in enumerate(NETWORKS):
                contrast = site_state_contrasts[
                    site_index, state_index, network_index_value
                ]
                standard_error = float(
                    np.sqrt(max(contrast @ covariance @ contrast, 0.0))
                )
                site_state_rows.append(
                    {
                        "site": site,
                        "state": state,
                        "network": network,
                        "slope_per_5y": float(
                            site_state_point[
                                site_index, state_index, network_index_value
                            ]
                        ),
                        "se": standard_error,
                    }
                )
    site_state_slopes = pd.DataFrame(site_state_rows)

    site_equal_rows = []
    for site_index, site in enumerate(SITES):
        for network_index_value, network in enumerate(NETWORKS):
            contrast = site_equal_contrasts[site_index, network_index_value]
            standard_error = float(
                np.sqrt(max(contrast @ covariance @ contrast, 0.0))
            )
            site_equal_rows.append(
                {
                    "site": site,
                    "network": network,
                    "slope_per_5y": float(
                        site_equal_point[site_index, network_index_value]
                    ),
                    "se": standard_error,
                }
            )
    site_equal_slopes = pd.DataFrame(site_equal_rows)
    pooled_equal_slopes = pd.DataFrame(
        {
            "network": NETWORKS,
            "slope_per_5y": pooled_equal_point,
            "se": [
                float(np.sqrt(max(contrast @ covariance @ contrast, 0.0)))
                for contrast in pooled_equal_contrasts
            ],
        }
    )

    # Shared Gaussian pedigree multipliers preserve all cross-site dependence.
    pedigrees = sorted(influence_vectors)
    influence_matrix = np.vstack(
        [np.asarray(influence_vectors[pedigree], float) for pedigree in pedigrees]
    )
    if influence_matrix.shape[1] != coefficient_n:
        raise AssertionError("Influence-vector width differs from coefficient count")
    covariance_reconstruction_error = float(
        np.abs(influence_matrix.T @ influence_matrix - covariance).max()
    )
    if covariance_reconstruction_error > 1e-10:
        raise AssertionError(
            "Influence covariance reconstruction error "
            f"{covariance_reconstruction_error:.3e}"
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    beta_draws = np.empty((bootstrap_b, coefficient_n), float)
    for start in range(0, bootstrap_b, args.bootstrap_chunk):
        stop = min(start + args.bootstrap_chunk, bootstrap_b)
        multipliers = rng.standard_normal((stop - start, len(pedigrees)))
        beta_draws[start:stop] = beta[None, :] + multipliers @ influence_matrix

    site_state_draws = np.einsum(
        "bp,jskp->bjsk", beta_draws, site_state_contrasts, optimize=True
    )
    pooled_state_draws = np.einsum(
        "bp,skp->bsk", beta_draws, pooled_state_contrasts, optimize=True
    )
    site_equal_draws = site_state_draws.mean(axis=2)
    pooled_equal_draws = pooled_state_draws.mean(axis=1)
    draw_sum_zero_error = max(
        float(np.abs(site_state_draws.sum(axis=3)).max()),
        float(np.abs(site_equal_draws.sum(axis=2)).max()),
    )
    if draw_sum_zero_error > 5e-9:
        raise AssertionError(
            f"Draw-level profile identity error {draw_sum_zero_error:.3e}"
        )

    # Primary correlation family: four site-to-other-three and six site-pair
    # correlations. Site-to-pooled estimates are retained only as a part-whole
    # audit and do not enter the family or figure.
    other_three_point = (
        site_equal_point.sum(axis=0, keepdims=True) - site_equal_point
    ) / (len(SITES) - 1)
    other_three_draws = (
        site_equal_draws.sum(axis=1, keepdims=True) - site_equal_draws
    ) / (len(SITES) - 1)
    site_other_three_point = row_corr(site_equal_point, other_three_point)
    site_other_three_draw = row_corr(site_equal_draws, other_three_draws)

    pair_labels = []
    pair_point = []
    pair_draw = []
    for first, second in combinations(range(len(SITES)), 2):
        pair_labels.append(f"{SITES[first]}__vs__{SITES[second]}")
        pair_point.append(
            float(row_corr(site_equal_point[first], site_equal_point[second]))
        )
        pair_draw.append(
            row_corr(
                site_equal_draws[:, first, :], site_equal_draws[:, second, :]
            )
        )
    pair_draw = np.column_stack(pair_draw)
    correlation_labels = [
        f"{site}__vs__OTHER_THREE" for site in SITES
    ] + pair_labels
    correlation_point = np.r_[site_other_three_point, pair_point]
    correlation_draw = np.column_stack([site_other_three_draw, pair_draw])
    equal_state_similarity = simultaneous_summary(
        correlation_labels, correlation_point, correlation_draw, "fisher_z"
    )
    equal_state_similarity.insert(
        0,
        "comparison_type",
        ["site_to_other_three"] * len(SITES) + ["site_pair"] * len(pair_labels),
    )

    site_pool_point = row_corr(site_equal_point, pooled_equal_point[None, :])
    partwhole_similarity_audit = pd.DataFrame(
        {
            "site": SITES,
            "correlation_with_pooled_profile": site_pool_point,
            "correlation_with_other_three_profile": site_other_three_point,
            "part_whole_dependence": True,
            "included_in_primary_similarity_family": False,
            "included_in_figure_panel_B": False,
        }
    )
    cosine_error = max(
        float(
            np.nanmax(
                np.abs(
                    row_cosine(site_equal_point, other_three_point)
                    - site_other_three_point
                )
            )
        ),
        float(
            np.nanmax(
                np.abs(
                    row_cosine(site_equal_point, pooled_equal_point[None, :])
                    - site_pool_point
                )
            )
        ),
    )
    if cosine_error > 1e-12:
        raise AssertionError(f"Pearson/cosine identity error {cosine_error:.3e}")

    site_rms_point = np.sqrt(np.mean(site_equal_point**2, axis=1))
    pooled_rms_point = float(np.sqrt(np.mean(pooled_equal_point**2)))
    site_rms_draw = np.sqrt(np.mean(site_equal_draws**2, axis=2))
    pooled_rms_draw = np.sqrt(np.mean(pooled_equal_draws**2, axis=1))
    magnitude_ratio_point = site_rms_point / pooled_rms_point
    magnitude_ratio_draw = site_rms_draw / pooled_rms_draw[:, None]
    projection_point = np.sum(
        site_equal_point * pooled_equal_point[None, :], axis=1
    ) / np.sum(pooled_equal_point**2)
    projection_draw = np.sum(
        site_equal_draws * pooled_equal_draws[:, None, :], axis=2
    ) / np.sum(pooled_equal_draws**2, axis=1)[:, None]
    magnitude_ratio_summary = simultaneous_summary(
        SITES, magnitude_ratio_point, magnitude_ratio_draw, "log"
    ).add_prefix("ratio_")
    projection_summary = simultaneous_summary(
        SITES, projection_point, projection_draw, "identity"
    ).add_prefix("projection_")
    equal_state_magnitude = pd.concat(
        [
            pd.DataFrame(
                {
                    "site": SITES,
                    "site_profile_rms": site_rms_point,
                    "pooled_profile_rms": pooled_rms_point,
                }
            ),
            magnitude_ratio_summary.drop(columns="ratio_label"),
            projection_summary.drop(columns="projection_label"),
        ],
        axis=1,
    )

    omnibus_rows = []
    for site_index, site in enumerate(SITES):
        statistic, degrees_freedom, p_value = wald_matrix(
            beta, covariance, site_equal_contrasts[site_index]
        )
        omnibus_rows.append(
            {
                "site": site,
                "descriptive_profile_wald_chi2": statistic,
                "descriptive_profile_wald_df": degrees_freedom,
                "descriptive_profile_p": p_value,
                "replication_gate": False,
            }
        )
    equal_state_magnitude = equal_state_magnitude.merge(
        pd.DataFrame(omnibus_rows), on="site", validate="one_to_one"
    )

    fp_index = network_index[FP_NETWORK]
    fp_point = (network_n / (network_n - 1.0)) * site_equal_point[:, fp_index]
    fp_draw = (network_n / (network_n - 1.0)) * site_equal_draws[:, :, fp_index]
    equal_state_fp = simultaneous_summary(
        SITES, fp_point, fp_draw, "identity"
    ).rename(
        columns={
            "label": "site",
            "estimate": "fp_relative_slope_per_5y",
        }
    )
    equal_state_fp["positive_point_estimate"] = equal_state_fp[
        "fp_relative_slope_per_5y"
    ].gt(0)
    equal_state_fp["simultaneous_interval_excludes_zero"] = (
        equal_state_fp["simultaneous_95_low"].gt(0)
        | equal_state_fp["simultaneous_95_high"].lt(0)
    )

    # Secondary state-resolved localization uses the same other-three reference
    # to avoid part-whole dependence.
    state_other_three_point = (
        site_state_point.sum(axis=0, keepdims=True) - site_state_point
    ) / (len(SITES) - 1)
    state_other_three_draws = (
        site_state_draws.sum(axis=1, keepdims=True) - site_state_draws
    ) / (len(SITES) - 1)
    state_site_point_r = row_corr(site_state_point, state_other_three_point)
    state_site_draw_r = row_corr(site_state_draws, state_other_three_draws)
    state_labels = [f"{site}__{state}" for site in SITES for state in STATES]
    state_similarity = simultaneous_summary(
        state_labels,
        state_site_point_r.reshape(-1),
        state_site_draw_r.reshape(bootstrap_b, -1),
        "fisher_z",
    ).rename(
        columns={
            "label": "site_state",
            "estimate": "r_with_other_three_state_profile",
        }
    )
    state_similarity[["site", "state"]] = state_similarity[
        "site_state"
    ].str.split("__", expand=True)

    site_state_rms_point = np.sqrt(np.mean(site_state_point**2, axis=2))
    pooled_state_rms_point = np.sqrt(np.mean(pooled_state_point**2, axis=1))
    site_state_rms_draw = np.sqrt(np.mean(site_state_draws**2, axis=3))
    pooled_state_rms_draw = np.sqrt(np.mean(pooled_state_draws**2, axis=2))
    state_ratio_point = site_state_rms_point / pooled_state_rms_point[None, :]
    state_ratio_draw = site_state_rms_draw / pooled_state_rms_draw[:, None, :]
    state_magnitude = simultaneous_summary(
        state_labels,
        state_ratio_point.reshape(-1),
        state_ratio_draw.reshape(bootstrap_b, -1),
        "log",
    ).rename(
        columns={
            "label": "site_state",
            "estimate": "profile_rms_ratio_to_pooled_state",
        }
    )
    state_magnitude[["site", "state"]] = state_magnitude[
        "site_state"
    ].str.split("__", expand=True)

    state_fp_point = (
        network_n / (network_n - 1.0)
    ) * site_state_point[:, :, fp_index]
    state_fp_draw = (
        network_n / (network_n - 1.0)
    ) * site_state_draws[:, :, :, fp_index]
    state_fp = simultaneous_summary(
        state_labels,
        state_fp_point.reshape(-1),
        state_fp_draw.reshape(bootstrap_b, -1),
        "identity",
    ).rename(
        columns={
            "label": "site_state",
            "estimate": "fp_relative_slope_per_5y",
        }
    )
    state_fp[["site", "state"]] = state_fp["site_state"].str.split(
        "__", expand=True
    )
    state_fp["positive_point_estimate"] = state_fp[
        "fp_relative_slope_per_5y"
    ].gt(0)

    state_pair_rows = []
    for state_index, state in enumerate(STATES):
        for first, second in combinations(range(len(SITES)), 2):
            state_pair_rows.append(
                {
                    "state": state,
                    "site_a": SITES[first],
                    "site_b": SITES[second],
                    "pearson_r": float(
                        row_corr(
                            site_state_point[first, state_index],
                            site_state_point[second, state_index],
                        )
                    ),
                    "ordinary_correlation_p_reported": False,
                }
            )
    state_pairwise = pd.DataFrame(state_pair_rows)

    # Validate profile-derived site-by-state FP slopes against the maintained
    # direct scalar model. Small differences are expected because the two fits
    # use different outcome/covariance parameterizations.
    scalar_site_slopes = pd.read_csv(scalar_site_slopes_path)
    scalar_site_slopes = scalar_site_slopes.loc[
        scalar_site_slopes["representation"].eq("AMPLITUDE_STANDARDIZED"),
        ["site", "state", "slope_per_5y"],
    ].rename(columns={"slope_per_5y": "direct_scalar_slope_per_5y"})
    profile_site_fp = state_fp[
        ["site", "state", "fp_relative_slope_per_5y"]
    ].rename(
        columns={
            "fp_relative_slope_per_5y": "joint_profile_derived_slope_per_5y"
        }
    )
    scalar_profile_reconciliation = scalar_site_slopes.merge(
        profile_site_fp, on=["site", "state"], validate="one_to_one"
    )
    if len(scalar_profile_reconciliation) != len(SITES) * len(STATES):
        raise AssertionError("Scalar/profile reconciliation does not contain 16 cells")
    scalar_profile_reconciliation["direct_minus_profile"] = (
        scalar_profile_reconciliation["direct_scalar_slope_per_5y"]
        - scalar_profile_reconciliation["joint_profile_derived_slope_per_5y"]
    )
    scalar_profile_reconciliation["absolute_difference"] = (
        scalar_profile_reconciliation["direct_minus_profile"].abs()
    )
    scalar_profile_max_difference = float(
        scalar_profile_reconciliation["absolute_difference"].max()
    )
    if scalar_profile_max_difference > 0.002:
        raise AssertionError(
            "Direct scalar and joint-profile site-state FP estimates differ by "
            f"{scalar_profile_max_difference:.6f}, above the frozen 0.002 audit tolerance"
        )

    # Supplementary figure.
    sns.set_theme(style="whitegrid", context="paper")
    site_colors = dict(zip(SITES, sns.color_palette("colorblind", len(SITES))))
    network_short = {
        "Visual1": "V1",
        "Visual2": "V2",
        "Somatomotor": "SM",
        "Cingulo-Opercular": "CO",
        "Dorsal-attention": "DA",
        "Language": "Lang",
        "Frontoparietal": "FP",
        "Auditory": "Aud",
        "Default": "DMN",
        "Posterior-Multimodal": "PMM",
    }
    network_order = np.argsort(pooled_equal_point)
    x_positions = np.arange(network_n)
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 8.5), constrained_layout=True
    )

    axis = axes[0, 0]
    for site_index, site in enumerate(SITES):
        axis.plot(
            x_positions,
            site_equal_point[site_index, network_order],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=site,
            color=site_colors[site],
        )
    axis.plot(
        x_positions,
        pooled_equal_point[network_order],
        color="black",
        linestyle="--",
        linewidth=2.0,
        marker="s",
        markersize=3.5,
        label="Pooled",
    )
    axis.axhline(0, color="0.55", linewidth=0.8)
    axis.set_xticks(
        x_positions,
        [network_short[NETWORKS[index]] for index in network_order],
        rotation=45,
        ha="right",
    )
    axis.set_ylabel("Equal-state slope per 5 years")
    axis.set_title("A  Site-specific equal-state profiles", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, fontsize=8)

    site_other_table = equal_state_similarity.loc[
        equal_state_similarity["comparison_type"].eq("site_to_other_three")
    ].copy()
    site_other_table["site"] = site_other_table["label"].str.replace(
        "__vs__OTHER_THREE", "", regex=False
    )
    site_other_table = site_other_table.set_index("site").loc[SITES].reset_index()
    y_positions = np.arange(len(SITES))
    axis = axes[0, 1]
    axis.errorbar(
        site_other_table["estimate"],
        y_positions,
        xerr=np.vstack(
            [
                site_other_table["estimate"]
                - site_other_table["simultaneous_95_low"],
                site_other_table["simultaneous_95_high"]
                - site_other_table["estimate"],
            ]
        ),
        fmt="o",
        color="black",
        capsize=3,
    )
    axis.axvline(0, color="0.6", linewidth=0.8)
    axis.set_yticks(y_positions, SITES)
    axis.invert_yaxis()
    axis.set_xlim(-1.02, 1.02)
    axis.set_xlabel("Correlation with mean of other three sites")
    axis.set_title("B  Equal-state shape agreement", loc="left", fontweight="bold")

    magnitude_plot = equal_state_magnitude.set_index("site").loc[SITES].reset_index()
    axis = axes[1, 0]
    axis.errorbar(
        magnitude_plot["ratio_estimate"],
        y_positions,
        xerr=np.vstack(
            [
                magnitude_plot["ratio_estimate"]
                - magnitude_plot["ratio_simultaneous_95_low"],
                magnitude_plot["ratio_simultaneous_95_high"]
                - magnitude_plot["ratio_estimate"],
            ]
        ),
        fmt="o",
        color="black",
        capsize=3,
    )
    axis.axvline(1, color="0.45", linestyle="--", linewidth=1)
    axis.set_yticks(y_positions, SITES)
    axis.invert_yaxis()
    axis.set_xlabel("Site / pooled profile RMS")
    axis.set_title("C  Equal-state profile magnitude", loc="left", fontweight="bold")

    heatmap = state_similarity.pivot(
        index="site", columns="state", values="r_with_other_three_state_profile"
    ).reindex(index=SITES, columns=STATES)
    axis = axes[1, 1]
    sns.heatmap(
        heatmap,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="vlag_r",
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title(
        "D  State-resolved agreement with other three sites",
        loc="left",
        fontweight="bold",
    )

    figure_pdf = figure_dir / "Figure_S_site_profile_heterogeneity.pdf"
    figure_png = figure_dir / "Figure_S_site_profile_heterogeneity.png"
    figure.savefig(figure_pdf, bbox_inches="tight")
    figure.savefig(figure_png, dpi=600, bbox_inches="tight")
    plt.close(figure)

    site_other_estimates = site_other_table.set_index("site").loc[SITES, "estimate"]
    site_other_simultaneous_lows = site_other_table.set_index("site").loc[
        SITES, "simultaneous_95_low"
    ]
    if site_other_estimates.le(0).any():
        equal_state_label = "QUALITATIVE_SITE_DISAGREEMENT"
    elif site_other_estimates.ge(0.70).all():
        equal_state_label = "BROADLY_CONSERVED_POINT_ORDERING"
    else:
        equal_state_label = "AMBIGUOUS_SITE_SHAPE_AGREEMENT"

    magnitude_intervals_include_one = bool(
        (
            equal_state_magnitude["ratio_simultaneous_95_low"].le(1)
            & equal_state_magnitude["ratio_simultaneous_95_high"].ge(1)
        ).all()
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "equal_state_interpretation": equal_state_label,
        "site_to_other_three_r": {
            site: float(site_other_estimates.loc[site]) for site in SITES
        },
        "minimum_site_to_other_three_r": float(site_other_estimates.min()),
        "minimum_site_to_other_three_simultaneous_low": float(
            site_other_simultaneous_lows.min()
        ),
        "minimum_direct_site_pair_r": float(min(pair_point)),
        "minimum_state_resolved_site_to_other_three_r": float(
            state_similarity["r_with_other_three_state_profile"].min()
        ),
        "all_equal_state_FP_point_estimates_positive": bool(
            equal_state_fp["positive_point_estimate"].all()
        ),
        "all_equal_state_FP_simultaneous_intervals_exclude_zero": bool(
            equal_state_fp["simultaneous_interval_excludes_zero"].all()
        ),
        "all_state_site_FP_point_estimates_positive": bool(
            state_fp["positive_point_estimate"].all()
        ),
        "all_equal_state_magnitude_intervals_include_one": magnitude_intervals_include_one,
        "descriptive_profile_p_min": float(
            equal_state_magnitude["descriptive_profile_p"].min()
        ),
        "descriptive_profile_p_max": float(
            equal_state_magnitude["descriptive_profile_p"].max()
        ),
        "scalar_profile_site_state_FP_max_absolute_difference": scalar_profile_max_difference,
        "participants_with_multiple_event_study_labels": int(
            person_core["study_label_n"].gt(1).sum()
        ),
        "participant_ID_origin_counts": {
            str(key): int(value)
            for key, value in person_core["id_prefix_origin"].value_counts().items()
        },
        "ordinary_network_correlation_p_values_reported": False,
        "site_to_pooled_used_for_inference": False,
        "pseudo_site_benchmark_run": False,
    }
    summary_path = output_dir / "09_site_profile_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tables = {
        "00_reproduction_gates.csv": gate_table,
        "01_site_composition.csv": site_composition,
        "01b_site_state_coverage.csv": site_state_coverage,
        "01c_site_scanner_composition.csv": site_scanner_composition,
        "01d_study_provenance.csv": study_provenance,
        "01e_cognition_study_provenance.csv": cognition_study_provenance,
        "02_site_state_profile_slopes.csv": site_state_slopes,
        "03_site_equal_state_profile_slopes.csv": site_equal_slopes,
        "03b_pooled_equal_state_profile_slopes.csv": pooled_equal_slopes,
        "04_site_equal_state_similarity.csv": equal_state_similarity,
        "04b_partwhole_similarity_audit.csv": partwhole_similarity_audit,
        "05_site_equal_state_magnitude.csv": equal_state_magnitude,
        "06_site_FP_relative_slopes.csv": equal_state_fp,
        "07_site_state_similarity.csv": state_similarity,
        "07b_site_state_pairwise_similarity_points.csv": state_pairwise,
        "08_site_state_magnitude.csv": state_magnitude,
        "08b_site_state_FP_relative_slopes.csv": state_fp,
        "09_scalar_profile_reconciliation.csv": scalar_profile_reconciliation,
    }
    written_paths = [contract_path, summary_path, figure_pdf, figure_png]
    for filename, table in tables.items():
        path = table_dir / filename
        table.to_csv(path, index=False)
        written_paths.append(path)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "source_script": str(source_script),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "bootstrap_B": bootstrap_b,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pedigrees": len(pedigrees),
        "coefficient_count": coefficient_n,
        "profile_sum_zero_max_error": sum_zero_error,
        "pooled_site_average_max_error": pooled_site_average_error,
        "influence_covariance_max_error": covariance_reconstruction_error,
        "draw_sum_zero_max_error": draw_sum_zero_error,
        "pearson_cosine_max_error": cosine_error,
        "scalar_profile_site_state_FP_max_absolute_difference": scalar_profile_max_difference,
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

    display(gate_table)
    display(study_provenance)
    display(cognition_study_provenance)
    display(equal_state_similarity)
    display(equal_state_magnitude)
    display(equal_state_fp)
    display(scalar_profile_reconciliation)
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
