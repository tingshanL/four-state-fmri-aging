#!/usr/bin/env python3
"""Create cohort and QC summaries for the four-state fMRI analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
PRESPECIFIED_METRICS = [
    "within_default",
    "within_frontoparietal",
    "within_cingulo_opercular",
    "default_frontoparietal",
    "default_cingulo_opercular",
    "frontoparietal_cingulo_opercular",
    "hippocampus_default",
    "hippocampus_visual",
    "visual_somatomotor",
    "overall_network_segregation",
]

# These counts are used only as a convenient check that the same frozen input
# release was found. They are paper-result checks, not general software tests.
EXPECTED = {
    "participants": 1390,
    "sessions": 2789,
    "participants_2plus_any_state_visits": 875,
    "participants_3plus_any_state_visits": 439,
    "all_four_state_complete_sessions": 2687,
    "participants_2plus_complete_four_state_sessions": 834,
    "participants_3plus_complete_four_state_sessions": 408,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Local directory containing the four raw prespecified-metric caches.",
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
        help="Local results root; outputs go into 01_qc_and_cohort/tables.",
    )
    parser.add_argument(
        "--skip-frozen-count-check",
        action="store_true",
        help="Write summaries without requiring the frozen paper counts.",
    )
    return parser.parse_args()


def normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def pick_column(
    frame: pd.DataFrame,
    aliases: list[str],
    *,
    required: bool = False,
    label: str | None = None,
) -> str | None:
    lookup = {normalized_name(column): column for column in frame.columns}
    for alias in aliases:
        match = lookup.get(normalized_name(alias))
        if match is not None:
            return match
    if required:
        raise KeyError(f"Missing {label or aliases[0]}; accepted aliases: {aliases}")
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def find_one(data_dir: Path, filenames: list[str], *, required: bool) -> Path | None:
    """Resolve one expected file without choosing between competing versions."""
    for filename in filenames:
        matches = sorted(path.resolve() for path in data_dir.rglob(filename) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            listed = "\n".join(f"  {path}" for path in matches[:10])
            raise RuntimeError(
                f"Found multiple copies of {filename}. Use a narrower --data-dir:\n{listed}"
            )
    if required:
        raise FileNotFoundError(
            f"Could not find any of {filenames} under {data_dir}"
        )
    return None


def load_metric_caches(data_dir: Path) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for state in STATES:
        stem = f"full_covariance__{state.lower()}__prespecified_metrics"
        path = find_one(data_dir, [f"{stem}.parquet", f"{stem}.csv"], required=True)
        assert path is not None
        if "primary_age_sex_adjusted" in path.name.lower():
            raise AssertionError("Age/sex-adjusted fMRI input is prohibited.")

        table = read_table(path)
        id_column = pick_column(table, ["id_event"], required=True, label="id_event")
        assert id_column is not None
        table = table.rename(columns={id_column: "id_event"})
        missing = sorted(set(PRESPECIFIED_METRICS) - set(table.columns))
        if missing:
            raise KeyError(f"{path} lacks frozen metrics: {missing}")

        table = table[["id_event", *PRESPECIFIED_METRICS]].copy()
        table["id_event"] = table["id_event"].astype(str).str.strip()
        if table["id_event"].duplicated().any():
            raise AssertionError(f"Duplicate id_event rows in {path}")
        table = table.rename(
            columns={
                metric: f"raw__{state.lower()}__{metric}"
                for metric in PRESPECIFIED_METRICS
            }
        )
        combined = (
            table
            if combined is None
            else combined.merge(table, on="id_event", how="outer", validate="one_to_one")
        )
        print(f"{state}: {path}")

    assert combined is not None
    for column in combined.columns:
        if column != "id_event":
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def generic_issue_flag(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    return (
        ~text.isna()
        & ~text.isin(["", "0", "NONE", "NO", "N/A", "NA", "<NA>"])
    ).astype(float)


def load_subjects(subject_table: Path) -> tuple[pd.DataFrame, bool]:
    raw = read_table(subject_table)
    id_event_column = pick_column(
        raw, ["id_event", "subject_event", "participant_event"]
    )
    raw_id_column = pick_column(
        raw, ["id", "subject_id", "participant_id", "src_subject_id"]
    )
    raw_event_column = pick_column(
        raw, ["eventname", "event_name", "event", "visit", "session"]
    )
    if id_event_column is None:
        if raw_id_column is None or raw_event_column is None:
            raise KeyError(
                "Subject table needs id_event, or both participant ID and event columns."
            )
        raw = raw.copy()
        raw["__constructed_id_event"] = (
            raw[raw_id_column].astype(str).str.strip()
            + "_"
            + raw[raw_event_column].astype(str).str.strip()
        )
        id_event_column = "__constructed_id_event"

    age_column = pick_column(
        raw,
        ["age_open", "age", "event_age", "interview_age"],
        required=True,
        label="event age",
    )
    sex_column = pick_column(raw, ["sex", "gender"], required=True, label="sex")
    site_column = pick_column(
        raw,
        ["site", "site_name", "site_id", "collection_site", "institution"],
        required=True,
        label="site",
    )
    scanner_column = pick_column(
        raw,
        [
            "scanner",
            "scanner_id",
            "scanner_name",
            "acquisition_scanner",
            "mri_scanner",
            "scanner_model",
            "scanner_serial_number",
        ],
    )
    days_column = pick_column(raw, ["days_from_V1", "days_from_v1", "daysfromv1"])
    qc_column = pick_column(raw, ["MR_QC_Issue_Codes", "mr_qc_issue_codes"])

    keep = {
        id_event_column: "id_event",
        age_column: "event_age",
        sex_column: "sex",
        site_column: "site",
    }
    if scanner_column is not None:
        keep[scanner_column] = "scanner"
    if days_column is not None:
        keep[days_column] = "days_from_v1"
    if raw_id_column is not None:
        keep[raw_id_column] = "id"
    if qc_column is not None:
        keep[qc_column] = "mr_qc_issue_codes"

    subjects = raw[list(keep)].rename(columns=keep).copy()
    subjects["id_event"] = subjects["id_event"].astype(str).str.strip()
    subjects["event_age"] = pd.to_numeric(subjects["event_age"], errors="coerce")
    if "id" not in subjects:
        subjects["id"] = subjects["id_event"].str.replace(r"_[^_]+$", "", regex=True)
    subjects["id"] = subjects["id"].astype(str).str.strip()
    if "scanner" not in subjects:
        subjects["scanner"] = "MISSING"
    if "days_from_v1" not in subjects:
        subjects["days_from_v1"] = np.nan
    subjects["days_from_v1"] = pd.to_numeric(
        subjects["days_from_v1"], errors="coerce"
    )
    for column in ["sex", "site", "scanner"]:
        subjects[column] = (
            subjects[column].astype("string").str.strip().fillna("MISSING")
        )

    if "mr_qc_issue_codes" in subjects:
        subjects["generic_mr_qc_flag"] = generic_issue_flag(
            subjects["mr_qc_issue_codes"]
        )
    else:
        subjects["generic_mr_qc_flag"] = np.nan

    if subjects["id_event"].duplicated().any():
        duplicate_count = int(subjects["id_event"].duplicated(keep=False).sum())
        raise AssertionError(
            f"Subject table has {duplicate_count} rows with duplicated id_event."
        )
    return subjects, qc_column is not None


def normalize_state(value: object) -> str | float:
    text = normalized_name(value)
    for state in STATES:
        if state.lower() in text:
            return state
    return np.nan


def load_optional_qc(data_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    qc_path = find_one(
        data_dir,
        [
            "AABC2_fMRI_QC_summary.csv",
            "AABC_Release2_fMRI_QC_summary.csv",
            "fMRI_QC_summary.csv",
            "AABC2_fMRI_QC_summary.parquet",
            "fMRI_QC_summary.parquet",
        ],
        required=False,
    )
    empty = pd.DataFrame(columns=["id_event", "state", "motion", "tsnr", "usable"])
    notes: list[dict[str, object]] = []
    if qc_path is None:
        return empty, notes

    raw = read_table(qc_path)
    id_column = pick_column(raw, ["id_event"], required=True, label="QC id_event")
    state_column = pick_column(raw, ["state", "task", "scan", "run", "paradigm"])
    if state_column is None:
        notes.append(
            {
                "resource": str(qc_path),
                "status": "FOUND_BUT_NO_STATE_COLUMN",
                "coverage": np.nan,
                "used_as_motion": False,
            }
        )
        return empty, notes

    motion_column = pick_column(
        raw,
        [
            "motion",
            "mean_fd",
            "meanfd",
            "framewise_displacement",
            "mean_framewise_displacement",
        ],
    )
    tsnr_column = pick_column(raw, ["tsnr", "temporal_snr", "temporalsnr"])
    usable_column = pick_column(
        raw, ["usable", "usability", "fmri_pass", "qc_status", "include"]
    )
    assert id_column is not None
    qc = pd.DataFrame(
        {
            "id_event": raw[id_column].astype(str).str.strip(),
            "state": raw[state_column].map(normalize_state),
            "motion": (
                pd.to_numeric(raw[motion_column], errors="coerce")
                if motion_column
                else np.nan
            ),
            "tsnr": (
                pd.to_numeric(raw[tsnr_column], errors="coerce")
                if tsnr_column
                else np.nan
            ),
            "usable_raw": raw[usable_column] if usable_column else np.nan,
        }
    ).dropna(subset=["state"])

    if usable_column:
        text = qc["usable_raw"].astype("string").str.strip().str.upper()
        numeric = pd.to_numeric(qc["usable_raw"], errors="coerce")
        qc["usable"] = np.where(
            numeric.notna(),
            numeric.gt(0).astype(float),
            text.isin(
                ["Y", "YES", "TRUE", "PASS", "PASSED", "USABLE", "INCLUDE"]
            ).astype(float),
        )
    else:
        qc["usable"] = np.nan

    qc = qc.groupby(["id_event", "state"], as_index=False).agg(
        motion=("motion", "mean"),
        tsnr=("tsnr", "mean"),
        usable=("usable", "min"),
    )
    return qc, notes


def build_qc_audit(
    qc_long: pd.DataFrame,
    extra_rows: list[dict[str, object]],
    subjects: pd.DataFrame,
    generic_qc_available: bool,
) -> pd.DataFrame:
    rows = list(extra_rows)
    for kind in ["motion", "tsnr", "usable"]:
        available = kind in qc_long and qc_long[kind].notna().any()
        rows.append(
            {
                "resource": kind,
                "status": "AVAILABLE" if available else "UNAVAILABLE",
                "coverage": (
                    float(qc_long[kind].notna().mean())
                    if len(qc_long) and kind in qc_long
                    else 0.0
                ),
                "used_as_motion": bool(kind == "motion" and available),
            }
        )
    rows.append(
        {
            "resource": "MR_QC_Issue_Codes",
            "status": "AVAILABLE" if generic_qc_available else "UNAVAILABLE",
            "coverage": float(subjects["generic_mr_qc_flag"].notna().mean()),
            "used_as_motion": False,
        }
    )
    return pd.DataFrame(rows)


def build_events(
    metric_wide: pd.DataFrame,
    subjects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = metric_wide.merge(subjects, on="id_event", how="inner", validate="one_to_one")
    if events.empty:
        raise AssertionError("No overlap between metric caches and subject-event table.")

    events["any_metric"] = False
    state_rows = []
    for state in STATES:
        columns = [
            f"raw__{state.lower()}__{metric}" for metric in PRESPECIFIED_METRICS
        ]
        events[f"{state.lower()}__any"] = events[columns].notna().any(axis=1)
        events["any_metric"] |= events[f"{state.lower()}__any"]

    events = events.loc[events["any_metric"]].copy()
    events["sort_time"] = events["days_from_v1"].where(
        events["days_from_v1"].notna(), events["event_age"] * 365.25
    )
    events = events.sort_values(["id", "sort_time", "id_event"]).copy()
    events["baseline_sort_time"] = events.groupby("id")["sort_time"].transform("min")
    events["baseline_age"] = events.groupby("id")["event_age"].transform("first")
    events["time_years"] = (
        events["sort_time"] - events["baseline_sort_time"]
    ) / 365.25

    for state in STATES:
        columns = [
            f"raw__{state.lower()}__{metric}" for metric in PRESPECIFIED_METRICS
        ]
        part = events[["id_event", "id", *columns]].copy()
        part["state"] = state
        part["available_metric_n"] = part[columns].notna().sum(axis=1)
        state_rows.append(part[["id_event", "id", "state", "available_metric_n"]])

    session_state = pd.concat(state_rows, ignore_index=True)
    session_state["any_metric"] = session_state["available_metric_n"].gt(0)
    session_state["all_metrics"] = session_state["available_metric_n"].eq(
        len(PRESPECIFIED_METRICS)
    )
    complete4_event_ids = (
        session_state.loc[session_state["all_metrics"]]
        .groupby("id_event")["state"]
        .nunique()
        .loc[lambda counts: counts.eq(4)]
        .index
    )
    events["complete_four_state_all_metrics"] = events["id_event"].isin(
        complete4_event_ids
    )
    return events, session_state


def make_summaries(
    events: pd.DataFrame,
    session_state: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    state_rows = []
    for state in STATES:
        subset = session_state.loc[
            session_state["state"].eq(state) & session_state["any_metric"]
        ]
        visit_counts = subset.groupby("id")["id_event"].nunique()
        state_rows.append(
            {
                "state": state,
                "participants": subset["id"].nunique(),
                "sessions": subset["id_event"].nunique(),
                "participants_2plus_visits": int(visit_counts.ge(2).sum()),
                "participants_3plus_visits": int(visit_counts.ge(3).sum()),
                "complete_all_metrics_sessions": int(subset["all_metrics"].sum()),
            }
        )
    cohort_by_state = pd.DataFrame(state_rows)

    participant_event_counts = events.groupby("id")["id_event"].nunique()
    complete4_counts = (
        events.loc[events["complete_four_state_all_metrics"]]
        .groupby("id")["id_event"]
        .nunique()
    )
    overall = pd.DataFrame(
        [
            {
                "participants": events["id"].nunique(),
                "sessions": events["id_event"].nunique(),
                "participants_2plus_any_state_visits": int(
                    participant_event_counts.ge(2).sum()
                ),
                "participants_3plus_any_state_visits": int(
                    participant_event_counts.ge(3).sum()
                ),
                "all_four_state_complete_sessions": int(
                    events["complete_four_state_all_metrics"].sum()
                ),
                "participants_2plus_complete_four_state_sessions": int(
                    complete4_counts.ge(2).sum()
                ),
                "participants_3plus_complete_four_state_sessions": int(
                    complete4_counts.ge(3).sum()
                ),
            }
        ]
    )

    person_summary = (
        events.sort_values(["id", "time_years"])
        .groupby("id", as_index=False)
        .agg(
            sessions=("id_event", "nunique"),
            followup_years=("time_years", "max"),
            site_n=("site", "nunique"),
            scanner_n=("scanner", "nunique"),
            baseline_age=("baseline_age", "first"),
            sex=("sex", "first"),
        )
    )
    person_summary["site_changed"] = person_summary["site_n"].gt(1)
    person_summary["scanner_changed"] = person_summary["scanner_n"].gt(1)

    longitudinal = person_summary["sessions"].ge(2)
    followup = (
        person_summary.loc[longitudinal, "followup_years"]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .rename("value")
        .reset_index()
        .rename(columns={"index": "statistic"})
    )
    site_scanner = pd.DataFrame(
        [
            {
                "longitudinal_participants": int(longitudinal.sum()),
                "same_site_n": int((longitudinal & ~person_summary["site_changed"]).sum()),
                "site_changed_n": int((longitudinal & person_summary["site_changed"]).sum()),
                "same_scanner_n": int(
                    (longitudinal & ~person_summary["scanner_changed"]).sum()
                ),
                "scanner_changed_n": int(
                    (longitudinal & person_summary["scanner_changed"]).sum()
                ),
            }
        ]
    )

    return {
        "cohort_by_state.csv": cohort_by_state,
        "overall_longitudinal_coverage.csv": overall,
        "followup_summary.csv": followup,
        "site_scanner_changes.csv": site_scanner,
        "person_summary.csv": person_summary,
    }


def check_frozen_counts(overall: pd.DataFrame) -> None:
    row = overall.iloc[0]
    mismatches = {
        key: {"expected": expected, "observed": int(row[key])}
        for key, expected in EXPECTED.items()
        if int(row[key]) != expected
    }
    if mismatches:
        raise AssertionError(
            "Frozen cohort counts differ. Confirm the selected data release or rerun "
            f"with --skip-frozen-count-check after review:\n{json.dumps(mismatches, indent=2)}"
        )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    subject_table = args.subject_table.expanduser().resolve()
    results_root = args.results_dir.expanduser().resolve()
    table_dir = results_root / "01_qc_and_cohort" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    if not subject_table.is_file():
        raise FileNotFoundError(subject_table)

    metrics = load_metric_caches(data_dir)
    subjects, generic_qc_available = load_subjects(subject_table)
    qc_long, qc_notes = load_optional_qc(data_dir)
    qc_audit = build_qc_audit(
        qc_long, qc_notes, subjects, generic_qc_available
    )
    events, session_state = build_events(metrics, subjects)
    outputs = make_summaries(events, session_state)
    outputs["qc_resource_audit.csv"] = qc_audit

    if not args.skip_frozen_count_check:
        check_frozen_counts(outputs["overall_longitudinal_coverage.csv"])

    for filename, frame in outputs.items():
        frame.to_csv(table_dir / filename, index=False)

    overall = outputs["overall_longitudinal_coverage.csv"].iloc[0]
    followup = outputs["followup_summary.csv"].set_index("statistic")["value"]
    scanner = outputs["site_scanner_changes.csv"].iloc[0]
    print("\nQC/COHORT EXTRACTION COMPLETE")
    print(f"Participants with any fMRI metric: {int(overall['participants']):,}")
    print(f"Sessions with any fMRI metric: {int(overall['sessions']):,}")
    print(
        "Participants with >=2 visits: "
        f"{int(overall['participants_2plus_any_state_visits']):,}"
    )
    print(f"Median follow-up: {float(followup.loc['50%']):.3f} years")
    print(
        "Same-scanner / scanner-changed longitudinal participants: "
        f"{int(scanner['same_scanner_n']):,} / {int(scanner['scanner_changed_n']):,}"
    )
    print(f"Wrote {len(outputs)} tables to {table_dir}")


if __name__ == "__main__":
    main()
