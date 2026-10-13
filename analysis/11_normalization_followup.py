#!/usr/bin/env python3
"""Summarize amplitude trajectories, equal-state network slopes, and matrix spectra."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

VERSION = "1.0.0"
STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
NOMINAL_TASK_FRAMES = {"CARIT": 300, "FACENAME": 345, "VISMOTOR": 194}
PROTOCOL_SOURCE = "Harms et al. 2018, doi:10.1016/j.neuroimage.2018.09.060"
NEAR_ZERO_RTOLS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_script10(path):
    require(path.is_file(), "Put script 11 beside 10_noise_amplitude_and_normalization.py, or use --script10.")
    spec = importlib.util.spec_from_file_location("normalization_followup_script10", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(module.VERSION == "1.0.0", "This follow-up expects script-10 version 1.0.0.")
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--analysis10-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--script10", type=Path, default=Path(__file__).with_name("10_noise_amplitude_and_normalization.py"))
    p.add_argument("--stage", choices=["all", "longitudinal", "pooled", "diagnostic"], default="all")
    p.add_argument("--subject-table", type=Path)
    p.add_argument("--frames-table", type=Path)
    p.add_argument("--correlation-scale", choices=["unverified", "raw_r", "fisher_z"], default="unverified")
    p.add_argument("--correlation-source", type=Path)
    p.add_argument("--identity-tol", type=float, default=1e-9)
    p.add_argument("--reproduction-tol", type=float, default=1e-6)
    a = p.parse_args()
    for key in ["data_dir", "results_dir", "script10", "subject_table", "frames_table", "correlation_source"]:
        value = getattr(a, key)
        if value is not None:
            setattr(a, key, value.expanduser().resolve())
    a.analysis10_dir = (a.analysis10_dir or a.results_dir / "10_noise_amplitude_and_normalization").expanduser().resolve()
    a.output_dir = (a.output_dir or a.results_dir / "11_normalization_followup").expanduser().resolve()
    protected = [a.data_dir, a.analysis10_dir, a.results_dir / "02_primary_profile_and_measurement", a.results_dir]
    for path in protected:
        if a.output_dir == path or a.output_dir in path.parents:
            p.error("--output-dir must be a dedicated directory outside input/analysis-02/analysis-10 directories")
    if a.data_dir in a.output_dir.parents or a.analysis10_dir in a.output_dir.parents or protected[2] in a.output_dir.parents:
        p.error("Do not put follow-up outputs inside data, analysis 02, or analysis 10")
    for key in ["identity_tol", "reproduction_tol"]:
        if not np.isfinite(getattr(a, key)) or getattr(a, key) <= 0:
            p.error(key + " must be finite and positive")
    if a.correlation_scale != "unverified" and (a.correlation_source is None or not a.correlation_source.is_file()):
        p.error("A documented correlation scale requires --correlation-source")
    if a.frames_table is not None and not a.frames_table.is_file():
        p.error("--frames-table does not exist")
    return a


def load_joint(path):
    with np.load(path, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in ["coef", "robust_cov", "design_names", "components", "outcome_coordinates"]}
    for name in ["design_names", "components", "outcome_coordinates"]:
        data[name] = data[name].astype(str).tolist()
    size = len(data["design_names"]) * len(data["components"]) * len(data["outcome_coordinates"])
    require(data["coef"].shape == (size,) and data["robust_cov"].shape == (size, size), "Unexpected joint checkpoint dimensions")
    require(np.isfinite(data["coef"]).all() and np.isfinite(data["robust_cov"]).all(), "Nonfinite saved joint fit")
    require(np.allclose(data["robust_cov"], data["robust_cov"].T, atol=1e-10, rtol=1e-10), "Asymmetric saved joint covariance")
    return data


def slope_matrix(names):
    require("t5" in names, "Saved design has no elapsed-follow-up coefficient")
    rows = []
    for state in STATES:
        v = np.zeros(len(names))
        v[names.index("t5")] = 1.0
        for coded in STATES[:-1]:
            name = "t5:state_sum[" + coded + "]"
            require(name in names, "Missing direct-state term: " + name)
            v[names.index(name)] = 1.0 if state == coded else -1.0 if state == STATES[-1] else 0.0
        rows.append(v)
    return np.vstack(rows)


def contrast(coef, cov, v):
    estimate = float(v @ coef)
    variance = float(v @ cov @ v)
    magnitude = max(1.0, float(np.linalg.norm(cov, ord=np.inf)) * float(v @ v))
    require(variance >= -1e-10 * magnitude, "Materially negative contrast variance")
    se = math.sqrt(max(0.0, variance))
    if se > 0:
        p = float(2 * norm.sf(abs(estimate / se)))
    else:
        p = 1.0 if estimate == 0 else 0.0
    return estimate, se, p


def interval_row(coef, cov, v, comparisons):
    b, se, p = contrast(coef, cov, v)
    point = float(norm.ppf(.975))
    simultaneous = float(norm.ppf(1 - .05 / (2 * comparisons)))
    return {"slope_per_5y": b, "se": se, "ci95_low": b - point * se, "ci95_high": b + point * se,
            "simultaneous_ci_low": b - simultaneous * se, "simultaneous_ci_high": b + simultaneous * se,
            "p_unadjusted": p}


def finish_figure(fig, out, name):
    for ext in ["pdf", "png"]:
        fig.savefig(out / "figures" / (name + "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def pooled_stage(a, m, out):
    print("Extract equal-state network contrasts from the saved joint covariance", flush=True)
    joint = load_joint(a.analysis10_dir / "private" / "joint_profile.npz")
    names, networks, components = joint["design_names"], joint["outcome_coordinates"], joint["components"]
    require(components == m.COMPONENTS and len(networks) == 10, "Unexpected profile checkpoint order")
    require("Frontoparietal" in networks, "FP is absent from saved network coordinates")
    p, k = len(names), len(networks)
    average = slope_matrix(names).mean(axis=0)
    rows = []
    for component in components:
        offset = components.index(component) * p * k
        for index, network in enumerate(networks):
            v = np.zeros(len(joint["coef"]))
            v[offset + index * p:offset + (index + 1) * p] = average
            rows.append({"component": component, "network": network, "target": "NETWORK_LOG_LEVEL",
                "state": "EQUAL_STATE_MEAN", **interval_row(joint["coef"], joint["robust_cov"], v, 30),
                "interval_family": "Bonferroni across 30 pooled slopes (three representations x ten networks)",
                "estimand": "Equal-state/equal-site slope under saved PROFILE GLS weights",
                "interpretation": "An interval containing zero does not establish maintenance or equivalence"})
    table = m.write_table(out, "30_pooled_network_slopes.csv", rows)
    old = pd.read_csv(a.analysis10_dir / "tables" / "20_longitudinal_slopes.csv")
    audits = []
    for row in rows:
        selected = old.loc[old.target.eq("NETWORK_LOG_LEVEL") & old.component.eq(row["component"]) & old.network.eq(row["network"])]
        require(len(selected) == 4 and set(selected.state) == set(STATES), "Saved four-state network slopes incomplete")
        error = abs(row["slope_per_5y"] - float(selected.slope_per_5y.mean()))
        audits.append({"component": row["component"], "network": row["network"],
                       "pooled_estimate_vs_four_state_mean_error": error, "pass": error < a.reproduction_tol})
    m.write_table(out, "31_pooled_contrast_reproduction.csv", audits)
    require(all(row["pass"] for row in audits), "Pooled contrasts disagree with script-10 state slopes")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=True)
    for ax, component in zip(axes, ["SIGNAL", "TOTAL"]):
        selected = table.loc[table.component.eq(component)].set_index("network").loc[networks]
        y = np.arange(k)
        b = selected.slope_per_5y.to_numpy()
        low, high = selected.simultaneous_ci_low.to_numpy(), selected.simultaneous_ci_high.to_numpy()
        ax.errorbar(b, y, xerr=np.vstack([b-low, high-b]), fmt="o", ms=4, color=m.COLORS[component], lw=1)
        ax.axvline(0, color=".65", lw=.8)
        ax.set_title("Signal-normalized" if component == "SIGNAL" else "Candidate total-normalized", fontsize=10)
        ax.set_xlabel("Equal-state log slope / 5 years")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(np.arange(k)); axes[0].set_yticklabels(networks, fontsize=9); axes[0].invert_yaxis()
    fig.tight_layout()
    finish_figure(fig, out, "03_pooled_network_slopes")
    return {"pooled_contrasts": len(rows), "intervals": "30-slope simultaneous family; no equivalence claim",
            "uses_existing_joint_covariance": True, "new_model_fit": False}


def amplitude_inputs(a, m, out):
    source, manifest = m.resolve_node_manifest(a.data_dir, a.results_dir)
    parcel_map, selection = m.select_cortical_parcels(manifest)
    m.write_table(out, "00_parcel_selection.csv", selection)
    require(len(parcel_map) == m.EXPECTED_PARCELS, "Expected 360 selected cortical parcels")
    networks = m.load_eligible_networks(a.results_dir)
    old_contract = read_json(a.analysis10_dir / "analysis_contract.json")
    require(networks == old_contract["network_order"], "Script-10 network order mismatch")
    archives = m.resolve_unique_files(a.data_dir, "*fMRIAmplitudes.zip", a.results_dir)
    amplitudes, inventory = {}, []
    for state in STATES:
        path = archives[state]
        require(path is not None, "Missing amplitude archive for " + state)
        print("Load signal/noise amplitude archive: " + state, flush=True)
        signal, sm = m.load_amplitude_product(path, parcel_map["Brain Area"].tolist(), "FullAmplitudes")
        noise, nm = m.load_amplitude_product(path, parcel_map["Brain Area"].tolist(), "NoiseAmplitudes")
        amplitudes[state] = {"signal": signal, "noise": noise}
        inventory.append({"state": state, "signal_member": sm, "noise_member": nm, **m.archive_signature(path)})
    m.write_table(out, "00_amplitude_inventory.csv", inventory)
    return parcel_map, networks, amplitudes


def longitudinal_stage(a, m, out, parcel_map, networks, amplitudes):
    print("Reconstruct the exact script-10 matched cohort; reuse COMMON GLS weights", flush=True)
    result = m.prepare_outcomes(a, parcel_map, networks, amplitudes, out)
    _, _, frame, outcomes, _, _, _ = result
    old_summary = read_json(a.analysis10_dir / "longitudinal_summary.json")
    require(m.key_hash(frame) == old_summary["matched_row_sha256"], "Matched row fingerprint differs from script 10")
    metadata = read_json(a.analysis10_dir / "private" / "model_common.json")
    require(metadata["optimizer_success"] and len(frame) == metadata["rows"], "Invalid saved common model")
    design = m.DirectStateDesign().fit(frame)
    require(design.names == metadata["design_names"], "Reconstructed design differs from saved common model")
    x = design.transform(frame)
    logs = {"SIGNAL_PAIR_AMPLITUDE": np.log(frame[["P_signal__"+n for n in networks]].to_numpy(float)),
            "TOTAL_PAIR_AMPLITUDE": np.log(frame[["P_total__"+n for n in networks]].to_numpy(float))}
    pn = np.full((len(frame), len(networks)), np.nan)
    for state in STATES:
        index = np.flatnonzero(frame.state.eq(state).to_numpy())
        events = frame.iloc[index].id_event.astype(str).tolist()
        for j, network in enumerate(networks):
            areas = parcel_map.loc[parcel_map.node.map(m.compact_token).eq(m.compact_token(network)), "Brain Area"].tolist()
            values = amplitudes[state]["noise"].reindex(events)[areas].to_numpy(float)
            pn[index, j] = m.pair_products(values)[0]
    m.write_table(out, "16_noise_product_support.csv", [{"state": state, "network": network,
        "matched_rows": int(frame.state.eq(state).sum()),
        "nonpositive_or_nonfinite_noise_products": int((~np.isfinite(pn[frame.state.eq(state), j]) | (pn[frame.state.eq(state), j] <= 0)).sum())}
        for j, network in enumerate(networks) for state in STATES])
    require(np.isfinite(pn).all() and (pn > 0).all(), "Noise pair products cannot be logged on the frozen cohort; no rows were dropped. Inspect table 16.")
    logs["NOISE_PAIR_AMPLITUDE"] = np.log(pn)
    logs["ATTENUATION_H"] = logs["SIGNAL_PAIR_AMPLITUDE"] - logs["TOTAL_PAIR_AMPLITUDE"]
    metrics = list(logs)
    # Including the three old outcomes provides a direct coefficient AND
    # joint covariance reproduction check under the saved weights.
    y = np.column_stack([logs[name].mean(axis=1) for name in metrics] +
                        [outcomes[name].mean(axis=1) for name in m.COMPONENTS])
    fit = m.fixed_joint_gls(x, y, frame.id, frame.state, frame.pedigree_group,
                          metadata["tau_participant"], metadata["tau_participant_state"])
    coef, cov, p = fit["coef"], fit["robust_cov"], x.shape[1]
    old = load_joint(a.analysis10_dir / "private" / "joint_common.npz")
    require(old["components"] == m.COMPONENTS and old["design_names"] == design.names, "Unexpected common checkpoint ordering")
    indices = np.arange(4*p, 7*p)
    checks = [{"identity": "SCRIPT10_COEFFICIENT_REPRODUCTION", "max_absolute_error": float(np.max(np.abs(coef[indices]-old["coef"])))},
              {"identity": "SCRIPT10_JOINT_COVARIANCE_REPRODUCTION", "max_absolute_error": float(np.max(np.abs(cov[np.ix_(indices, indices)]-old["robust_cov"])))},
              {"identity": "H_EQUALS_LOG_SIGNAL_MINUS_LOG_TOTAL_COEFFICIENTS", "max_absolute_error": float(np.max(np.abs(coef[3*p:4*p]-coef[:p]+coef[p:2*p])))},
              {"identity": "H_EQUALS_LOG_SIGNAL_MINUS_LOG_TOTAL_INFLUENCES", "max_absolute_error": float(np.max(np.abs(fit["influence"][:, 3*p:4*p]-fit["influence"][:, :p]+fit["influence"][:, p:2*p])))},
              {"identity": "H_REPRODUCES_EXISTING_H", "max_absolute_error": float(np.max(np.abs(coef[3*p:4*p]-old["coef"][p:2*p])))}]
    for row in checks:
        tol = a.reproduction_tol if row["identity"].startswith("SCRIPT10") else a.identity_tol
        row["tolerance"] = tol; row["pass"] = row["max_absolute_error"] < tol
    m.write_table(out, "23_amplitude_fit_reproduction_and_identities.csv", checks)
    require(all(row["pass"] for row in checks), "Saved-weight reproduction or exact amplitude identity failed")
    cs = design.state_slope_matrix()
    slope_rows, tests, difference_rows = [], [], []
    for j, metric in enumerate(metrics):
        for label, c in zip(STATES + ["EQUAL_STATE_MEAN"], list(cs) + [cs.mean(axis=0)]):
            v = np.zeros(len(coef)); v[j*p:(j+1)*p] = c
            row = {"metric": metric, "state": label, **interval_row(coef, cov, v, 20),
                   "interval_family": "Bonferroni across 20 reported slopes (four outcomes x five state summaries)",
                   "working_covariance": "Saved script-10 MATCHED_SIGNAL_COMMON"}
            for suffix, column in [("estimate", "slope_per_5y"), ("ci95_low", "ci95_low"), ("ci95_high", "ci95_high"),
                                   ("simultaneous_ci_low", "simultaneous_ci_low"), ("simultaneous_ci_high", "simultaneous_ci_high")]:
                row["percent_change_"+suffix] = float(100*np.expm1(row[column]))
            slope_rows.append(row)
        if metric != "ATTENUATION_H":
            v = np.zeros((4, len(coef))); v[:, j*p:(j+1)*p] = cs
            stat, df, pvalue = m.wald_matrix(coef, cov, v)
            tests.append({"metric": metric, "test": "ALL_FOUR_STATE_SLOPES_ZERO", "chi2": stat, "df": df, "p_unadjusted": pvalue})
    adjusted = m.holm_adjust([row["p_unadjusted"] for row in tests])
    for row, pv in zip(tests, adjusted):
        row["p_holm_three_amplitude_tests"] = pv
    # Paired slope contrast answers whether signal and noise products change
    # differently; separate significant/nonsignificant tests cannot answer it.
    for label, c in zip(STATES + ["EQUAL_STATE_MEAN"], list(cs) + [cs.mean(axis=0)]):
        v = np.zeros(len(coef)); v[:p] = c; v[2*p:3*p] = -c
        difference_rows.append({"state": label, "contrast": "LOG_SIGNAL_PAIR_PRODUCT_MINUS_LOG_NOISE_PAIR_PRODUCT",
                               **interval_row(coef, cov, v, 5), "interval_family": "Five paired state/pooled contrasts; Bonferroni"})
    slopes = m.write_table(out, "20_common_amplitude_slopes.csv", slope_rows)
    m.write_table(out, "21_common_amplitude_omnibus_tests.csv", tests)
    m.write_table(out, "22_signal_minus_noise_paired_slopes.csv", difference_rows)
    saved_components = metrics + ["REPRO_"+name for name in m.COMPONENTS]
    np.savez_compressed(out / "private" / "joint_common_amplitudes.npz", coef=coef, robust_cov=cov,
        design_names=np.asarray(design.names), components=np.asarray(saved_components), outcome_coordinates=np.asarray(["COMMON"]))
    m.save_json(out / "private" / "model_common_amplitudes.json", {**metadata, "component_order": saved_components,
        "new_REML_fit": False, "matched_row_sha256": m.key_hash(frame)})
    attenuation_rows = []
    for state in STATES:
        subset = frame.loc[frame.state.eq(state)]
        for network in networks:
            factor = subset["P_signal__"+network].to_numpy(float) / subset["P_total__"+network].to_numpy(float)
            require(np.isfinite(factor).all() and (factor > 0).all() and (factor <= 1+a.identity_tol).all(), "Invalid cohort attenuation factor")
            qs = np.quantile(factor, [.05, .25, .5, .75, .95])
            attenuation_rows.append({"state": state, "network": network, "observation_n": len(subset), "participant_n": subset.id.nunique(),
                "P_signal_over_P_total_min": float(factor.min()), "P_signal_over_P_total_p05": qs[0], "P_signal_over_P_total_p25": qs[1],
                "P_signal_over_P_total_median": qs[2], "P_signal_over_P_total_p75": qs[3], "P_signal_over_P_total_p95": qs[4],
                "P_signal_over_P_total_max": float(factor.max()), "inverse_factor_median": float(np.median(1/factor)),
                "unit": "Network ratio of mean pair SD products; not mean parcel variance fraction",
                "weighting": "Unweighted matched session-state observations; repeated participants contribute multiple rows"})
    m.write_table(out, "24_cohort_network_attenuation_summary.csv", attenuation_rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.4))
    for j, metric in enumerate(metrics):
        selected = slopes.loc[slopes.metric.eq(metric)].set_index("state").loc[STATES]
        b = selected.slope_per_5y.to_numpy(); lo = selected.simultaneous_ci_low.to_numpy(); hi = selected.simultaneous_ci_high.to_numpy()
        pos = np.arange(4) + (j-1.5)*.16
        axes[0].errorbar(pos, b, yerr=np.vstack([b-lo, hi-b]), fmt="o", ms=3, lw=1, label=metric.replace("_PAIR_AMPLITUDE", "").replace("ATTENUATION_H", "H"))
    axes[0].axhline(0, color=".65", lw=.8); axes[0].set_xticks(np.arange(4)); axes[0].set_xticklabels(STATES, fontsize=8)
    axes[0].set_ylabel("Common log slope / 5 years"); axes[0].legend(frameon=False, fontsize=8)
    summary = pd.DataFrame(attenuation_rows)
    for j, state in enumerate(STATES):
        selected = summary.loc[summary.state.eq(state)].set_index("network").loc[networks]
        axes[1].plot(np.arange(len(networks)), selected.P_signal_over_P_total_median, "o-", ms=3, lw=.7, label=state)
    axes[1].set_xticks(np.arange(len(networks))); axes[1].set_xticklabels(networks, rotation=65, ha="right", fontsize=7)
    axes[1].set_ylabel("Median network P_signal / P_total"); axes[1].set_ylim(0, 1); axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); finish_figure(fig, out, "02_common_amplitude_and_cohort_attenuation")
    return {"rows": len(frame), "participants": int(frame.id.nunique()), "matched_row_sha256": m.key_hash(frame),
            "new_REML_fit": False, "saved_common_weights_reused": True, "identities_and_old_joint_fit_reproduce": True,
            "noise_stability_inferred_from_nonsignificance": False, "causal_attribution_from_log_identity": False}


def frames_input(a):
    if a.frames_table is None:
        return None
    table = pd.read_csv(a.frames_table, dtype=str)
    require("state" in table and "source" in table, "Frame table needs state and source")
    keys = [name for name in ["diagnostic_session", "id_event"] if name in table]
    require(len(keys) == 1, "Frame table needs exactly one of diagnostic_session or id_event")
    key = keys[0]
    require(table.state.isin(STATES).all(), "Unknown frame-table state")
    require(table[key].notna().all() and not table.duplicated(["state", key]).any(), "Missing/duplicate frame-table key")
    require(table.source.notna().all() and table.source.str.strip().ne("").all(), "Frame-table source is required")
    for name in ["acquired_frames", "retained_frames"]:
        if name not in table:
            table[name] = np.nan
        else:
            raw = table[name]
            table[name] = pd.to_numeric(raw, errors="coerce")
            require(not (raw.notna() & table[name].isna()).any(), "Nonnumeric " + name)
            value = table[name].dropna().to_numpy(float)
            require(np.isfinite(value).all() and (value > 1).all() and (value == np.floor(value)).all(), "Frame counts must be integers >1")
    require(table[["acquired_frames", "retained_frames"]].notna().any(axis=1).all(), "A frame row has no count")
    both = table.acquired_frames.notna() & table.retained_frames.notna()
    require((table.loc[both, "retained_frames"] <= table.loc[both, "acquired_frames"]).all(), "Retained count exceeds acquired count")
    table.attrs["key"] = key
    return table


def frame_annotation(state, label, event, frames):
    if frames is not None:
        key = frames.attrs["key"]
        selected = frames.loc[frames.state.eq(state) & frames[key].eq(label if key == "diagnostic_session" else event)]
        if len(selected):
            row = selected.iloc[0]
            retained = row.retained_frames
            return {"frames_used": int(retained if pd.notna(retained) else row.acquired_frames),
                    "frame_count_basis": "OBSERVED_RETAINED" if pd.notna(retained) else "OBSERVED_ACQUIRED_UPPER_BOUND",
                    "frame_source": str(row.source), "acquired_frames": row.acquired_frames, "retained_frames": retained}
    return {"frames_used": NOMINAL_TASK_FRAMES.get(state, np.nan), "frame_count_basis": "NOMINAL_PROTOCOL_SCENARIO" if state in NOMINAL_TASK_FRAMES else "UNKNOWN_REST_RUN_AGGREGATION",
            "frame_source": PROTOCOL_SOURCE if state in NOMINAL_TASK_FRAMES else "Actual concatenation not inferred",
            "acquired_frames": np.nan, "retained_frames": np.nan}


def diagnostic_ids(a, m, state, covariance_path, areas, amplitudes, old_contract):
    frozen = m.load_existing_diagnostic_ids(a.results_dir).get(state)
    if frozen:
        return list(frozen)
    common = sorted(set(amplitudes[state]["signal"].index) & set(amplitudes[state]["noise"].index) &
                    set(m.archive_event_ids(covariance_path, areas[0])))
    n = int(old_contract.get("diagnostic_sessions_per_state", 3))
    manifest = read_json(a.analysis10_dir / "run_manifest.json")
    n = int(manifest.get("command_arguments", {}).get("diagnostic_sessions_per_state", n))
    return m.deterministic_diagnostic_ids(common, n)


def diagnostic_stage(a, m, out, parcel_map, networks, amplitudes):
    print("Read script-10 matrix checkpoints; report spectral and frame-count scenarios", flush=True)
    old_contract = read_json(a.analysis10_dir / "analysis_contract.json")
    diagnostic = read_json(a.analysis10_dir / "diagnostic_summary.json")
    require(not diagnostic["technical_failures"], "Script-10 diagnostic technical failures are unresolved")
    args = SimpleNamespace(**{name: old_contract[name] for name in ["psd_rtol", "symmetry_rtol", "symmetry_atol"]},
                           csv_entry_rtol=old_contract["csv_entry_rtol_assumed"], csv_entry_atol=old_contract["csv_entry_atol_assumed"])
    areas = parcel_map["Brain Area"].tolist()
    archives = m.resolve_unique_files(a.data_dir, "*FullCovarianceConnectivity.zip", a.results_dir)
    corrarchives = m.resolve_unique_files(a.data_dir, "*FullCorrelationConnectivity.zip", a.results_dir) if a.correlation_scale != "unverified" else {}
    frames = frames_input(a)
    old_structure = pd.read_csv(a.analysis10_dir / "tables" / "02_matrix_structure.csv")
    rows, sensitivity, eigen_rows, comparisons, network_comparisons, audits = [], [], [], [], [], []
    correlation_status = []
    observed_frame_keys = set()
    for state in STATES:
        path = archives[state]
        require(path is not None, "Covariance archive metadata missing for " + state)
        ids = diagnostic_ids(a, m, state, path, areas, amplitudes, old_contract)
        payload = {"archive": m.archive_signature(path), "areas": areas, "event_ids": ids, "version": m.VERSION}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache = a.analysis10_dir / "private" / ("diagnostic_"+key+".npz")
        require(cache.is_file(), "Script-10 covariance checkpoint not found for "+state+". Rerun script 10 --stage diagnostic to rebuild it; no covariance reload was attempted.")
        with np.load(cache, allow_pickle=False) as z:
            require(z["areas"].astype(str).tolist() == areas and z["event_ids"].astype(str).tolist() == ids, "Diagnostic cache alignment mismatch")
            matrices = z["matrices"].copy()
        require(matrices.shape == (len(ids), len(areas), len(areas)), "Diagnostic cache dimensions differ")
        corr = None
        if a.correlation_scale != "unverified" and corrarchives[state] is not None:
            print("Read documented released correlation diagnostic rows: "+state, flush=True)
            corr, _ = m.load_diagnostic_matrices(corrarchives[state], areas, ids, out / "private", force=False)
            if a.correlation_scale == "fisher_z":
                corr = np.tanh(corr)
        correlation_status.append({"state": state, "status": "COMPARED" if corr is not None else "NOT_RUN_SCALE_UNVERIFIED" if a.correlation_scale == "unverified" else "ARCHIVE_UNAVAILABLE"})
        for j, event in enumerate(ids):
            label = state+"_"+str(j+1).zfill(2)
            if frames is not None:
                observed_frame_keys.add((state, label if frames.attrs["key"] == "diagnostic_session" else event))
            av = amplitudes[state]["signal"].loc[event, areas].to_numpy(float)
            nv = amplitudes[state]["noise"].loc[event, areas].to_numpy(float)
            require(np.isfinite(av).all() and np.isfinite(nv).all(), "Nonfinite diagnostic amplitudes")
            total_sd = np.hypot(av, nv)
            c = matrices[j]
            ct = c.copy(); ct[np.diag_indices(len(c))] += nv**2
            frame = frame_annotation(state, label, event, frames)
            t = frame["frames_used"]
            size = len(c)
            nullity = max(0, size-int(t)+1) if pd.notna(t) else np.nan
            for kind, matrix in [("RELEASED_SIGNAL_DIAGONAL", c), ("CANDIDATE_TOTAL", ct)]:
                stats, symmetric = m.symmetry_and_spectrum(matrix, args)
                require(symmetric is not None, "Nonfinite/materially asymmetric matrix: "+label)
                eigen = np.linalg.eigvalsh(symmetric)
                scale = max(float(np.max(np.abs(eigen))), np.finfo(float).tiny)
                bound = float(np.max(nv**2)/t) if pd.notna(t) else np.nan
                exact_bound = float(np.max(nv**2)/(t-1)) if pd.notna(t) else np.nan
                rows.append({"state": state, "diagnostic_session": label, "matrix": kind, **frame,
                    "conditional_centered_sample_nullity_lower_bound": nullity, **stats,
                    "max_noise_variance": float(np.max(nv**2)), "median_candidate_total_diagonal": float(np.median(np.diag(ct))),
                    "lambda_min_abs_over_max_noise_variance": abs(eigen[0])/max(float(np.max(nv**2)), np.finfo(float).tiny),
                    "hypothetical_noise_df_error_max_n2_over_T": bound, "hypothetical_n2_T_to_Tminus1_error_bound": exact_bound,
                    "negative_eigenvalue_magnitude_over_n2_over_T": max(0., -eigen[0])/bound if bound > 0 else np.nan,
                    "negative_eigenvalue_magnitude_over_df_error_bound": max(0., -eigen[0])/exact_bound if exact_bound > 0 else np.nan,
                    "interpretation": "Frame/rank/df annotations are conditional scenarios, not identification of the release algorithm"})
                previous = old_structure.loc[old_structure.state.eq(state) & old_structure.diagnostic_session.eq(label) & old_structure.matrix.eq(kind)]
                require(len(previous) == 1, "Diagnostic label/sample differs from script 10")
                difference = abs(float(eigen[0])-float(previous.iloc[0].eigenvalue_min))
                audits.append({"state": state, "diagnostic_session": label, "matrix": kind,
                    "lambda_min_reproduction_error": difference, "pass": difference < a.reproduction_tol*max(1., scale)})
                for rtol in NEAR_ZERO_RTOLS:
                    threshold = rtol*scale
                    sensitivity.append({"state": state, "diagnostic_session": label, "matrix": kind, "near_zero_rtol": rtol,
                        "near_zero_absolute_threshold": threshold, "negative_beyond_threshold_n": int((eigen < -threshold).sum()),
                        "within_near_zero_band_n": int((np.abs(eigen) <= threshold).sum()),
                        "positive_beyond_threshold_n": int((eigen > threshold).sum()),
                        "conditional_centered_sample_nullity_lower_bound": nullity,
                        "interpretation": "Threshold sensitivity; no half-nullity or twice-negative-count rule"})
                eigen_rows.extend({"state": state, "diagnostic_session": label, "matrix": kind, "ascending_index": i+1,
                    "eigenvalue": value, "eigenvalue_over_spectral_scale": value/scale} for i, value in enumerate(eigen))
            if corr is not None:
                candidate, _, _ = m.pair_ratio((c+c.T)/2, total_sd)
                require(np.allclose(corr[j], corr[j].T, rtol=args.symmetry_rtol, atol=args.symmetry_atol, equal_nan=True),
                        "Materially asymmetric released correlation matrix")
                released_matrix = (corr[j]+corr[j].T)/2
                iu = np.triu_indices(size, 1); released = released_matrix[iu]
                require(np.isfinite(released).all() and (np.abs(released) <= 1.01).all(), "Documented correlation conversion yielded invalid off-diagonal values")
                good = np.isfinite(candidate)
                differences = candidate[good]-released[good]
                comparisons.append({"state": state, "diagnostic_session": label, "numeric_scale": a.correlation_scale,
                    "pair_n": int(good.sum()), "candidate_vs_released_rmse": float(np.sqrt(np.mean(differences**2))),
                    "candidate_vs_released_median_abs_error": float(np.median(np.abs(differences))),
                    "candidate_vs_released_max_abs_error": float(np.max(np.abs(differences)))})
                for network in networks:
                    positions = np.flatnonzero(parcel_map.node.map(m.compact_token).eq(m.compact_token(network)))
                    sub = np.triu_indices(len(positions), 1)
                    weight = np.outer(total_sd[positions], total_sd[positions])[sub]
                    off = ((c+c.T)/2)[np.ix_(positions, positions)][sub]
                    rr = released_matrix[np.ix_(positions, positions)][sub]
                    candidate_network = float(off.mean()/weight.mean())
                    weighted_released = float(np.sum(weight*rr)/np.sum(weight))
                    network_comparisons.append({"state": state, "diagnostic_session": label, "network": network,
                        "candidate_C_over_P_total": candidate_network, "total_amplitude_weighted_released_r": weighted_released,
                        "unweighted_released_r_mean": float(rr.mean()), "candidate_minus_weighted_released": candidate_network-weighted_released,
                        "interpretation": "Network ratio matches an amplitude-product-weighted mean only if pairwise reconstruction matches"})
    if frames is not None:
        supplied_keys = set(zip(frames.state, frames[frames.attrs["key"]]))
        require(supplied_keys.issubset(observed_frame_keys), "Frame table contains keys outside the frozen diagnostic sample; check state/session labels")
    m.write_table(out, "01_matrix_spectrum_followup.csv", rows)
    m.write_table(out, "02_near_zero_threshold_sensitivity.csv", sensitivity)
    spectrum = m.write_table(out, "03_eigenvalue_spectra.csv", eigen_rows)
    m.write_table(out, "04_diagnostic_reproduction.csv", audits)
    require(all(row["pass"] for row in audits), "Matrix diagnostics no longer reproduce script 10")
    m.write_table(out, "05_released_correlation_comparison.csv", comparisons,
        columns=["state", "diagnostic_session", "numeric_scale", "pair_n", "candidate_vs_released_rmse", "candidate_vs_released_median_abs_error", "candidate_vs_released_max_abs_error"])
    m.write_table(out, "06_network_released_correlation_comparison.csv", network_comparisons,
        columns=["state", "diagnostic_session", "network", "candidate_C_over_P_total", "total_amplitude_weighted_released_r", "unweighted_released_r_mean", "candidate_minus_weighted_released", "interpretation"])
    m.write_table(out, "07_correlation_status.csv", correlation_status)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    for ax, state in zip(axes.ravel(), STATES):
        selected = spectrum.loc[spectrum.state.eq(state) & spectrum.matrix.eq("CANDIDATE_TOTAL")]
        for label, group in selected.groupby("diagnostic_session", sort=False):
            group = group.iloc[:min(200, len(group))]
            ax.plot(group.ascending_index, group.eigenvalue_over_spectral_scale, lw=.9, label=label)
        ax.axhline(0, color=".55", lw=.7); ax.set_title(state, fontsize=10)
        ax.set_xlabel("Ascending eigenvalue index"); ax.set_ylabel("Eigenvalue / spectral scale")
        ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); finish_figure(fig, out, "01_candidate_total_low_spectra")
    return {"diagnostic_sessions": len(rows)//2, "construction_uniquely_identified": False,
            "covariance_archives_reread": False, "near_zero_thresholds": NEAR_ZERO_RTOLS,
            "actual_frames_supplied": a.frames_table is not None, "correlation_status": correlation_status}


def write_report(a, m, out, summary):
    lines = ["# Normalization follow-up", "", "Status: " + summary["status"], "",
        "## Outputs", "",
        "- tables/20_common_amplitude_slopes.csv: signal, total, and noise pair-amplitude slopes.",
        "- tables/22_signal_minus_noise_paired_slopes.csv: paired trajectory differences.",
        "- tables/23_amplitude_fit_reproduction_and_identities.csv: coefficient and covariance checks.",
        "- tables/24_cohort_network_attenuation_summary.csv: matched-cohort attenuation summaries.",
        "- tables/30_pooled_network_slopes.csv: equal-state network slopes and joint-covariance intervals.",
        "- tables/01_matrix_spectrum_followup.csv: matrix spectra and frame-count scenarios.",
        "- tables/05_released_correlation_comparison.csv: documented-scale pairwise comparison.",
        "", "## Construction", "",
        "Total amplitude uses the candidate reconstruction t = sqrt(a^2 + n^2).",
        "P_total averages total-amplitude pair products; it is not P_signal + P_noise.",
        "The common and profile families retain their respective signal-model GLS weights.",
        "Frame-count scenarios and near-zero eigenvalue thresholds are recorded separately from measured rank.", ""]
    if summary["status"] != "COMPLETED":
        lines.extend(["Error: "+summary.get("error", ""), ""])
    (out / "normalization_followup_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    a = parse_args(); m = import_script10(a.script10); out = a.output_dir
    require(a.analysis10_dir.is_dir(), "Completed local script-10 output directory is required")
    require(read_json(a.analysis10_dir / "noise_normalization_summary.json")["status"] == "COMPLETED", "Script 10 did not complete")
    contract10 = read_json(a.analysis10_dir / "analysis_contract.json")
    for sub in ["tables", "figures", "private"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out / "private", 0o700)
    except OSError:
        pass
    # Each invocation represents one selected stage set. Remove this script's
    # previous public products so a failed rerun cannot expose stale estimates.
    for directory in ["tables", "figures"]:
        for path in (out / directory).iterdir():
            if path.is_file() and path.suffix in [".csv", ".pdf", ".png"]:
                path.unlink()
    for name in ["normalization_followup_summary.json", "normalization_followup_report.md", "run_manifest.json"]:
        (out / name).unlink(missing_ok=True)
    requested = ["longitudinal", "pooled", "diagnostic"] if a.stage == "all" else [a.stage]
    manifest10 = read_json(a.analysis10_dir / "run_manifest.json")
    contract = {"version": VERSION, "timestamp_utc": m.utc_now(), "requested_stages": requested,
        "arguments": vars(a), "script10_version": m.VERSION, "script10_sha256": m.sha256_file(a.script10),
        "script10_matches_recorded_hash": m.sha256_file(a.script10) == manifest10.get("script_sha256"),
        "analysis10_contract_sha256": m.sha256_file(a.analysis10_dir / "analysis_contract.json"),
        "measurement_semantics_from_script10": {key: contract10.get(key) for key in ["amplitude_units", "noise_decomposition", "measurement_source_sha256"]},
        "near_zero_rtols": NEAR_ZERO_RTOLS, "nominal_task_frames": NOMINAL_TASK_FRAMES, "nominal_frame_source": PROTOCOL_SOURCE,
        "frames_table_sha256": m.sha256_file(a.frames_table) if a.frames_table else None,
        "correlation_source_sha256": m.sha256_file(a.correlation_source) if a.correlation_source else None,
        "new_REML_fit": False, "no_clipping_or_PSD_projection": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    m.save_json(out / "analysis_contract.json", contract)
    summary = {"status": "RUNNING", "requested_stages": requested, "stages": {}}
    m.save_json(out / "normalization_followup_summary.json", summary)
    try:
        inputs = amplitude_inputs(a, m, out) if any(stage in requested for stage in ["longitudinal", "diagnostic"]) else None
        for stage in requested:
            if stage == "longitudinal":
                result = longitudinal_stage(a, m, out, *inputs)
            elif stage == "pooled":
                result = pooled_stage(a, m, out)
            else:
                result = diagnostic_stage(a, m, out, *inputs)
            summary["stages"][stage] = result
            m.save_json(out / "normalization_followup_summary.json", summary)
        summary["status"] = "COMPLETED"
    except Exception as exc:
        summary.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        m.save_json(out / "normalization_followup_summary.json", summary)
        write_report(a, m, out, summary)
        public = [path for path in out.rglob("*") if path.is_file() and "private" not in path.relative_to(out).parts and path.name != "run_manifest.json"]
        m.save_json(out / "run_manifest.json", {"version": VERSION, "status": summary["status"], "timestamp_utc": m.utc_now(),
            "script_sha256": m.sha256_file(Path(__file__)), "files": [{"path": str(path.relative_to(out)),
                "sha256": m.sha256_file(path), "bytes": path.stat().st_size} for path in sorted(public)],
            "private_directory": "Local checkpoints only; exclude from shared exports"})
    print("Completed. Results: "+str(out), flush=True)


if __name__ == "__main__":
    main()
