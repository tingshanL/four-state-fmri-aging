#!/usr/bin/env python3
"""Build supplementary Tables S1--S13 as LaTeX from analysis result tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["REST", "CARIT", "FACENAME", "VISMOTOR"]
SL = {"REST": "REST", "CARIT": "CARIT", "FACENAME": "FaceName", "VISMOTOR": "Visual--motor"}
SITES = ["MGH", "UCLA", "UMinn", "WashU"]
FP = "Frontoparietal"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def n(x, d=3, sign=True):
    """Signed number in math mode with a true minus."""
    x = float(x)
    while x != 0 and abs(x) < 0.5 * 10 ** (-d) and d < 4:
        d += 1  # show one more decimal rather than a signed zero
    if abs(x) < 0.5 * 10 ** (-d):
        x = 0.0
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return f"${s}$"


def ci(lo, hi, d=3, sign=True):
    return f"[{n(lo, d, sign)}, {n(hi, d, sign)}]"


def est_ci(e, lo, hi, d=3, sign=True):
    return f"{n(e, d, sign)} {ci(lo, hi, d, sign)}"


def p(x):
    x = float(x)
    if x < 0.001:
        m, e = f"{x:.2e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    if x < 0.1:
        return f"${x:.4f}$".replace("0.", ".", 1)
    return f"${x:.3f}$".replace("0.", ".", 1)


def r_(x, d=3):
    s = f"{float(x):.{d}f}"
    s = s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    return f"${s.replace('-', '-')}$"


def table(label, caption, colspec, header, rows, size="\\footnotesize", resize=False, notes=None):
    body = [f"\\begin{{table}}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    tab = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    if header:
        tab += [header + " \\\\", "\\midrule"]
    for row in rows:
        done = row.startswith("\\") or row.rstrip().endswith("\\\\")
        tab.append(row if done else row + " \\\\")
    tab += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        tab = ["\\begin{adjustbox}{max width=\\textwidth}"] + tab + ["\\end{adjustbox}"]
    body += tab
    if notes:
        body += ["\\par\\smallskip", f"\\parbox{{\\textwidth}}{{\\footnotesize {notes}}}"]
    body.append("\\end{table}")
    return "\n".join(body) + "\n"


def panel(title, ncol):
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{title}}}}} \\\\"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
class R:
    def __init__(self, a):
        self.a = a

    def __call__(self, where: str, rel: str) -> pd.DataFrame:
        base = {"res": self.a.results_dir, "supp": self.a.supplement_tables,
                "a10": self.a.analysis10_tables, "a11": self.a.analysis11_tables}[where]
        path = base / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path)


def matched_partial(rd) -> pd.DataFrame:
    """FP-relative slopes (10/9 * R_FP) from the profile fits on partial-covariance rows."""
    from statistics import NormalDist
    sens = rd("res", "02_primary_profile_and_measurement/tables/sensitivity_network_slopes.csv")
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(STATES)))
    out = []
    for sc in ("FULL_PARTIAL_MATCHED", "PARTIAL_MATCHED"):
        for s in STATES:
            r = sens[(sens.scenario == sc) & (sens.network == FP) & (sens.state == s)].iloc[0]
            est, se = r.relative_profile_slope_per_5y * 10 / 9, r.se * 10 / 9
            out.append(dict(scenario=sc, state=s, est=est, se=se, lo=est - z * se, hi=est + z * se))
    return pd.DataFrame(out).set_index(["scenario", "state"])


def networks(rd) -> list[str]:
    s5 = rd("supp", "Table_S05_network_profile_and_convergence.csv")
    nets = list(dict.fromkeys(s5[s5.table_component.eq("network_profile")].network))
    assert len(nets) == 10 and FP in nets
    return nets


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def table_s1(rd) -> str:
    ratio = rd("a10", "04_pairwise_normalization.csv")
    spec = rd("a11", "01_matrix_spectrum_followup.csv")
    rows = []
    for s in STATES:
        for i in (1, 2, 3):
            sess = f"{s}_{i:02d}"
            rs = ratio[(ratio.diagnostic_session == sess) & (ratio.representation == "SIGNAL_NORMALIZED")].iloc[0]
            rt = ratio[(ratio.diagnostic_session == sess) & (ratio.representation == "CANDIDATE_TOTAL_NORMALIZED")].iloc[0]
            es = spec[(spec.diagnostic_session == sess) & (spec.matrix == "RELEASED_SIGNAL_DIAGONAL")].iloc[0]
            et = spec[(spec.diagnostic_session == sess) & (spec.matrix == "CANDIDATE_TOTAL")].iloc[0]
            frames = "--" if pd.isna(es.frames_used) else f"{int(es.frames_used)}"
            rows.append(" & ".join([
                f"{SL[s]} {i}", frames,
                f"[{n(rs.ratio_min, 2, False)}, {n(rs.ratio_max, 2, False)}]",
                f"{100 * rs.fraction_abs_gt_1_01:.1f}",
                f"[{n(rt.ratio_min, 2, False)}, {n(rt.ratio_max, 2, False)}]",
                f"{int(es.negative_eigenvalue_n)}", n(es.eigenvalue_min_over_spectral_scale, 4, False),
                f"{int(et.negative_eigenvalue_n)}", n(et.eigenvalue_min_over_spectral_scale, 4, False),
            ]))
        if s != STATES[-1]:
            rows.append("\\addlinespace[2pt]")
    header = ("Session & $T$ & \\multicolumn{2}{c}{$C_{pq}/(a_pa_q)$} & $C_{pq}/(t_pt_q)$ & "
              "\\multicolumn{2}{c}{Released $C$} & \\multicolumn{2}{c}{$C+\\mathrm{diag}(n_p^2)$} \\\\\n"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
              " & & Range & \\% $|\\cdot|>1.01$ & Range & $n_{<0}$ & $\\lambda_{\\min}/\\lambda_{\\max}$ & "
              "$n_{<0}$ & $\\lambda_{\\min}/\\lambda_{\\max}$")
    cap = ("Measurement diagnostics in 12 diagnostic sessions (three per state). Each session has 360 parcels and "
           "64,620 parcel pairs. $a_p$, released signal amplitude; $n_p$, released noise amplitude; "
           "$t_p=\\sqrt{a_p^2+n_p^2}$, candidate total amplitude. $T$ is the nominal number of frames for task runs "
           "(Harms et al., 2018); the REST aggregation was not inferred. $n_{<0}$, number of negative eigenvalues "
           "(of 360); $\\lambda_{\\min}/\\lambda_{\\max}$, smallest eigenvalue divided by the largest. Diagnostic "
           "sessions describe the construction of the released products and are not cohort prevalence estimates.")
    return table("tab:supp-measurement", cap, "lrrrrrrrr", header, rows, resize=True)


def table_s2(rd, nets) -> str:
    att = rd("a11", "24_cohort_network_attenuation_summary.csv")
    rows = []
    for net in nets:
        cells = []
        for s in STATES:
            r = att[(att.network == net) & (att.state == s)].iloc[0]
            cells.append(f"{r.P_signal_over_P_total_median:.2f} [{r.P_signal_over_P_total_p05:.2f}, "
                         f"{r.P_signal_over_P_total_p95:.2f}]")
        rows.append(f"{net} & " + " & ".join(cells))
    nobs = [int(att[att.state == s].observation_n.iloc[0]) for s in STATES]
    header = "Network & " + " & ".join(f"{SL[s]} ($n={k:,}$)".replace(",", "{,}", 1) for s, k in zip(STATES, nobs))
    cap = ("Cohort distribution of the network signal-to-candidate-total pair-amplitude ratio "
           "$P_k/P_k^{\\mathrm{TOT}}=\\exp(H_k)$. Entries are the median [5th, 95th percentile] across matched "
           "session--state observations (852 participants). The ratio compares mean parcel-pair amplitude "
           "products; it is not a mean parcel signal-variance fraction.")
    return table("tab:supp-attenuation", cap, "lcccc", header, rows, resize=True)


def table_s3(rd) -> str:
    s3 = rd("supp", "Table_S03_common_covariance_and_pair_amplitude_models.csv")
    ordm = s3[s3.table_component.eq("ordinary_models")]
    q = rd("res", "02_primary_profile_and_measurement/tables/gate1_residual_fraction_intervals.csv").set_index("state")
    amp = rd("a11", "20_common_amplitude_slopes.csv")
    pair = rd("a11", "22_signal_minus_noise_paired_slopes.csv").set_index("state")
    omni11 = rd("a11", "21_common_amplitude_omnibus_tests.csv").set_index("metric")
    sl10 = rd("a10", "20_longitudinal_slopes.csv")
    om10 = rd("a10", "21_omnibus_tests.csv")
    rows = [panel("A. Native common slope and its fixed-weight decomposition (native-model weights)", 6)]
    for s in STATES:
        raw = ordm[(ordm.scenario == "RAW_MATCHED") & (ordm.state == s)].iloc[0]
        pac = ordm[(ordm.scenario == "PAIR_AMPLITUDE_COMPONENT") & (ordm.state == s)].iloc[0]
        qq = q.loc[s]
        assert abs(qq.raw_fixed_weight_slope_per_5y - raw.slope_per_5y) < 1e-6
        b_a = qq.raw_fixed_weight_slope_per_5y - qq.standardized_fixed_weight_slope_per_5y
        rows.append(" & ".join([SL[s], est_ci(raw.slope_per_5y, raw.simultaneous_ci_low, raw.simultaneous_ci_high),
                                est_ci(pac.slope_per_5y, pac.simultaneous_ci_low, pac.simultaneous_ci_high, 4),
                                n(b_a), n(qq.standardized_fixed_weight_slope_per_5y),
                                f"${qq.q_point:.3f}$ [${qq.simultaneous_95_ci_low:.3f}$, ${qq.simultaneous_95_ci_high:.3f}$]"]))
    rows.insert(1, "State & Native $\\beta^{\\mathrm{RAW}}$ & Pair amplitude, own weights & "
                   "$\\beta^{\\mathrm{A}}$ & $\\beta^{\\mathrm{STD}}$ & $q_s$ \\\\")
    rows.insert(2, "\\midrule")
    ctest = s3[s3.table_component.eq("common_level_tests") & s3.scenario.eq("RAW_AMPLITUDE_MATCHED")].set_index("test")
    rows.append(f"Site$\\times$time; site$\\times$time$\\times$state ($p$) & "
                f"{p(ctest.loc['common_level_site_x_time', 'p'])}; {p(ctest.loc['common_level_site_x_time_x_state', 'p'])}"
                " & & & & ")
    rows.append("\\midrule")

    rows.append(panel("B. Pair amplitudes (signal-model weights)", 6))
    rows.append("State & Signal $\\log P$ & Candidate total $\\log P^{\\mathrm{TOT}}$ & Noise $\\log P^{\\mathrm{N}}$ & Signal $-$ noise & \\\\")
    rows.append("\\midrule")
    for s in STATES + ["EQUAL_STATE_MEAN"]:
        cells = [SL.get(s, "Equal-state mean")]
        for m in ("SIGNAL_PAIR_AMPLITUDE", "TOTAL_PAIR_AMPLITUDE", "NOISE_PAIR_AMPLITUDE"):
            r = amp[(amp.metric == m) & (amp.state == s)].iloc[0]
            cells.append(est_ci(r.slope_per_5y, r.simultaneous_ci_low, r.simultaneous_ci_high))
        r = pair.loc[s]
        cells.append(est_ci(r.slope_per_5y, r.simultaneous_ci_low, r.simultaneous_ci_high))
        cells.append("")
        rows.append(" & ".join(cells))
    rows.append("Four-state $\\chi^2_4$ ($p_{\\mathrm{Holm}}$) & " + " & ".join(
        f"${omni11.loc[m, 'chi2']:.2f}$ ({p(omni11.loc[m, 'p_holm_three_amplitude_tests'])})"
        for m in ("SIGNAL_PAIR_AMPLITUDE", "TOTAL_PAIR_AMPLITUDE", "NOISE_PAIR_AMPLITUDE")) + " & & ")
    rows.append("\\midrule")

    rows.append(panel("C. Common level under signal and candidate total normalization (signal-model weights)", 6))
    rows.append("State & Signal-normalized & $H$ & Candidate total-normalized & & \\\\")
    rows.append("\\midrule")
    com = sl10[sl10.family.eq("COMMON")]
    for s in STATES:
        cells = [SL[s]]
        for c in ("SIGNAL", "ATTENUATION_H", "TOTAL"):
            r = com[(com.component == c) & (com.state == s)].iloc[0]
            cells.append(est_ci(r.slope_per_5y, r.simultaneous_ci_low, r.simultaneous_ci_high))
        rows.append(" & ".join(cells + ["", ""]))
    om = om10[(om10.family == "COMMON") & (om10.test == "all_four_common_level_slopes_zero")].set_index("component")
    rows.append("Four-state $\\chi^2_4$ ($p_{\\mathrm{Holm}}$) & " + " & ".join(
        f"${om.loc[c, 'chi2']:.2f}$ ({p(om.loc[c, 'p_holm_sensitivity_family'])})"
        for c in ("SIGNAL", "ATTENUATION_H", "TOTAL")) + " & & ")
    st = om10[om10.family.eq("COMMON")].set_index(["component", "test"])
    rows.append("Site$\\times$time; site$\\times$time$\\times$state ($p$) & " + " & ".join(
        f"{p(st.loc[(c, 'common_level_site_x_time'), 'p_unadjusted'])}; "
        f"{p(st.loc[(c, 'common_level_site_x_time_x_state'), 'p_unadjusted'])}"
        for c in ("SIGNAL", "ATTENUATION_H", "TOTAL")) + " & & ")
    cap = ("Common-level longitudinal slopes (log units per 5 years) with simultaneous 95\\% intervals. "
           "Panel A: native common slope and the separately fitted pair-amplitude component (Bonferroni across "
           "four states); $\\beta^{\\mathrm{A}}$ and $\\beta^{\\mathrm{STD}}$ are refits of the two components "
           "with the native model's GLS weights, so that $\\beta^{\\mathrm{RAW}}_s=\\beta^{\\mathrm{A}}_s+"
           "\\beta^{\\mathrm{STD}}_s$ exactly; $q_s=|\\beta^{\\mathrm{STD}}_s/\\beta^{\\mathrm{RAW}}_s|$ with a "
           "max-$|t|$ simultaneous band from 4,999 pedigree-cluster multiplier draws. Panel B: log pair-amplitude "
           "products refitted with the signal-normalized common model's weights (Bonferroni across 20 slopes); "
           "signal $-$ noise is the directly paired contrast (Bonferroni across five summaries); Holm adjustment "
           "across the three amplitude omnibus tests. Panel C: signal-normalized and candidate total-normalized "
           "common levels, $H=\\log(P/P^{\\mathrm{TOT}})$, and the identity $L^{\\mathrm{TOT}}=L^{\\mathrm{STD}}+H$ "
           "(Bonferroni across four states within each column); Holm adjustment across the three common-level tests. "
           "Site interaction tests (3 and 9 df) are nominal and descriptive.")
    return table("tab:supp-common", cap, "llllll", None, rows, resize=True)


def table_s4(rd, nets) -> str:
    s5 = rd("supp", "Table_S05_network_profile_and_convergence.csv")
    prof = s5[s5.table_component.eq("network_profile")]
    conv = s5[s5.table_component.eq("convergence")]
    tests = s5[s5.table_component.eq("profile_tests") & s5.scenario.eq("AMPLITUDE_STANDARDIZED")].set_index("test")
    rows = [panel("A. Network-relative slopes [Bonferroni intervals across 40 network--state cells]", 6)]
    rows.append("Network & REST & CARIT & FaceName & Visual--motor & Equal-state \\\\")
    rows.append("\\midrule")
    for net in nets:
        t = prof[prof.network.eq(net)].set_index("state")
        cells = []
        for s in STATES:
            r = t.loc[s]
            mark = "$^{*}$" if (r.simultaneous_ci_low > 0 or r.simultaneous_ci_high < 0) else ""
            cells.append(f"{n(r.relative_profile_slope_per_5y)}{mark} {ci(r.simultaneous_ci_low, r.simultaneous_ci_high)}")
        rows.append(f"{net} & " + " & ".join(cells) + f" & {n(t.equal_state_mean_slope.iloc[0])}")
    rows.append("\\midrule")
    rows.append(panel("B. Fitted-profile geometry [95\\% multiplier intervals]", 6))
    rows.append("State & Baseline level--slope $r$ & 5-year level--slope $r$ & SD ratio, 5 y/baseline & "
                "FP residual from trend & \\\\")
    rows.append("\\midrule")
    for s in STATES + ["EQUAL_STATE_AVERAGE"]:
        c = conv[conv.state.eq(s)].set_index("metric")
        def cc(m, d=3, sign=True):
            r = c.loc[m]
            return est_ci(r.observed, r.bootstrap_q025, r.bootstrap_q975, d, sign)
        rows.append(" & ".join([SL.get(s, "Equal-state average"), cc("baseline_level_slope_r"),
                                cc("five_year_level_slope_r"), cc("sd_ratio_5y_over_baseline", 3, False),
                                cc("fp_residual_from_baseline_trend"), ""]))
    tn = tests.loc["time_x_network"]
    tns = tests.loc["time_x_network_x_state"]
    cap = ("Signal-normalized ten-network profile. Panel A: slopes of the centered log-ratio coordinates $R_k$ "
           "(log units per 5 years); $^{*}$ interval excludes zero. Joint tests: time$\\times$network "
           f"$\\chi^2_{{{int(tn.df)}}}={tn.chi2:.2f}$, $p={p(tn.p).strip('$')}$; time$\\times$network$\\times$state "
           f"$\\chi^2_{{{int(tns.df)}}}={tns.chi2:.2f}$, $p={p(tns.p).strip('$')}$. Panel B: summaries of the fitted "
           "profile recomputed in 4,999 pedigree-cluster multiplier draws of the full coefficient vector; SD is the "
           "population SD across the ten fitted network levels.")
    return table("tab:supp-profile", cap, "llllll", None, rows, resize=True)


def table_s5(rd, nets) -> str:
    s6 = rd("supp", "Table_S06_FP_network_comparisons.csv")
    fw = s6[s6.table_component.eq("four_state_familywise_tests")].set_index("comparator")
    eq = s6[s6.table_component.eq("equal_state_simultaneous_intervals")].set_index("comparator")
    rows = []
    for net in nets:
        if net == FP:
            continue
        f, e = fw.loc[net], eq.loc[net]
        diffs = " & ".join(n(f[f"{s}_difference"]) for s in STATES)
        pfw = f"${f.p_fwer_maxwald:.4f}$".replace("$0.", "$.")
        rows.append(f"{net} & {diffs} & ${f.wald_chi2_observed:.2f}$ & {pfw} & "
                    f"{est_ci(e.equal_state_mean_difference, e.simultaneous_ci_low, e.simultaneous_ci_high, 4)}")
    header = ("Comparator & REST & CARIT & FaceName & Visual--motor & Wald $\\chi^2_4$ & Familywise $p$ & "
              "Equal-state difference [95\\% CI]")
    B = int(fw.bootstrap_B.iloc[0])
    cap = ("FP-minus-comparator slope differences of uncentered signal-normalized network log levels (log units per "
           "5 years). Familywise $p$ values are max-Wald adjusted across the nine four-state trajectory tests "
           f"({B:,} pedigree-cluster multiplier draws; the smallest resolvable value is $1/(B+1)=.0002$). Equal-state "
           "intervals are max-$|t|$ simultaneous across the nine comparators and are given to four decimals because "
           "one bound lies close to zero.")
    return table("tab:supp-fp-comparisons", cap, "lrrrrrcl", header, rows, resize=True)


def table_s6(rd, nets) -> str:
    t = rd("a11", "30_pooled_network_slopes.csv")
    rows = []
    for net in nets:
        cells = []
        for c in ("SIGNAL", "ATTENUATION_H", "TOTAL"):
            r = t[(t.component == c) & (t.network == net)].iloc[0]
            mark = "$^{*}$" if (r.simultaneous_ci_low > 0 or r.simultaneous_ci_high < 0) else ""
            cells.append(f"{n(r.slope_per_5y)}{mark} {ci(r.simultaneous_ci_low, r.simultaneous_ci_high)}")
        rows.append(f"{net} & " + " & ".join(cells))
    header = "Network & Signal-normalized $L_k$ & $H_k$ & Candidate total-normalized $L_k^{\\mathrm{TOT}}$"
    cap = ("Equal-state slopes of uncentered network log levels (log units per 5 years), from the profile model's "
           "weights, with Bonferroni intervals across the 30 entries; $^{*}$ interval excludes zero. "
           "$L^{\\mathrm{TOT}}_k=L_k+H_k$. These slopes combine common and network-relative change; an interval that "
           "includes zero does not establish maintenance. They are not comparable to the separately weighted "
           "common-level slopes in Table~\\ref{tab:supp-common}.")
    return table("tab:supp-network-levels", cap, "llll", header, rows, resize=True)


def table_norm(rd, nets) -> str:
    sl = rd("a10", "20_longitudinal_slopes.csv")
    om = rd("a10", "21_omnibus_tests.csv")
    pdiff = rd("a10", "22_paired_representation_differences.csv")
    prof = om[om.family.eq("PROFILE")].set_index(["component", "test"])
    comps = (("SIGNAL", "Signal-normalized"), ("ATTENUATION_H", "$H$ (total $-$ signal)"),
             ("TOTAL", "Candidate total-normalized"))
    rows = [panel("A. Joint tests (profile tests Holm-adjusted across six; FP tests nominal)", 6),
            "Representation & Time$\\times$network & Time$\\times$network$\\times$state & "
            "FP-relative, four states & FP-relative, state heterogeneity & \\\\", "\\midrule"]
    def ht(c, t):
        r = prof.loc[(c, t)]
        return f"$\\chi^2_{{{int(r.df)}}}={r.chi2:.2f}$, $p_{{\\mathrm{{Holm}}}}={p(r.p_holm_sensitivity_family).strip('$')}$"
    def nt(c, t):
        r = prof.loc[(c, t)]
        return f"$\\chi^2_{{{int(r.df)}}}={r.chi2:.2f}$, $p={p(r.p_unadjusted).strip('$')}$"
    for c, lab in comps:
        rows.append(f"{lab} & {ht(c, 'time_x_network')} & {ht(c, 'time_x_network_x_state')} & "
                    f"{nt(c, 'FP_all_four_state_slopes_zero')} & {nt(c, 'FP_state_heterogeneity')} & ")
    rows += ["\\midrule", panel("B. FP-relative slopes [simultaneous 95\\% CI across four states]", 6),
             "State & Signal-normalized & Candidate total-normalized & Paired difference ($H$) & Paired $p$ & \\\\",
             "\\midrule"]
    fpr = sl[sl.target.eq("FP_RELATIVE")]
    for s in STATES:
        cells = [SL[s]]
        for c in ("SIGNAL", "TOTAL", "ATTENUATION_H"):
            r = fpr[(fpr.component == c) & (fpr.state == s)].iloc[0]
            cells.append(est_ci(r.slope_per_5y, r.simultaneous_ci_low, r.simultaneous_ci_high))
        d = pdiff[(pdiff.target == "FP_RELATIVE") & (pdiff.state == s)].iloc[0]
        h = fpr[(fpr.component == "ATTENUATION_H") & (fpr.state == s)].iloc[0]
        assert abs(d.slope_difference_per_5y - h.slope_per_5y) < 1e-9
        cells += [p(d.p_unadjusted), ""]
        rows.append(" & ".join(cells))
    rows += ["\\midrule", panel("C. Network-relative slopes under candidate total normalization "
                                 "[Bonferroni across 40 cells]", 6),
             "Network & REST & CARIT & FaceName & Visual--motor & Equal-state \\\\", "\\midrule"]
    tot = sl[(sl.target == "NETWORK_RELATIVE") & (sl.component == "TOTAL")]
    for net in nets:
        t = tot[tot.network.eq(net)].set_index("state")
        cells = []
        for s in STATES:
            r = t.loc[s]
            mark = "$^{*}$" if (r.simultaneous_ci_low > 0 or r.simultaneous_ci_high < 0) else ""
            cells.append(f"{n(r.slope_per_5y)}{mark} {ci(r.simultaneous_ci_low, r.simultaneous_ci_high)}")
        rows.append(f"{net} & " + " & ".join(cells) + f" & {n(t.loc[STATES, 'slope_per_5y'].mean())}")
    cap = ("Signal versus candidate total normalization of the network profile (log units per 5 years). All three "
           "representations use the signal-normalized profile model's GLS weights, so "
           "$L^{\\mathrm{TOT}}=L^{\\mathrm{STD}}+H$ holds exactly for every estimate and the paired difference "
           "equals the $H$ slope. Panel A: the six profile tests form one Holm family; FP-relative tests are nominal; FP was "
           "characterized after examining the profile in this cohort. Panel B: the paired $p$ is unadjusted; the four-state test of the "
           "paired differences is the $H$ row of Panel A. Panel C: $^{*}$ interval excludes zero; the equal-state "
           "column is the mean of the four point estimates. The corresponding signal-normalized estimates are in "
           "Table~\\ref{tab:supp-profile}A.")
    return table("tab:supp-normalization", cap, "llllll", None, rows, resize=True)


def table_s7(rd) -> str:
    sim = rd("res", "07_site_profile_shape_diagnostic/tables/04_site_equal_state_similarity.csv").set_index("label")
    mag = rd("res", "07_site_profile_shape_diagnostic/tables/05_site_equal_state_magnitude.csv").set_index("site")
    fp = rd("res", "07_site_profile_shape_diagnostic/tables/06_site_FP_relative_slopes.csv").set_index("site")
    rows = [panel("A. Each site", 5),
            "Site & Profile Wald $p$ & $r$ with other three & Site/pooled RMS & FP-relative slope \\\\", "\\midrule"]
    for s in SITES:
        a, m, f = sim.loc[f"{s}__vs__OTHER_THREE"], mag.loc[s], fp.loc[s]
        rows.append(" & ".join([s, p(m.descriptive_profile_p),
                                est_ci(a.estimate, a.simultaneous_95_low, a.simultaneous_95_high, 3, False),
                                est_ci(m.ratio_estimate, m.ratio_simultaneous_95_low, m.ratio_simultaneous_95_high, 3, False),
                                est_ci(f.fp_relative_slope_per_5y, f.simultaneous_95_low, f.simultaneous_95_high)]))
    rows += ["\\midrule", panel("B. Site pairs", 5), "Pair & $r$ [95\\% CI] & & & \\\\", "\\midrule"]
    pairs = sim[sim.comparison_type.eq("site_pair")]
    for lab, r in pairs.iterrows():
        a, b = lab.split("__vs__")
        rows.append(f"{a}--{b} & {est_ci(r.estimate, r.simultaneous_95_low, r.simultaneous_95_high, 3, False)} & & & ")
    s5 = rd("supp", "Table_S05_network_profile_and_convergence.csv")
    pt = s5[s5.table_component.eq("profile_tests")].set_index(["scenario", "test"])
    om10 = rd("a10", "21_omnibus_tests.csv")
    tot = om10[(om10.family == "PROFILE") & (om10.component == "TOTAL")].set_index("test")
    rows += ["\\midrule", panel("C. Site heterogeneity of the ten-network profile (nominal tests)", 5),
             "Representation & Site$\\times$time$\\times$network & Site$\\times$time$\\times$state$\\times$network & & \\\\",
             "\\midrule"]
    def cell(chi2, df, pv):
        return f"$\\chi^2_{{{int(df)}}}={chi2:.2f}$, $p={p(pv).strip('$')}$"
    for lab, get in (("Native", lambda t: pt.loc[("FULL_PRIMARY", t)]),
                     ("Signal-normalized", lambda t: pt.loc[("AMPLITUDE_STANDARDIZED", t)]),
                     ("Candidate total-normalized", lambda t: tot.loc[t].rename({"p_unadjusted": "p"}))):
        a_, b_ = get("site_x_time_x_network"), get("site_x_time_x_state_x_network")
        rows.append(f"{lab} & {cell(a_.chi2, a_.df, a_.p)} & {cell(b_.chi2, b_.df, b_.p)} & & ")
    cap = ("Model-implied equal-state site profiles, extracted as contrasts from the pooled joint profile model "
           "(signal normalization). Correlation intervals are simultaneous within the ten-correlation family "
           "(Fisher-$z$ scale), RMS-ratio intervals within the four ratios (log scale), and FP-relative intervals "
           "within the four sites; all from 4,999 pedigree-cluster multiplier draws. Site-specific profile Wald "
           "tests (9 df) are reported as nominal descriptive tests. Panel C gives the site interaction tests "
           "of the pooled profile models; they are nominal and not part of the Holm family.")
    return table("tab:supp-site", cap, "lcccc", None, rows, resize=True)


def table_s8(rd) -> str:
    comp = rd("res", "07_site_profile_shape_diagnostic/tables/01_site_composition.csv").set_index("site")
    cov = rd("res", "07_site_profile_shape_diagnostic/tables/01b_site_state_coverage.csv")
    rows = []
    for s in SITES:
        c = comp.loc[s]
        st = [f"{int(cov[(cov.site == s) & (cov.state == x)].participants.iloc[0])}/"
              f"{int(cov[(cov.site == s) & (cov.state == x)].rows.iloc[0])}" for x in STATES]
        rows.append(" & ".join([s, f"{int(c.participants)}", f"{int(c.sessions)}", f"{int(c.session_state_rows):,}",
                                f"{c.baseline_age_mean:.1f} [{int(c.baseline_age_min)}, {int(c.baseline_age_max)}]",
                                f"{c.followup_median:.2f} [{c.followup_q1:.2f}, {c.followup_q3:.2f}]",
                                f"{int(c.scanner_changed_participants)}"] + st))
    tot = comp.sum(numeric_only=True)
    rows.append("\\midrule")
    rows.append(f"Total & {int(tot.participants)} & {int(tot.sessions):,} & {int(tot.session_state_rows):,} & & & "
                f"{int(tot.scanner_changed_participants)} & & & & ")
    header = ("Site & Participants & Sessions & Analyzed rows & Baseline age, mean [range] & Follow-up, y & "
              "Scanner changes & REST & CARIT & FaceName & Visual--motor")
    cap = ("Site composition and state coverage. Analyzed rows are session--state observations (8,727 in total). "
           "Follow-up is the median [interquartile range] in years. State entries are participants/analyzed rows. "
           "All participant identifiers had HCA prefixes; the event-level study label (HCA or AABC) was treated as "
           "visit-level provenance.")
    return table("tab:supp-site-composition", cap, "lrrrccrcccc", header, rows,
                 resize=True)


def table_s9(rd, nets) -> str:
    s8 = rd("supp", "Table_S08_scanner_site_reference_QC_and_partial_covariance.csv")
    ref = s8[s8.table_component.eq("reference_sensitivity")].set_index("reference_choice")
    s6 = rd("supp", "Table_S06_FP_network_comparisons.csv")
    fps = s6[s6.table_component.eq("fp_state_slopes")]
    fpt = s6[s6.table_component.eq("fp_tests")]
    coh = rd("res", "02_primary_profile_and_measurement/tables/cohort_summary.csv").set_index("cohort")
    rows = [panel("A. Reference-network deletion (equal-state FP-relative slope; joint four-state test)", 4),
            "Reference & Equal-state slope & Change from all nine & Joint $p$ \\\\", "\\midrule"]
    order = ["ALL_NINE"] + [f"LEAVE_OUT_{x}" for x in nets if x != FP]
    for k in order:
        r = ref.loc[k]
        lab = "All nine" if k == "ALL_NINE" else f"Without {k.replace('LEAVE_OUT_', '')}"
        rows.append(f"{lab} & {n(r.mean_four_state_slope, 4)} & {n(r.change_from_all_nine_mean, 4)} & {p(r.joint_p)}")
    mp = matched_partial(rd)
    rows += ["\\midrule",
             panel("B. Native full versus native partial covariance on the same rows (profile-model FP-relative slopes)", 4),
             "State & Full covariance & Partial covariance & \\\\", "\\midrule"]
    for s in STATES:
        f, q = mp.loc[("FULL_PARTIAL_MATCHED", s)], mp.loc[("PARTIAL_MATCHED", s)]
        rows.append(f"{SL[s]} & {est_ci(f.est, f.lo, f.hi)} & {est_ci(q.est, q.lo, q.hi)} & ")
    rows.append(f"Equal-state mean (point estimate) & {n(mp.loc['FULL_PARTIAL_MATCHED'].est.mean())} & "
                f"{n(mp.loc['PARTIAL_MATCHED'].est.mean())} & ")
    s5 = rd("supp", "Table_S05_network_profile_and_convergence.csv")
    pt = s5[s5.table_component.eq("profile_tests")].set_index(["scenario", "test"])
    for t_, lab in (("time_x_network", "Profile time$\\times$network"),
                    ("time_x_network_x_state", "Profile time$\\times$network$\\times$state")):
        a_, b_ = pt.loc[("FULL_PARTIAL_MATCHED", t_)], pt.loc[("PARTIAL_MATCHED", t_)]
        rows.append(f"{lab} & $\\chi^2_{{{int(a_.df)}}}={a_.chi2:.2f}$, $p={p(a_.p).strip('$')}$ & "
                    f"$\\chi^2_{{{int(b_.df)}}}={b_.chi2:.2f}$, $p={p(b_.p).strip('$')}$ & ")
    c_part = coh.loc["PARTIAL_MATCHED"]
    assert (coh.loc["FULL_PARTIAL_MATCHED"][["participants", "sessions", "session_state_rows"]] ==
            c_part[["participants", "sessions", "session_state_rows"]]).all()
    rows.append(f"Participants/sessions/rows & \\multicolumn{{2}}{{c}}{{{int(c_part.participants)}/"
                f"{int(c_part.sessions):,}/{int(c_part.session_state_rows):,} (both columns)}} & ")
    jt = lambda sc, test: fpt[(fpt.scenario == sc) & (fpt.test == test)].iloc[0]  # noqa: E731
    q0 = jt("PARTIAL_MATCHED", "all_four_state_slopes_zero")
    rows += ["\\midrule", panel("C. Site interactions in the FP-relative model ($p$)", 4),
             "Representation & Site$\\times$time & Site$\\times$time$\\times$state & \\\\", "\\midrule"]
    for sc, lab in (("RAW_FULL", "Native"), ("AMPLITUDE_STANDARDIZED", "Signal-normalized"),
                    ("PARTIAL_MATCHED", "Partial covariance")):
        a, b = jt(sc, "site_x_time"), jt(sc, "site_x_time_x_state")
        rows.append(f"{lab} & {p(a.p)} ($\\chi^2_{{{int(a.df)}}}={a.chi2:.2f}$) & {p(b.p)} "
                    f"($\\chi^2_{{{int(b.df)}}}={b.chi2:.2f}$) & ")
    cap = ("FP-relative robustness (log units per 5 years). Panel A: the nine-network reference recomputed after "
           "deleting each network; values come from the joint profile fit. Panel B: both columns are native (not "
           "amplitude-normalized) representations fitted to the same rows with available partial covariance; "
           "FP-relative slopes are $\\tfrac{10}{9}R_{\\mathrm{FP}}$ from the ten-network profile models, with "
           "Bonferroni intervals across the four states computed from the scaled standard errors. The difference "
           "between columns is descriptive; no paired test was performed. In the direct FP-minus-nine model, the "
           f"partial-covariance four-state test was $\\chi^2_4={q0.chi2:.2f}$, $p={p(q0.p).strip('$')}$. Panel C: nominal site interaction tests.")
    return table("tab:supp-robustness", cap, "llll", None, rows, resize=True)


def table_s10(rd) -> str:
    st = rd("res", "04_structural_sensitivity/tables/structural_slopes.csv").set_index("outcome")
    area = rd("res", "04_structural_sensitivity/tables/surface_area_relative_slope.csv").iloc[0]
    chg = rd("res", "04_structural_sensitivity/tables/fmri_coefficient_change_joint.csv").set_index("comparison")
    fm = rd("res", "04_structural_sensitivity/tables/fmri_state_slopes.csv")
    tests = rd("res", "04_structural_sensitivity/tables/fmri_tests.csv")
    summ = rd("res", "04_structural_sensitivity/tables/fmri_model_summary.csv").set_index("scenario")
    coef = rd("supp", "Table_S10_structural_fMRI_coefficients.csv")
    rows = [panel("A. FP-relative structural trajectories", 5),
            "Measure & Slope per 5 years & 95\\% CI & $p$ & \\\\", "\\midrule"]
    for lab, r in (("Cortical thickness (mm)", st.loc["Y_THICK_REL"]),
                   ("ICV-adjusted log volume", st.loc["Y_VOLUME_REL"]), ("Log surface area", area)):
        rows.append(f"{lab} & {n(r.slope_per_5y, 5 if abs(r.slope_per_5y) < 0.01 else 4)} & "
                    f"{ci(r.ci_low, r.ci_high, 5 if abs(r.slope_per_5y) < 0.01 else 4)} & {p(r.p)} & ")
    rows += ["\\midrule", panel("B. Change in the four-state FP-relative fMRI slope vector", 5),
             "Adjustment & Norm change & Joint $\\chi^2_4$ & $p$ & \\\\", "\\midrule"]
    for k, lab in (("THICKNESS_MEAN_MINUS_ORIGINAL", "Between-person thickness"),
                   ("THICKNESS_FULL_MINUS_ORIGINAL", "Within- and between-person thickness"),
                   ("VOLUME_FULL_MINUS_ORIGINAL", "Within- and between-person volume")):
        r = chg.loc[k]
        pct = 100 * (r.adjusted_vector_l2 / r.original_vector_l2 - 1)
        rows.append(f"{lab} & {n(pct, 1)}\\% & ${r.chi2:.2f}$ & {p(r.p)} & ")
    rows += ["\\midrule", panel("C. Thickness-adjusted FP-relative fMRI slopes in restricted samples", 5),
             "Sample & " + " & ".join(SL[s] for s in STATES) + " \\\\", "\\midrule"]
    for sc, lab in [("THICKNESS_WITHIN_BETWEEN", "All sites"), ("THICKNESS_FINAL_SAME_SCANNER", "Same scanner")] + \
                   [(f"THICKNESS_FINAL_LOO_{s.upper()}", f"Without {s}") for s in SITES]:
        t = fm[fm.scenario.eq(sc)].set_index("state")
        cells = []
        for s in STATES:
            r = t.loc[s]
            mark = "" if r.simultaneous_ci_low > 0 else "$^{\\dagger}$"
            cells.append(f"{n(r.slope_per_5y, 4)}{mark}")
        j = tests[(tests.scenario == sc) & (tests.test == "all_four_state_slopes_zero")].iloc[0]
        m = summ.loc[sc]
        rows.append(f"{lab} ({int(m.participants)}/{int(m.rows):,}; $p={p(j.p).strip('$')}$) & " + " & ".join(cells))
    rows += ["\\midrule", panel("D. Structural coefficients in the FP-relative fMRI model (per 1 SD)", 5),
             "Component & " + " & ".join(SL[s] for s in STATES) + " \\\\", "\\midrule"]
    cc = coef[coef.table_component.eq("coefficients")]
    jt = coef[coef.table_component.ne("coefficients")]
    for sc, comp, lab in (("THICKNESS_WITHIN_BETWEEN", "thickness_between_z", "Thickness, between"),
                          ("THICKNESS_WITHIN_BETWEEN", "thickness_within_z", "Thickness, within"),
                          ("VOLUME_WITHIN_BETWEEN", "volume_between_z", "Volume, between"),
                          ("VOLUME_WITHIN_BETWEEN", "volume_within_z", "Volume, within")):
        t = cc[(cc.scenario == sc) & (cc.structure_component == comp)].set_index("state")
        jj = jt[(jt.scenario == sc) & (jt.structure_component == comp) &
                (jt.test == "all_four_component_coefficients_zero")]
        jp = f"; joint $p={p(jj.p.iloc[0]).strip('$')}$" if len(jj) else ""
        rows.append(f"{lab}{jp} & " + " & ".join(n(t.loc[s, 'coefficient_per_1sd'], 4) for s in STATES))
    cap = ("Structural sensitivity. Positive relative structural slopes indicate a more positive FP trajectory than "
           "the nine-network reference. Panel B: norm change is "
           "$100(\\lVert\\beta_{\\mathrm{adj}}\\rVert/\\lVert\\beta_{\\mathrm{unadj}}\\rVert-1)$ for the four-state "
           "FP-relative fMRI slope vectors on matched rows; the joint test is a 4-df Wald test that the four "
           "adjusted-minus-unadjusted slope differences are all zero, with their covariance computed from the "
           "joint pedigree-clustered influence functions of the two fits. Panel C gives participants/rows and the four-state joint test; "
           "$^{\\dagger}$ simultaneous interval includes zero. Panel D: within- and between-person structural terms "
           "from the adjusted models.")
    return table("tab:supp-structure", cap, "lllll", None, rows, resize=True)


def table_s11(rd) -> str:
    t = rd("supp", "Table_S07_age_moderation.csv")
    j = t[t.table_component.eq("joint_tests")]
    labels = {"AMPLITUDE_STANDARDIZED_TEN_NETWORK_PROFILE": "Signal-normalized ten-network profile: age$\\times$time$\\times$network (equal-state)",
              "FP_RELATIVE_AMPLITUDE_STANDARDIZED": "Signal-normalized FP-relative trajectory: age$\\times$time$\\times$state",
              "NATIVE_COMMON_COVARIANCE": "Native common covariance: age$\\times$time$\\times$state",
              "AMPLITUDE_STANDARDIZED_COMMON_COVARIANCE": "Signal-normalized common level: age$\\times$time$\\times$state"}
    rows = [f"{labels[r.analysis]} & ${r.chi2:.2f}$ & {int(r.df)} & {p(r.p)}" for _, r in j.iterrows()]
    return table("tab:supp-age", "Joint tests of baseline-age moderation of longitudinal change.", "p{0.62\\textwidth}rrr",
                 "Analysis & $\\chi^2$ & df & $p$", rows, size="\\small")


def table_s12(rd) -> str:
    s11 = rd("supp", "Table_S11_cognition_precision_context_and_selection.csv")
    wp = s11[s11.table_component.eq("within_person_precision")].set_index("state")
    wb = s11[s11.table_component.eq("within_between_coefficients")]
    jt = s11[s11.table_component.eq("joint_tests")].set_index("test")
    bal = s11[s11.table_component.eq("selection_balance")]
    var = s11[s11.table_component.eq("fluidiq_variance")]
    rows = [panel("A. FluidIQ coefficients [simultaneous 95\\% CI across four states]", 4),
            "State & Within, standardized & Within, per 0.1 log & Between, per 0.1 log \\\\", "\\midrule"]
    for s in STATES:
        w = wp.loc[s]
        b = wb[(wb.component == "BETWEEN_PERSON") & (wb.state == s)].iloc[0]
        rows.append(" & ".join([SL[s],
                                est_ci(w.estimate_standardized_using_observed_within_person_sds,
                                       w.ci_low_standardized_using_observed_within_person_sds,
                                       w.ci_high_standardized_using_observed_within_person_sds),
                                est_ci(w.estimate_fluidiq_points_per_0_1_log, w.ci_low_fluidiq_points_per_0_1_log,
                                       w.ci_high_fluidiq_points_per_0_1_log),
                                est_ci(b.coefficient_per_0_1_log_unit, b.simultaneous_ci_low / 10, b.simultaneous_ci_high / 10)]))
    a, b_, c = (jt.loc["FOUR_WITHIN_PERSON_COUPLINGS_ZERO"], jt.loc["WITHIN_PERSON_COUPLINGS_EQUAL_ACROSS_STATES"],
                jt.loc["FOUR_BETWEEN_PERSON_ASSOCIATIONS_ZERO"])
    rows.append(f"Joint tests & \\multicolumn{{2}}{{l}}{{within $\\chi^2_4={a.chi2:.2f}$, $p={p(a.p).strip('$')}$; "
                f"equality $\\chi^2_3={b_.chi2:.2f}$, $p={p(b_.p).strip('$')}$}} & $\\chi^2_4={c.chi2:.2f}$, "
                f"$p={p(c.p).strip('$')}$")
    rows += ["\\midrule", panel("B. Repeated-cognition selection", 4),
             "Variable & Included ($n=746$) & Outside subset & Standardized difference \\\\", "\\midrule"]
    spec = [("n_eligible_fmri_visits", None, "Eligible fMRI visits, mean", 2),
            ("fmri_followup_years", None, "Imaging follow-up, years", 2),
            ("baseline_age", None, "Baseline age, years", 1),
            ("sex", "F", "Female, proportion", 3),
            ("first_equal_state_mean_fp_relative", None, "First equal-state FP-relative value", 3)]
    for v, lev, lab, d in spec:
        r = bal[bal.variable.eq(v)]
        r = (r[r.level.eq(lev)] if lev else r[r.level.isna()]).iloc[0]
        rows.append(f"{lab} & {n(r.included_value, d, False)} & {n(r.excluded_value, d, False)} & "
                    f"{n(r.standardized_difference, 2)}")
    vr = var[var.record_type.eq("VARIANCE_RATIO")].iloc[0]
    rows.append(f"FluidIQ variance ratio, $\\geq2$ vs.\\ 1 paired visit & \\multicolumn{{3}}{{l}}{{"
                f"${vr.variance_ratio_two_plus_over_one:.3f}$ [${vr.variance_ratio_bootstrap_ci_low:.3f}$, "
                f"${vr.variance_ratio_bootstrap_ci_high:.3f}$]}}")
    cap = ("Cognition coupling and repeated-cognition selection. Within-person standardized coefficients use the "
           "observed within-person SDs of FluidIQ and the FP-relative measure; they are regression coefficients, "
           "not correlations. Per-0.1-log coefficients give the FluidIQ difference associated with a 0.1-unit "
           "FP-relative log-covariance difference. Panel B compares the 746 included participants with those outside "
           "the repeated-cognition subset (continuous variables $n=102$; categorical $n=106$).")
    return table("tab:supp-cognition", cap, "llll", None, rows, resize=True)


def parse_args():
    p_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p_.add_argument("--results-dir", type=Path, required=True)
    p_.add_argument("--supplement-tables", type=Path)
    p_.add_argument("--analysis10-tables", type=Path)
    p_.add_argument("--analysis11-tables", type=Path)
    p_.add_argument("--output-dir", type=Path, help="Default: RESULTS/manuscript_outputs/tables/latex")
    a = p_.parse_args()
    r = a.results_dir.expanduser().resolve()
    a.results_dir = r
    a.output_dir = (a.output_dir or r / "manuscript_outputs" / "tables/latex").expanduser().resolve()
    a.supplement_tables = a.supplement_tables or r / "manuscript_outputs" / "tables"
    a.analysis10_tables = a.analysis10_tables or r / "10_noise_amplitude_and_normalization" / "tables"
    a.analysis11_tables = a.analysis11_tables or r / "11_normalization_followup" / "tables"
    return a


def main():
    a = parse_args()
    rd = R(a)
    nets = networks(rd)
    out = {
        "Table_S01": table_s1(rd), "Table_S02": table_s2(rd, nets), "Table_S03": table_s3(rd),
        "Table_S04": table_s4(rd, nets), "Table_S05": table_s5(rd, nets), "Table_S06": table_s6(rd, nets),
        "Table_S07": table_norm(rd, nets),
        "Table_S08": table_s7(rd), "Table_S09": table_s8(rd), "Table_S10": table_s9(rd, nets),
        "Table_S11": table_s10(rd), "Table_S12": table_s11(rd), "Table_S13": table_s12(rd),
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        (a.output_dir / f"{k}.tex").write_text(v)
        print(f"wrote {k}.tex")


if __name__ == "__main__":
    main()
