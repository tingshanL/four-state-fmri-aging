# Four-state fMRI aging

Analysis and figure code for longitudinal covariance change across rest, CARIT, FaceName, and visual–motor fMRI in AABC Release 2.

## Requirements

Python 3.9 or later and authorized local access to the study inputs:

```bash
python -m pip install -r requirements.txt
```

The scripts start from the state-specific metric and node-connectivity caches, parcel-to-network manifest, released covariance and signal/noise amplitude archives, structural and non-imaging files, and longitudinal subject-event table. These inputs and all generated outputs remain outside the repository. Source paths and hashes are recorded in the analysis outputs. Reading legacy `.xls` inputs additionally requires `xlrd`.

## Analyses

Set the local paths, then run from the repository root:

```bash
DATA_DIR="/path/to/multimodal"
SUBJECT_TABLE="/path/to/longitudinal_subject_event_table.csv"
RESULTS_DIR="/path/to/four_state_fmri_results"

python analysis/01_qc_and_cohort.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/02_primary_profile_and_measurement.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/03_fp_specificity_and_robustness.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/04_structural_sensitivity.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/05_cognition.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/06_final_checks.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/07_site_profile_shape_diagnostic.py --data-dir "$DATA_DIR" --subject-table "$SUBJECT_TABLE" --results-dir "$RESULTS_DIR"
python analysis/09_released_measurement_audit.py --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR"
python analysis/10_noise_amplitude_and_normalization.py --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR"
python analysis/11_normalization_followup.py --data-dir "$DATA_DIR" --results-dir "$RESULTS_DIR"
```

Scripts 01–07 cover the cohort, primary models, FP specificity, structural sensitivity, cognition, age and convergence checks, and site profiles. Scripts 09–11 audit released measurements and compare signal and candidate total normalization. Optional documented measurement and correlation definitions are supplied through their command-line arguments. Use the default draw counts for manuscript results; `--fast` is for pipeline checks.

## Manuscript outputs

With completed analysis outputs, run these four builders in order. They read existing results without fitting models.

```bash
python analysis/08_make_manuscript_outputs.py --results-dir "$RESULTS_DIR"
python analysis/12_make_main_figures.py --results-dir "$RESULTS_DIR"
python analysis/13_make_supplement_figures.py --results-dir "$RESULTS_DIR"
python analysis/14_make_supplement_tables.py --results-dir "$RESULTS_DIR"
```

| Builder | Outputs under `RESULTS_DIR/manuscript_outputs/` |
| --- | --- |
| 08 | Figure 1 in `figures/`; Table 1 and source CSVs in `tables/`; assembly records in `audit/` |
| 12 | Figures 2–4 in `figures/` |
| 13 | Figures S1–S9 in `figures/` |
| 14 | Tables S1–S13 in `tables/latex/` |

`figure_style.py` supplies shared plotting styles. Source CSV filenames retain their analysis identifiers; the final supplement tables are numbered by script 14. Each builder accepts `--output-dir`; builders 12–14 also accept explicit source-table directories.
