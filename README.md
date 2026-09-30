# Four-state fMRI longitudinal covariance analysis

Analysis code for the NeuroImage manuscript on longitudinal covariance change across rest, CARIT, FaceName, and visual–motor fMRI states.

## Contents

`analysis/01` through `analysis/07` reproduce the cohort summaries, primary models, sensitivity analyses, cognition analysis, final checks, and site-profile diagnostic. `analysis/08_make_manuscript_outputs.py` assembles the manuscript figures and source tables without fitting new models.

## Requirements

- Python 3.9 or later
- Packages listed in `requirements.txt`
- Authorized local access to the AABC Release 2/HCP-Aging inputs used in the study

The participant-level source data are not distributed with this repository.

## Inputs

`DATA_DIR` must contain the derived inputs used by the analysis:

- four state-specific prespecified-metric caches;
- full- and partial-covariance node-connectivity caches;
- `fmri_node_parcel_manifest.csv`;
- four parcel-amplitude archives and the matching full-covariance archives;
- the released structural and non-imaging files used by analyses 04 and 05.

`SUBJECT_TABLE` is the longitudinal subject-event table used to define visits, timing, site, scanner, and covariates. The repository begins from these derived caches; their source files and hashes are recorded in the analysis outputs.

## Run order

Set paths to the local project data, longitudinal subject-event table, and results directory, then run:

```bash
python analysis/01_qc_and_cohort.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/02_primary_profile_and_measurement.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/03_fp_specificity_and_robustness.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/04_structural_sensitivity.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/05_cognition.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/06_final_checks.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/07_site_profile_shape_diagnostic.py --data-dir DATA_DIR --subject-table SUBJECT_TABLE --results-dir RESULTS_DIR
python analysis/08_make_manuscript_outputs.py --results-dir RESULTS_DIR
```

The analysis scripts use fixed seeds and write tables, figures, model checkpoints, and run manifests under `RESULTS_DIR`. The manuscript files are written directly to `RESULTS_DIR/manuscript_outputs/figures` and `RESULTS_DIR/manuscript_outputs/tables`.

Use `--fast` only for pipeline checks. Manuscript results use the default draw counts.
