# Changelog

## [1.2.0]

### Added
- `interpretation` module: `permutation_gene_importance`, `tf_activity_from_coefs`, `regulon_ora`.
- `load_heldout_data(cell_type)` — held-out validation donors, the data the published permutation-importance analysis runs on.
- `load_aging_stats(cell_type)` — per-gene empirical aging direction (`pooled_rho`) from the discovery-cohort meta-analysis.

### Changed
- Bundled consensus GRNs and CD4T/CD8T models retrained on skeleton-pruned GRNs.
- `requires-python` raised to `>=3.9`; `scipy>=1.11.0` (for `false_discovery_control`); added `decoupler>=2.0.0`.

---

## [1.1.0]

### Changed
- **Training API**: `train_aging_clock()` now receives a prepared `AnnData` object directly instead of dataset names. Data preparation (retrieve, filter to GRN genes, merge) is the caller's responsibility. This removes all `hiara` dependencies from the clock package.
- **Single model location**: Trained models are always saved to and loaded from the bundled package directory (`grnimmuneclock/models/<cell_type>/`). The `use_local_clocks` flag and `CLOCKS_DIR` concept are removed.
- **`MODEL_VERSION`**: Introduced `MODEL_VERSION = "V1"` as a separate constant for model artifact versioning. `__version__` now correctly reflects the pip package version (`1.1.0`).
- **`retrieve_function`**: Accepts optional `model_dir` argument for custom workflows and tests.
- **`_align_feature_space`**: Replaced sparse `lil_matrix` path with a plain numpy implementation (faster for typical sample sizes).
- **`AgingClock.__repr__`**: Now shows the resolved model version and number of features.
- **`AgingClock.SUPPORTED_CELL_TYPES`**: Restricted to `['CD4T', 'CD8T']` for v1.1.0; MONO/B/NK support planned for a future release.

### Removed
- `merge_training_data()` — data preparation moved to caller (hiara).
- `merge_adata()` — no longer needed.
- `use_local_clocks` parameter from `predict_age()`, `AgingClock.__init__()`, and `retrieve_function()`.
- Internal helpers (`build_model`, `tune_ridge_params`, `save_function`, `retrieve_function`) removed from public `__all__`.

### Fixed
- Email address in package metadata.
- Stale B/MONO/NK model artifacts removed from the package.

### Added
- Test suite (`tests/`) with 29 tests covering prediction, training, GRN loading, and package metadata.

---

## [1.0.0]

- Initial release with CD4T and CD8T aging clocks.
- Ridge regression pipeline trained on onek1k and abf300 cohorts.
- Bundled consensus GRNs for CD4T and CD8T.
