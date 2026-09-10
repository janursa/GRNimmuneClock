"""
Training module for GRN-based aging clocks.

This module provides functions to train cell-type specific aging clocks
using gene expression data and various regression models.
"""

import numpy as np
import pandas as pd
import anndata as ad
import joblib
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, make_scorer

warnings.filterwarnings('ignore')


def spearman_scorer(y_true, y_pred):
    """Spearman correlation scorer for model evaluation."""
    return spearmanr(y_true, y_pred).correlation


def get_custom_cv(groups, main_code=0):
    """
    Leave-one-group-out CV, where each fold leaves out one non-main group for testing,
    and always includes the main_code group in training.
    
    Parameters
    ----------
    groups : array-like
        Group labels for each sample
    main_code : int, optional
        Code for the main group to always include in training (default: 0)
    
    Returns
    -------
    tuple
        (ordered_test_groups, custom_splits)
    """
    unique_groups = np.unique(groups)
    ordered_test_groups = [g for g in unique_groups if g != main_code]

    custom_splits = []
    for test_group in ordered_test_groups:
        test_mask = (groups == test_group)
        train_mask = (groups != test_group)
        custom_splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    
    return ordered_test_groups, custom_splits


class _DefaultTrial:
    """Stand-in for an optuna trial that returns library defaults, for tune_model=False."""
    _D = {'alpha': 1.0, 'l1_ratio': 0.1, 'max_iter': 100, 'learning_rate': 0.1,
          'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'l2_regularization': 0.0,
          'width': 64, 'depth': 2, 'nn_alpha': 1e-4, 'lr_init': 1e-3}

    def suggest_float(self, name, *a, **k):
        return self._D[name]
    suggest_int = suggest_categorical = suggest_float


def _suggest_model(trial, reg_type):
    """Search space per model family. Same optuna budget and same CV for all of them,
    so a model that loses, loses on form and not on tuning effort."""
    if reg_type == 'ridge':
        return Pipeline([('standardscaler', StandardScaler()),
                         ('ridge', Ridge(alpha=trial.suggest_float("alpha", 0.01, 1000.0, log=True),
                                         random_state=42))])
    if reg_type == 'lasso':
        return Pipeline([('standardscaler', StandardScaler()),
                         ('lasso', Lasso(alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                                         max_iter=10000, random_state=42))])
    if reg_type == 'elasticnet':
        return Pipeline([('standardscaler', StandardScaler()),
                         ('elasticnet', ElasticNet(
                             alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                             l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
                             max_iter=10000, random_state=42))])
    if reg_type == 'gradientboosting':
        # ponytail: HistGradientBoosting, not GradientBoostingRegressor -- same algorithm
        # family, binned splits, ~100x faster at p=5k so it can afford the same 30 trials.
        from sklearn.ensemble import HistGradientBoostingRegressor
        return Pipeline([('standardscaler', StandardScaler()),
                         ('gb', HistGradientBoostingRegressor(
                             max_iter=trial.suggest_int("max_iter", 100, 600),
                             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                             max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 63),
                             min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
                             l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                             early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=20, random_state=42))])
    if reg_type == 'nn':
        from sklearn.neural_network import MLPRegressor
        width = trial.suggest_categorical("width", [32, 64, 128, 256])
        depth = trial.suggest_int("depth", 1, 3)
        return Pipeline([('standardscaler', StandardScaler()),
                         ('nn', MLPRegressor(
                             hidden_layer_sizes=tuple(max(8, width // 2 ** i) for i in range(depth)),
                             alpha=trial.suggest_float("nn_alpha", 1e-5, 10.0, log=True),
                             learning_rate_init=trial.suggest_float("lr_init", 1e-4, 1e-2, log=True),
                             max_iter=1000, early_stopping=True, n_iter_no_change=20,
                             random_state=42))])
    raise ValueError(f"Unknown reg_type: {reg_type}")


def tune_params(X, y, cv_groups=None, reg_type='ridge', n_trials=30, scoring='r2', verbose=True):
    """
    Tune hyperparameters with Optuna, on the leave-one-cohort-out CV defined by cv_groups.

    Parameters
    ----------
    X, y : array-like
        Feature matrix and target ages.
    cv_groups : array-like
        Cohort labels; required (defines the CV folds).
    reg_type : str
        'ridge', 'lasso', 'elasticnet', 'gradientboosting' or 'nn'.
    n_trials : int
        Optimization trials (default: 30). Identical across reg_types by design.
    scoring : str or callable
        sklearn scoring (default: 'r2').

    Returns
    -------
    Pipeline
        Unfitted pipeline with the best hyperparameters.
    """
    import optuna

    if cv_groups is None:
        raise ValueError("cv_groups must be provided")
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    _, cv = get_custom_cv(cv_groups)

    def objective(trial):
        scores = cross_val_score(_suggest_model(trial, reg_type), X, y, cv=cv, scoring=scoring)
        # A degenerate fit (constant predictions) makes spearman nan; optuna rejects nan
        # and would abort the study if every trial did it. Score it as worst instead.
        return np.nan_to_num(np.mean(scores), nan=-1e9)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print(f"[{reg_type}] best CV score: {study.best_value:.4f}, params: {study.best_params}")

    return _suggest_model(optuna.trial.FixedTrial(study.best_params), reg_type)


def tune_ridge_params(X, y, cv_groups=None, n_trials=30, scoring='r2', verbose=True):
    """Backward-compatible alias for `tune_params(..., reg_type='ridge')`."""
    return tune_params(X, y, cv_groups, reg_type='ridge', n_trials=n_trials,
                       scoring=scoring, verbose=verbose)


def build_model(
    adata: ad.AnnData,
    reg_type: str = 'ridge',
    tune_model: bool = True,
    verbose: bool = True,
    scoring: str = 'spearman'
) -> Tuple[Pipeline, np.ndarray]:
    """
    Build and train an aging clock model.
    
    Parameters
    ----------
    adata : AnnData
        Training data with gene expression in .X and age in .obs['age']
        Must have 'dataset' column in .obs for cross-validation
    reg_type : str, optional
        'ridge', 'lasso', 'elasticnet', 'gradientboosting' or 'nn'
        (default: 'ridge'). All of them honour tune_model.
    tune_model : bool, optional
        Whether to tune hyperparameters (default: True)
    verbose : bool, optional
        Whether to print progress (default: True)
    
    Returns
    -------
    tuple
        (trained_model, predictions_on_training_data)
    """
    # Prepare data
    X = adata.X
    y = adata.obs['age'].values
    
    if issparse(X):
        X = X.toarray()
    
    # Get batch labels for CV
    if 'dataset' not in adata.obs.columns:
        raise ValueError("AnnData must have 'dataset' column in .obs for cross-validation")
    adata.obs['dataset'] = adata.obs['dataset'].astype('category')
    batch_labels = adata.obs['dataset'].cat.codes.values
    
    # Build model
    if scoring == 'spearman':
        scoring_func = make_scorer(spearman_scorer)
    elif scoring == 'r2':
        scoring_func = make_scorer(r2_score)
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")
    if tune_model:
        model = tune_params(X, y, batch_labels, reg_type=reg_type,
                            scoring=scoring_func, verbose=verbose)
    else:
        # ponytail: untuned path just takes each library's own defaults.
        model = _suggest_model(_DefaultTrial(), reg_type)

    # Fit model
    model.fit(X, y)
    y_pred = model.predict(X)
    
    return model, y_pred


def train_aging_clock(
    adata: ad.AnnData,
    cell_type: str,
    version: str,
    output_dir: Optional[Path] = None,
    reg_type: str = 'ridge',
    tune_model: bool = True,
    scoring: str = 'spearman',
    verbose: bool = True
) -> Tuple[Pipeline, np.ndarray, ad.AnnData]:
    """
    Complete pipeline to train an aging clock.

    Parameters
    ----------
    adata : AnnData
        Training data with gene expression in .X, age in .obs['age'],
        and dataset label in .obs['dataset'] (used for cross-validation).
    cell_type : str
        Cell type to train on (used for saving the model).
    version : str
        Model version string (e.g. 'V1'). Used in the saved filename.
    output_dir : Path, optional
        Directory to save the trained model. Defaults to the package's
        bundled models directory (grnimmuneclock/models/), so re-training
        overrides the published model in-place.
    reg_type : str, optional
        Regression type (default: 'ridge').
    tune_model : bool, optional
        Whether to tune hyperparameters with Optuna (default: True).
    scoring : str, optional
        CV scoring metric: 'spearman' or 'r2' (default: 'spearman').
    verbose : bool, optional
        Print progress (default: True).

    Returns
    -------
    tuple
        (model, predictions, training_adata)
    """
    from grnimmuneclock import save_function

    if output_dir is None:
        output_dir = Path(__file__).parent / 'models'

    if verbose:
        print(f"Training aging clock for {cell_type}")
        print(f"Data shape: {adata.shape}")
        print(f"Datasets: {adata.obs['dataset'].value_counts().to_dict()}")

    model, y_pred = build_model(adata, reg_type, tune_model, verbose, scoring=scoring)

    adata.obs['predicted_age'] = y_pred

    save_function(
        model=model,
        feature_names=adata.var_names.values,
        cell_type=cell_type,
        output_dir=output_dir,
        version=version
    )

    return model, y_pred, adata


def evaluate_cv_performance(
    adata: ad.AnnData,
    model: Pipeline,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate model performance using cross-validation.
    
    Parameters
    ----------
    adata : AnnData
        Training data
    model : Pipeline
        Trained model
    verbose : bool, optional
        Print results (default: True)
    
    Returns
    -------
    DataFrame
        Cross-validation scores per dataset
    """
    X = adata.X
    y = adata.obs['age'].values
    
    if issparse(X):
        X = X.toarray()
    
    batch_labels = adata.obs['dataset'].cat.codes.values
    dataset_code_map = dict(zip(
        range(len(adata.obs['dataset'].cat.categories)),
        adata.obs['dataset'].cat.categories
    ))
    
    ordered_test_groups, cv = get_custom_cv(batch_labels)
    fold_scores = {}
    
    for i, code in enumerate(ordered_test_groups):
        train_idx, test_idx = cv[i]
        
        model.fit(X[train_idx], y[train_idx])
        y_true = y[test_idx]
        y_pred = model.predict(X[test_idx])
        
        spearman_corr = spearmanr(y_true, y_pred).correlation
        r2 = r2_score(y_true, y_pred)
        
        dataset_name = dataset_code_map[code]
        fold_scores[dataset_name] = {
            'spearman': round(spearman_corr, 3),
            'r2': round(r2, 3)
        }
        
        if verbose:
            print(f"{dataset_name}: R²={r2:.3f}, Spearman={spearman_corr:.3f}")
    
    scores_df = pd.DataFrame.from_dict(fold_scores, orient='index')
    return scores_df
