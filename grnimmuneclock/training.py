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
from sklearn.linear_model import Ridge, ElasticNet
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


def tune_ridge_params(X, y, cv_groups=None, n_trials=30, scoring='r2', verbose=True):
    """
    Tune Ridge regression hyperparameters using Optuna.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    cv_groups : array-like, optional
        Group labels for cross-validation
    n_trials : int, optional
        Number of optimization trials (default: 30)
    scoring : str, optional
        Scoring metric (default: 'r2')
    verbose : bool, optional
        Whether to show progress (default: True)
    
    Returns
    -------
    Pipeline
        Trained pipeline with optimal alpha
    """
    import optuna
    
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1000.0, log=True)
        model = Pipeline([
            ('standardscaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        
        if cv_groups is not None:
            _, cv = get_custom_cv(cv_groups)
        else:
            raise ValueError("cv_groups must be provided")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_alpha = study.best_params['alpha']
    best_score = study.best_value

    if verbose:
        print(f"Best alpha: {best_alpha:.4f}, Best CV score: {best_score:.4f}")

    best_model = Pipeline([
        ('standardscaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha, random_state=42))
    ])

    return best_model


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
        Regression type: 'ridge' or 'elasticnet' (default: 'ridge')
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
    if reg_type == 'ridge':
        if tune_model:
            model = tune_ridge_params(
                X, y, batch_labels, 
                scoring=scoring_func, 
                verbose=verbose
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0, random_state=42)
            )
    elif reg_type == 'elasticnet':
        model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=1.0, l1_ratio=0.1, random_state=42)
        )
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}")
    
    # Fit model
    model.fit(X, y)
    y_pred = model.predict(X)
    
    return model, y_pred


def merge_training_data(
    datasets: List[str],
    cell_type: str,
    data_type: str = 'bulk',
    main_dataset: str = 'data1'
) -> ad.AnnData:
    """
    Merge multiple datasets for training.
    
    Parameters
    ----------
    datasets : list of str
        Dataset names to merge
    cell_type : str
        Cell type to train on
    feature_type : str, optional
        Feature type (default: 'gene_expression')
    data_type : str, optional
        Data type (default: 'bulk')
    main_dataset : str, optional
        Main dataset to use as reference (default: 'data1')
    
    Returns
    -------
    AnnData
        Merged training data
    """
    from hiara import retrieve_adata, retrieve_net_consensus

    
    adata_list = []
    for dataset in datasets:
        adata = retrieve_adata(dataset=dataset, cell_type=cell_type, data_type=data_type)
        net = retrieve_net_consensus(cell_type=cell_type)
        common_genes = adata.var_names.intersection(net['target'].unique())
        adata = adata[:, common_genes].copy()
        assert adata.n_vars > 0, f"No common genes between {dataset} and network for cell type {cell_type}"
        adata_list.append(adata)
    
    if len(adata_list) == 0:
        raise ValueError("No datasets found for training")
    
    # Concatenate
    adata_all = ad.concat(adata_list, join='inner', axis=0)
    
    # Order datasets with main_dataset first
    if False:
        all_datasets = adata_all.obs['dataset'].unique().tolist()
        ordered_datasets = [main_dataset] + [d for d in all_datasets if d != main_dataset]
        adata_all.obs['dataset'] = adata_all.obs['dataset'].astype(
            pd.CategoricalDtype(categories=ordered_datasets, ordered=True)
        )
    print(f"Merged data shape: {adata_all.shape}")
    print(f"Datasets: {adata_all.obs['dataset'].value_counts().to_dict()}")
    
    return adata_all


def train_aging_clock(
    cell_type: str,
    datasets: List[str],
    version: str,
    output_dir: Path,
    data_type: str = 'bulk',
    reg_type: str = 'ridge',
    tune_model: bool = True,
    scoring: str = 'spearman',
    
    verbose: bool = True
) -> Tuple[Pipeline, np.ndarray, ad.AnnData]:
    """
    Complete pipeline to train an aging clock.
    
    Parameters
    ----------
    cell_type : str
        Cell type to train on
    datasets : list of str
        Dataset names for training
    feature_type : str, optional
        Feature type (default: 'gene_expression')
    data_type : str, optional
        Data type (default: 'bulk')
    reg_type : str, optional
        Regression type (default: 'ridge')
    tune_model : bool, optional
        Whether to tune hyperparameters (default: True)
    age_limit : int, optional
        Minimum age (default: 20)
    output_dir : Path, optional
        Where to save model
    version : str, optional
        Model version (default: 'v1.0')
    verbose : bool, optional
        Print progress (default: True)
    
    Returns
    -------
    tuple
        (model, predictions, training_adata)
    """
    from grnimmuneclock import save_function
    if verbose:
        print(f"Training aging clock for {cell_type}")
        print(f"Datasets: {datasets}")
    
    # Merge data
    adata = merge_training_data(
        datasets, cell_type, data_type 
    )
    # Train model
    model, y_pred = build_model(adata, reg_type, tune_model, verbose, scoring=scoring)
    
    # Add predictions to adata
    adata.obs['predicted_age'] = y_pred
    
    save_function(model=model, feature_names=adata.var_names.values, cell_type=cell_type, output_dir=output_dir, reg_type=reg_type, version=version)
    
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
