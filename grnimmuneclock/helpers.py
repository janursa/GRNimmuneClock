"""
Helper functions for aging clock analysis.

This module provides compatibility functions and utilities for working with
aging clocks, including data formatting and prediction.
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Optional, List, Union
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import joblib


# def format_data(
#     datasets: List[str],
#     cell_type: Optional[str] = None,
#     data_type: str = 'bulk',
#     only_ctr: bool = False,
#     only_targets: bool = False
# ) -> ad.AnnData:
#     """
#     Format and merge datasets for training or evaluation.
    
#     This function maintains compatibility with the hiara.src.clock.helper.format_data
#     function while using the GRNimmuneClock package infrastructure.
    
#     Parameters
#     ----------
#     datasets : list of str
#         Dataset names to merge
#     cell_type : str, optional
#         Cell type to filter for
#     data_type : str, optional
#         Data type (default: 'bulk')
#     only_ctr : bool, optional
#         Whether to only include control samples (default: False)
#     only_targets : bool, optional
#         Whether to only include target genes from GRN (default: False)
    
#     Returns
#     -------
#     AnnData
#         Merged and formatted data
#     """
#     try:
#         from hiara.src.utils.util import retrieve_adata, retrieve_net_consensus
#         from hiara.src.common import datasets_all
#     except ImportError:
#         raise ImportError("ciim package required for format_data function")
    
#     adata_store = []
#     for d in datasets:
#         adata = retrieve_adata(dataset=d, type=data_type)
#         adata_store.append(adata)

#     # Concatenate with outer join
#     adata_train = ad.concat(
#         adata_store,
#         axis=0,
#         join='outer',
#         merge='first'
#     )

#     # Restrict to common genes
#     common_genes = set.intersection(*(set(a.var_names) for a in adata_store))
#     adata_train = adata_train[:, list(common_genes)]
    
#     if cell_type is not None:
#         adata_train = adata_train[adata_train.obs['cell_type'] == cell_type, :].copy()

#     adata_train.obs_names_make_unique()

#     if only_ctr:
#         adata_train = adata_train[adata_train.obs['is_control']]
    
#     print('Datasets in merged adata:', adata_train.obs['dataset'].unique())
#     print('Cell types in merged adata:', adata_train.obs['cell_type'].unique())

#     for col in adata_train.obs.columns:
#         if adata_train.obs[col].dtype == "object":
#             adata_train.obs[col] = adata_train.obs[col].astype(str)

#     if only_targets:
#         assert cell_type is not None, "cell_type must be specified to filter for target genes"
#         net = retrieve_net_consensus(datasets_all, cell_type)
#         target_genes = net['target'].unique()
#         adata_train = adata_train[:, adata_train.var_names.isin(target_genes)].copy()

#     return adata_train


def predict_age(
    adata: ad.AnnData,
    cell_type: str,
    use_local_clocks: bool = False,
) -> ad.AnnData:
    """
    Predict age using a trained aging clock.
    
    """
    from grnimmuneclock import AgingClock
    
    # Load aging clock
    clock = AgingClock(cell_type=cell_type, use_local_clocks=use_local_clocks)
    
    # Predict
    adata = clock.predict(adata)
    
    return adata
# Save model if output_dir provided
def save_function(model, feature_names, cell_type, output_dir, reg_type, version):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / cell_type / f"model_{reg_type}_{version}.pkl"
    (output_dir / cell_type).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    features_path = output_dir / cell_type / f"feature_names_{reg_type}_{version}.txt"
    np.savetxt(features_path, feature_names, fmt='%s')

def retrieve_function(
    cell_type: str, 
    reg_type: str = 'ridge',
    use_local_clocks: bool = False
):
    """
    Retrieve a trained model and feature names.
    
    
    """
    if use_local_clocks:
        from hiara import CLOCKS_DIR
        output_dir = CLOCKS_DIR
    else:
        from grnimmuneclock import __version__
        output_dir = Path(__file__).parent / 'data' / 'models'
        version = __version__
        raise NotImplementedError("Non-local clocks not implemented in this function yet.")
    if use_local_clocks:
        from hiara import CLOCKS_DIR, clock_version
        output_dir = CLOCKS_DIR
        version = clock_version
        if reg_type == 'NN':
            from hiara.src.clock.NN.helper import save_path_train
            import cpa
            model = cpa.CPA.load(dir_path=save_path_train)
            gene_names = model.adata.var_names
        else:
            model_path = Path(output_dir) / f"{cell_type}/" / f"model_{reg_type}_{version}.pkl"
            features_path = Path(output_dir) / f"{cell_type}/" / f"feature_names_{reg_type}_{version}.txt"
            model = joblib.load(model_path)
            gene_names = np.loadtxt(features_path, dtype=str)
        return model, gene_names

    else:
        from grnimmuneclock import AgingClock
        clock = AgingClock(cell_type=cell_type)
        return clock.model, clock.feature_names


def evaluate_groupwise_median(obs: pd.DataFrame) -> dict:
    """
    Evaluate model performance using grouped median predictions.
    
    Parameters
    ----------
    obs : DataFrame
        Observations with 'age', 'predicted_age', and 'donor_age' columns
    
    Returns
    -------
    dict
        Dictionary with 'Spearman' and 'R2' scores
    """
    df = obs.copy()
    df['age'] = df['age'].astype(float)
    df['donor_age'] = df['donor_age'].astype(str)
    
    grouping_cols = ['donor_age', 'age']
    predicted_age = df.groupby(grouping_cols)['predicted_age'].median().values
    actual_age = df.groupby(grouping_cols)['age'].median().values

    sp = spearmanr(actual_age, predicted_age)[0]
    r2 = r2_score(actual_age, predicted_age)
    
    scores = {
        'Spearman': sp,
        'R2': r2
    }
    return scores

def merge_adata(
    datasets: List[str],
    feature_type: str,
    cell_type: str,
    data_type: str,
    age_limit: int = 0
) -> ad.AnnData:
    """
    Merge multiple datasets for training.
    
    Maintains compatibility with hiara.src.clock.helper.merge_adata
    
    Parameters
    ----------
    datasets : list of str
        Dataset names
    feature_type : str
        Feature type
    cell_type : str
        Cell type
    data_type : str
        Data type
    age_limit : int, optional
        Minimum age (default: 0)
    
    Returns
    -------
    AnnData
        Merged data
    """
    try:
        from hiara.src.common import OUTPUT_DIR
        save_dir = Path(OUTPUT_DIR)
    except ImportError:
        raise ImportError("ciim package required for merge_adata function")
    
    adata_store = []
    for dataset in datasets:
        adata = ad.read_h5ad(
            save_dir / f"{feature_type}_smoothed" / f"{dataset}_{cell_type}_{data_type}.h5ad"
        )
        adata = adata[adata.obs['age'] >= age_limit].copy()
        adata = adata[adata.obs['is_control']].copy()
        adata_store.append(adata)
    
    adata_all = ad.concat(adata_store, join='inner', axis=0)
    print(adata_all.obs['dataset'].value_counts())
    return adata_all


def prepare_user_data(
    adata: ad.AnnData,
    cell_type: Optional[str] = None
) -> ad.AnnData:
    """
    Optional helper to add cell_type annotation to user data.
    
    **Note**: This function is OPTIONAL. You can directly pass your AnnData
    to AgingClock.predict() - it will handle everything automatically including:
    - Gene alignment to model feature space
    - Missing gene handling  
    - Scaling (using trained StandardScaler)
    
    This function only adds a 'cell_type' column to .obs for convenience.
    
    Parameters
    ----------
    adata : AnnData
        User's gene expression data with:
        - .X: expression matrix (samples × genes)
        - .var_names: gene symbols
        - Optional .obs['age']: actual ages for validation
    cell_type : str, optional
        Cell type to add to adata.obs['cell_type'] (e.g., 'CD4T', 'CD8T')
    
    Returns
    -------
    AnnData
        AnnData with cell_type annotation added
    
    Examples
    --------
    >>> from grnimmuneclock import prepare_user_data, AgingClock
    >>> 
    >>> # Option 1: Use prepare_user_data (optional)
    >>> adata_with_celltype = prepare_user_data(adata, cell_type='CD4T')
    >>> clock = AgingClock(cell_type='CD4T')
    >>> predictions = clock.predict(adata_with_celltype)
    >>> 
    >>> # Option 2: Direct prediction (recommended - simpler!)
    >>> clock = AgingClock(cell_type='CD4T')
    >>> predictions = clock.predict(adata)  # Works directly!
    
    Notes
    -----
    The AgingClock.predict() method automatically handles:
    - Feature alignment (your genes → model features)
    - Missing genes (filled with zeros)
    - Scaling (StandardScaler applied during prediction)
    - Works with both raw counts and normalized data
    """
    # Make a copy
    adata = adata.copy()
    
    # Add cell_type if provided
    if cell_type is not None:
        adata.obs['cell_type'] = cell_type
    
    return adata


def load_consensus_grn(cell_type: str) -> pd.DataFrame:
    """
    Load pre-computed consensus GRN for a given cell type.
    
    The consensus GRN is computed from multiple datasets and filtered for
    links that are consistent across datasets.
    
    Parameters
    ----------
    cell_type : str
        Cell type ('CD4T' or 'CD8T')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source', 'target', 'weight']
        - source: Transcription factor (TF) gene name
        - target: Target gene name
        - weight: Mean z-scored weight across datasets
    
    Examples
    --------
    >>> from grnimmuneclock import load_consensus_grn
    >>> grn = load_consensus_grn('CD4T')
    >>> print(f"Number of regulatory links: {len(grn)}")
    >>> print(f"Number of TFs: {grn['source'].nunique()}")
    >>> print(f"Number of target genes: {grn['target'].nunique()}")
    """
    import grnimmuneclock
    
    if cell_type not in ['CD4T', 'CD8T']:
        raise ValueError(f"cell_type must be 'CD4T' or 'CD8T', got '{cell_type}'")
    
    package_dir = Path(grnimmuneclock.__file__).parent
    grn_path = package_dir / 'data' / f'consensus_grn_{cell_type}.csv'
    
    if not grn_path.exists():
        raise FileNotFoundError(
            f"Consensus GRN file not found: {grn_path}\n"
            f"Please ensure the package data is properly installed."
        )
    
    grn = pd.read_csv(grn_path)
    return grn
