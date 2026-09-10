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
    version: Optional[str] = None
) -> ad.AnnData:
    """
    Predict age using a trained aging clock.
    """
    from grnimmuneclock import AgingClock

    clock = AgingClock(cell_type=cell_type, version=version)
    adata = clock.predict(adata)

    return adata
def save_function(model, feature_names, cell_type, output_dir, version):
    output_dir = Path(output_dir)
    cell_dir = output_dir / cell_type
    cell_dir.mkdir(parents=True, exist_ok=True)

    model_path = cell_dir / f"model_{version}.pkl"
    joblib.dump(model, model_path)

    features_path = cell_dir / f"feature_names_{version}.txt"
    np.savetxt(features_path, feature_names, fmt='%s')


def retrieve_function(
    cell_type: str,
    version: Optional[str] = None,
    model_dir: Optional[Path] = None,
):
    """
    Load the trained model and feature names from the package's bundled models directory.
    Training always saves here, so this is the single source of truth.
    An explicit model_dir can be passed (used in tests or custom workflows).
    """
    from grnimmuneclock.__version__ import MODEL_VERSION
    if model_dir is None:
        model_dir = Path(__file__).parent / 'models'
    else:
        model_dir = Path(model_dir)
    if version is None:
        version = MODEL_VERSION

    model_path = model_dir / cell_type / f"model_{version}.pkl"
    features_path = model_dir / cell_type / f"feature_names_{version}.txt"
    model = joblib.load(model_path)
    gene_names = np.loadtxt(features_path, dtype=str)
    return model, gene_names


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
    
    SUPPORTED_CELL_TYPES = ['CD4T', 'CD8T']

    if cell_type not in SUPPORTED_CELL_TYPES:
        raise ValueError(f"cell_type must be one of {SUPPORTED_CELL_TYPES}, got '{cell_type}'")
    
    package_dir = Path(grnimmuneclock.__file__).parent
    grn_path = package_dir / 'data' / f'consensus_grn_{cell_type}.csv'
    
    if not grn_path.exists():
        raise FileNotFoundError(
            f"Consensus GRN file not found: {grn_path}\n"
            f"Please ensure the package data is properly installed."
        )
    
    grn = pd.read_csv(grn_path)
    return grn


def load_heldout_data(cell_type: str) -> "ad.AnnData":
    """Held-out validation donors (aida + perez_sle + zhang pseudobulk, healthy only),
    restricted to the clock's own feature space.

    This is exactly the data the published permutation-importance analysis is run on, so
    `permutation_gene_importance` reproduces the published t statistics from it.

    Returns
    -------
    AnnData with obs['age'] and obs['dataset'].
    """
    import anndata as ad
    import grnimmuneclock

    path = Path(grnimmuneclock.__file__).parent / 'data' / f'heldout_{cell_type}.h5ad'
    if not path.exists():
        raise FileNotFoundError(f"Held-out data not found: {path}")
    return ad.read_h5ad(path)


def load_aging_stats(cell_type: str) -> pd.DataFrame:
    """Per-gene empirical aging direction (`pooled_rho`) from the discovery-cohort
    meta-analysis, indexed by gene. Used to sign permutation importance before ULM.
    """
    import grnimmuneclock

    path = Path(grnimmuneclock.__file__).parent / 'data' / f'aging_stats_{cell_type}.csv'
    if not path.exists():
        raise FileNotFoundError(f"Aging stats not found: {path}")
    return pd.read_csv(path).set_index('gene')
