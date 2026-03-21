"""Shared fixtures for GRNimmuneClock tests."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest
from pathlib import Path
from grnimmuneclock import load_consensus_grn


@pytest.fixture(scope="session")
def example_adata():
    from grnimmuneclock import load_example_data
    return load_example_data()


@pytest.fixture(scope="session")
def synthetic_adata():
    """
    Synthetic AnnData for training tests: 40 samples, genes drawn from the
    CD4T consensus GRN, random log-normalised expression, ages 20-80.
    """
    rng = np.random.default_rng(42)
    grn = load_consensus_grn('CD4T')
    genes = grn['target'].unique()[:500]   # 500 GRN genes is plenty
    n_samples = 40

    X = rng.lognormal(mean=1.0, sigma=0.8, size=(n_samples, len(genes))).astype(np.float32)
    obs = pd.DataFrame({
        'age':      rng.uniform(20, 80, n_samples).astype(float),
        'donor_id': [f'donor_{i}' for i in range(n_samples)],
        'sex':      rng.choice(['Male', 'Female'], n_samples),
        'dataset':  'synthetic',
        'cell_type': 'CD4T',
    })
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)
