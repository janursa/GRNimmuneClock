"""
GRNimmuneClock: Cell-Type Specific Immune Aging Clocks

This package provides pre-trained aging clocks for immune cell types,
built using gene regulatory network (GRN) analysis.

Example
-------
>>> from grnimmuneclock import AgingClock, load_example_data
>>> 
>>> # Load pre-trained clock
>>> clock = AgingClock(cell_type='CD4T')
>>> 
>>> # Load example data
>>> adata = load_example_data()
>>> 
>>> # Predict ages
>>> adata_predicted = clock.predict(adata)
>>> print(adata_predicted.obs['predicted_age'])
"""

from .core import AgingClock, load_example_data
from .__version__ import __version__, __author__, __license__
from . import plotting
from .training import (
    train_aging_clock,
    build_model,
    merge_training_data,
    evaluate_cv_performance,
    tune_ridge_params
)
from .helpers import (
    predict_age,
    retrieve_function,
    save_function,
    merge_adata,
    evaluate_groupwise_median,
    load_consensus_grn,
    prepare_user_data
)

__all__ = [
    'AgingClock',
    'load_example_data',
    'plotting',
    'train_aging_clock',
    'build_model',
    'merge_training_data',
    'evaluate_cv_performance',
    'tune_ridge_params',
    'predict_age',
    'retrieve_function',
    'save_function',
    'merge_adata',
    'evaluate_groupwise_median',
    'load_consensus_grn',
    'prepare_user_data',
    '__version__',
    '__author__',
    '__license__'
]
