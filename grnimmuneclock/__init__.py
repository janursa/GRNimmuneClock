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
from .__version__ import __version__, MODEL_VERSION, __author__, __license__
from . import plotting
from .training import (
    train_aging_clock,
    build_model,
    evaluate_cv_performance,
    tune_ridge_params
)
from .helpers import (
    predict_age,
    retrieve_function,
    save_function,
    evaluate_groupwise_median,
    load_consensus_grn,
    load_aging_stats,
    prepare_user_data
)
from .interpretation import (
    tf_activity_from_coefs,
    regulon_ora,
)

__all__ = [
    # Core public API
    'AgingClock',
    'load_example_data',
    'predict_age',
    'load_consensus_grn',
    'load_aging_stats',
    'prepare_user_data',
    # Training
    'train_aging_clock',
    'evaluate_cv_performance',
    # Interpretation
    'tf_activity_from_coefs',
    'regulon_ora',
    # Plotting
    'plotting',
    # Evaluation
    'evaluate_groupwise_median',
    # Package metadata
    '__version__',
    'MODEL_VERSION',
    '__author__',
    '__license__',
]
