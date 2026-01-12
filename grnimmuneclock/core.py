"""
Core module for GRNimmuneClock - Cell-type specific immune aging clocks.
"""

import os
import warnings
import json
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from scipy.sparse import issparse
import anndata as ad
from anndata import AnnData


class AgingClock:
    """
    Cell-type specific aging clock for immune cells.
    
    This class provides an interface to load pre-trained aging clock models
    and predict biological age from gene expression data.
    
    Parameters
    ----------
    cell_type : str
        Cell type for the aging clock. Options: 'CD4T', 'CD8T', 'MONO', 'B', 'NK'
    
    Attributes
    ----------
    cell_type : str
        The cell type this clock is trained for
    model : sklearn.Pipeline
        The trained model (StandardScaler + Ridge regression)
    feature_names : np.ndarray
        Names of genes used as features
    metadata : dict
        Model metadata including training info and performance metrics
    
    Examples
    --------
    >>> from grnimmuneclock import AgingClock
    >>> clock = AgingClock(cell_type='CD4T')
    >>> adata_predicted = clock.predict(adata)
    >>> print(adata_predicted.obs['predicted_age'])
    """
    
    SUPPORTED_CELL_TYPES = ['CD4T', 'CD8T', 'MONO', 'B', 'NK']
    
    def __init__(
        self,
        cell_type:  str,
        use_local_clocks: bool = False,
    ):
        if cell_type not in self.SUPPORTED_CELL_TYPES:
            raise ValueError(
                f"Unsupported cell type: {cell_type}. "
                f"Choose from {self.SUPPORTED_CELL_TYPES}"
            )
        
        self.cell_type = cell_type
        self.feature_type = 'gene_expression'
        self.data_type = 'bulk'
        self.reg_type = 'ridge'
        self.use_local_clocks = use_local_clocks
        # Load model and metadata
        self._load_model()
        # self._load_metadata()
    
    def _load_model(self):
        """Load the trained model and feature names."""
        from grnimmuneclock import retrieve_function
        self.model, self.feature_names = retrieve_function(
            cell_type=self.cell_type,
            reg_type=self.reg_type,
            use_local_clocks=self.use_local_clocks
        )
    
    # def _load_metadata(self):
    #     """Load model metadata if available."""
    #     metadata_path = self.model_dir / 'metadata.json'
        
    #     if metadata_path.exists():
    #         with open(metadata_path, 'r') as f:
    #             self.metadata = json.load(f)
    #     else:
    #         # Create basic metadata
    #         self.metadata = {
    #             'cell_type': self.cell_type,
    #             'version': self.version,
    #             'feature_type': self.feature_type,
    #             'data_type': self.data_type,
    #             'reg_type': self.reg_type,
    #             'n_features': len(self.feature_names)
    #         }
    
    def _align_feature_space(self, adata: AnnData) -> AnnData:
        """
        Align input data features to match the model's feature space.
        
        Missing features will be filled with zeros.
        
        Parameters
        ----------
        adata : AnnData
            Input data with gene expression
        
        Returns
        -------
        AnnData
            Data with aligned features
        """
        var_names = np.array(adata.var.index.tolist())
        var_index = {gene: i for i, gene in enumerate(var_names)}
        
        # Collect indices or mark as -1 for missing
        idxs = np.array([var_index.get(gene, -1) for gene in self.feature_names])
        
        # Create a matrix with correct shape
        rows = adata.obs.shape[0]
        cols = len(self.feature_names)
        X_aligned = sparse.lil_matrix((rows, cols))
        
        # Fill in available gene columns
        present = idxs != -1
        if present.sum() > 0:
            X_aligned[:, present] = adata[:].X[:, idxs[present]]
        
        # Convert to CSR for efficiency
        X_aligned = X_aligned.tocsr()
        
        # Create new AnnData object
        new_adata = AnnData(
            X=X_aligned,
            obs=adata.obs.copy(),
            var={"gene_symbols": self.feature_names},
        )
        new_adata.var_names = self.feature_names
        
        # Warn about missing features
        n_missing = (~present).sum()
        if n_missing > 0:
            coverage = present.sum() / len(self.feature_names)
            warnings.warn(
                f"{n_missing} features ({(1-coverage)*100:.1f}%) missing from input data. "
                f"They will be set to zero."
            )
        
        return new_adata
    
    def _validate_input(self, adata: AnnData):
        """Validate input data format."""
        if not isinstance(adata, AnnData):
            raise TypeError("Input must be an AnnData object")
        
        if adata.X is None:
            raise ValueError("Input AnnData has no expression matrix (X)")
        
        if adata.n_obs == 0:
            raise ValueError("Input AnnData has no observations (cells/samples)")
        
        if adata.n_vars == 0:
            raise ValueError("Input AnnData has no variables (genes)")
    
    def predict(self, adata: AnnData, return_adata: bool = True) -> Union[AnnData, np.ndarray]:
        """
        Predict biological age from gene expression data.
        
        Parameters
        ----------
        adata : AnnData
            Input data with gene expression in .X
            Rows are samples/cells, columns are genes
        return_adata : bool, optional
            If True, return AnnData with 'predicted_age' in .obs
            If False, return numpy array of predictions (default: True)
        
        Returns
        -------
        AnnData or np.ndarray
            If return_adata=True: Input AnnData with 'predicted_age' column added to .obs
            If return_adata=False: Array of predicted ages
        
        Examples
        --------
        >>> clock = AgingClock(cell_type='CD4T')
        >>> adata_with_predictions = clock.predict(adata)
        >>> ages = clock.predict(adata, return_adata=False)
        """
        # Validate input
        self._validate_input(adata)
        
        # Align features to model's feature space
        adata_aligned = self._align_feature_space(adata)
        
        # Get expression matrix
        X = adata_aligned.X.copy()
        if issparse(X):
            X = X.toarray()
        
        # Predict
        predicted_age = self.model.predict(X)
        
        if return_adata:
            # Add predictions to original adata
            adata.obs['predicted_age'] = predicted_age.copy()
            
            # Calculate age acceleration if actual age is available
            if 'age' in adata.obs.columns:
                adata.obs['age_acceleration'] = adata.obs['predicted_age'] - adata.obs['age']
            
            return adata
        else:
            return predicted_age
    
    def predict_batch(
        self,
        adata_list: List[AnnData],
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[np.ndarray]]:
        """
        Predict ages for multiple AnnData objects.
        
        Parameters
        ----------
        adata_list : list of AnnData
            List of AnnData objects to predict on
        return_dataframe : bool, optional
            If True, return consolidated DataFrame (default: True)
            If False, return list of prediction arrays
        
        Returns
        -------
        pd.DataFrame or list of np.ndarray
            Predictions for all inputs
        """
        predictions = []
        
        for i, adata in enumerate(adata_list):
            pred = self.predict(adata, return_adata=False)
            predictions.append(pred)
        
        if return_dataframe:
            # Combine into DataFrame
            df_list = []
            for i, (adata, pred) in enumerate(zip(adata_list, predictions)):
                df = pd.DataFrame({
                    'predicted_age': pred,
                    'batch': i
                })
                # Add other obs columns if available
                for col in adata.obs.columns:
                    df[col] = adata.obs[col].values
                df_list.append(df)
            
            return pd.concat(df_list, ignore_index=True)
        else:
            return predictions
    
    def __repr__(self):
        return (
            f"AgingClock(cell_type='{self.cell_type}', "
            f"version='{self.version}', "
            f"n_features={len(self.feature_names)})"
        )
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importances (model coefficients).
        
        Parameters
        ----------
        top_n : int, optional
            Return only top N features by absolute importance
            If None, return all features
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'feature', 'coefficient'
            Sorted by absolute coefficient value (descending)
        """
        # Get coefficients from Ridge model
        coefs = self.model.named_steps['ridge'].coef_
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefs
        })
        
        # Sort by absolute value
        df['abs_coefficient'] = np.abs(df['coefficient'])
        df = df.sort_values('abs_coefficient', ascending=False)
        df = df.drop('abs_coefficient', axis=1)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df.reset_index(drop=True)


def load_example_data() -> AnnData:
    """
    Load example data for testing the aging clock.
    
    Returns
    -------
    AnnData
        Small example dataset from a single donor
    """
    package_dir = Path(__file__).parent
    data_path = package_dir / 'data' / 'example_data.h5ad'
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Example data not found at {data_path}. "
            "Please ensure the package is properly installed."
        )
    
    return ad.read_h5ad(data_path)
