"""Tests for GRN loading and model/package-level utilities."""
import pytest
from grnimmuneclock import load_consensus_grn
from grnimmuneclock.helpers import retrieve_function, save_function
import grnimmuneclock


class TestLoadConsensusGRN:
    def test_cd4t_returns_dataframe(self):
        grn = load_consensus_grn('CD4T')
        import pandas as pd
        assert isinstance(grn, pd.DataFrame)

    def test_cd8t_returns_dataframe(self):
        grn = load_consensus_grn('CD8T')
        assert len(grn) > 0

    def test_required_columns_present(self):
        grn = load_consensus_grn('CD4T')
        assert {'source', 'target', 'weight'}.issubset(grn.columns)

    def test_non_empty(self):
        for ct in ['CD4T', 'CD8T']:
            grn = load_consensus_grn(ct)
            assert len(grn) > 100, f"{ct} GRN suspiciously small"

    def test_unsupported_cell_type_raises(self):
        with pytest.raises(Exception):
            load_consensus_grn('NEUTROPHIL')


class TestRetrieveFunction:
    def test_cd4t_loads_from_package(self):
        model, feature_names = retrieve_function(cell_type='CD4T')
        assert model is not None
        assert len(feature_names) > 0

    def test_cd8t_loads_from_package(self):
        model, feature_names = retrieve_function(cell_type='CD8T')
        assert model is not None
        assert len(feature_names) > 0

    def test_custom_model_dir(self, tmp_path):
        """retrieve_function respects an explicit model_dir."""
        import numpy as np
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        pipe = Pipeline([('standardscaler', StandardScaler()), ('ridge', Ridge())])
        ct_dir = tmp_path / 'CD4T'
        ct_dir.mkdir()
        joblib.dump(pipe, ct_dir / 'model_vX.pkl')
        np.savetxt(ct_dir / 'feature_names_vX.txt', ['GENE1', 'GENE2'], fmt='%s')

        model, features = retrieve_function(cell_type='CD4T', version='vX', model_dir=tmp_path)
        assert 'GENE1' in features


class TestPackageMetadata:
    def test_version_string_exists(self):
        assert hasattr(grnimmuneclock, '__version__')
        assert grnimmuneclock.__version__

    def test_bundled_models_exist(self):
        from pathlib import Path
        from grnimmuneclock import MODEL_VERSION
        pkg_dir = Path(grnimmuneclock.__file__).parent
        for ct in ['CD4T', 'CD8T']:
            model_file = pkg_dir / 'models' / ct / f'model_{MODEL_VERSION}.pkl'
            assert model_file.exists(), f"Missing bundled model: {model_file}"
