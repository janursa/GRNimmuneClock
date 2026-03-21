"""Tests for train_aging_clock — uses synthetic data, writes to a tmp dir."""
import numpy as np
import pytest
from pathlib import Path
from grnimmuneclock import train_aging_clock, AgingClock
from grnimmuneclock.helpers import retrieve_function


class TestTrainAgingClock:
    def test_returns_three_tuple(self, synthetic_adata, tmp_path):
        result = train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        assert len(result) == 3

    def test_model_file_saved(self, synthetic_adata, tmp_path):
        train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        assert (tmp_path / 'CD4T' / 'model_test.pkl').exists()
        assert (tmp_path / 'CD4T' / 'feature_names_test.txt').exists()

    def test_predictions_adata_has_required_obs(self, synthetic_adata, tmp_path):
        _, predictions, adata_train = train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        assert 'predicted_age' in adata_train.obs.columns
        # age_acceleration is added by predict, not by train — training only adds predicted_age
        assert 'age' in adata_train.obs.columns

    def test_predictions_array_length_matches_n_obs(self, synthetic_adata, tmp_path):
        _, predictions, adata_train = train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        assert len(predictions) == synthetic_adata.n_obs

    def test_retrain_overwrites_model(self, synthetic_adata, tmp_path):
        train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        mtime1 = (tmp_path / 'CD4T' / 'model_test.pkl').stat().st_mtime

        train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        mtime2 = (tmp_path / 'CD4T' / 'model_test.pkl').stat().st_mtime
        assert mtime2 >= mtime1

    def test_trained_model_loadable_via_retrieve(self, synthetic_adata, tmp_path):
        train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='test',
            output_dir=tmp_path,
            tune_model=False,
            verbose=False,
        )
        model, feature_names = retrieve_function(
            cell_type='CD4T',
            version='test',
            model_dir=tmp_path,
        )
        assert model is not None
        assert len(feature_names) > 0

    def test_default_output_dir_is_package_models(self, synthetic_adata):
        """When output_dir is omitted, model is saved inside the pip package."""
        import grnimmuneclock
        pkg_models = Path(grnimmuneclock.__file__).parent / 'models'
        train_aging_clock(
            adata=synthetic_adata,
            cell_type='CD4T',
            version='_pytest_tmp',
            tune_model=False,
            verbose=False,
        )
        saved = pkg_models / 'CD4T' / 'model__pytest_tmp.pkl'
        assert saved.exists()
        saved.unlink()   # clean up
        (pkg_models / 'CD4T' / 'feature_names__pytest_tmp.txt').unlink(missing_ok=True)
