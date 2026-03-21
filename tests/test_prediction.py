"""Tests for AgingClock prediction API."""
import numpy as np
import pytest
from grnimmuneclock import AgingClock, predict_age, load_example_data


class TestAgingClockLoad:
    def test_cd4t_loads(self):
        clock = AgingClock('CD4T')
        assert clock.cell_type == 'CD4T'
        assert len(clock.feature_names) > 0

    def test_cd8t_loads(self):
        clock = AgingClock('CD8T')
        assert clock.cell_type == 'CD8T'
        assert len(clock.feature_names) > 0

    def test_feature_names_are_strings(self):
        clock = AgingClock('CD4T')
        assert all(isinstance(g, str) for g in clock.feature_names)

    def test_repr_contains_cell_type(self):
        clock = AgingClock('CD4T')
        r = repr(clock)
        assert 'CD4T' in r

    def test_feature_importance_returns_dataframe(self):
        clock = AgingClock('CD4T')
        df = clock.get_feature_importance(top_n=20)
        assert 'feature' in df.columns
        assert len(df) == 20
        # Ridge coefficients are sorted by absolute value descending
        score_col = [c for c in df.columns if c != 'feature'][0]
        assert df[score_col].abs().is_monotonic_decreasing

    def test_unsupported_cell_type_raises(self):
        with pytest.raises(Exception):
            AgingClock('NEUTROPHIL').predict(load_example_data())


class TestAgingClockPredict:
    def test_predict_adds_required_columns(self, example_adata):
        clock = AgingClock('CD4T')
        result = clock.predict(example_adata.copy())
        assert 'predicted_age' in result.obs.columns
        assert 'age_acceleration' in result.obs.columns

    def test_predicted_age_is_numeric(self, example_adata):
        clock = AgingClock('CD4T')
        result = clock.predict(example_adata.copy())
        assert np.isfinite(result.obs['predicted_age'].values).all()

    def test_age_acceleration_equals_diff(self, example_adata):
        clock = AgingClock('CD4T')
        result = clock.predict(example_adata.copy())
        expected = result.obs['predicted_age'] - result.obs['age']
        np.testing.assert_allclose(result.obs['age_acceleration'].values,
                                   expected.values, rtol=1e-5)

    def test_predict_age_functional_api(self, example_adata):
        result = predict_age(example_adata.copy(), cell_type='CD4T')
        assert 'predicted_age' in result.obs.columns

    def test_predict_handles_extra_genes(self, example_adata):
        """Extra genes beyond the model's feature set are silently ignored."""
        import anndata as ad
        import numpy as np
        import pandas as pd
        adata = example_adata.copy()
        extra = ad.AnnData(
            X=np.zeros((1, 10), dtype=np.float32),
            obs=adata.obs,
            var=pd.DataFrame(index=[f'FAKE_{i}' for i in range(10)])
        )
        adata_augmented = ad.concat([adata, extra], axis=1)
        clock = AgingClock('CD4T')
        result = clock.predict(adata_augmented)
        assert 'predicted_age' in result.obs.columns

    def test_predicted_age_in_plausible_range(self, example_adata):
        """Predicted age should be within 60 years of actual age for the example sample."""
        clock = AgingClock('CD4T')
        result = clock.predict(example_adata.copy())
        actual = result.obs['age'].values[0]
        predicted = result.obs['predicted_age'].values[0]
        assert abs(predicted - actual) < 60, \
            f"Predicted age {predicted:.1f} too far from actual {actual:.1f}"
