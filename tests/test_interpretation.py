"""Sanity checks for grnimmuneclock.interpretation."""
import numpy as np
import pandas as pd

from grnimmuneclock import tf_activity_from_coefs, regulon_ora


def test_tf_activity_from_coefs_recovers_driver_tf():
    rng = np.random.default_rng(1)
    genes = [f'g{i}' for i in range(10)]
    coefs = 1.0 + rng.normal(scale=0.05, size=10)
    coefs[6:] = rng.normal(scale=0.05, size=4)  # TF2's 4 targets carry no clock weight
    net = pd.DataFrame({
        'source': ['TF1'] * 6 + ['TF2'] * 4,
        'target': genes,
        'weight': 1.0,
    })
    tf_df = tf_activity_from_coefs(genes, coefs, net, min_targets=3)
    top = tf_df.set_index('tf')['score'].abs().idxmax()
    assert top == 'TF1'


def test_regulon_ora_recovers_enriched_tf():
    background = [f'g{i}' for i in range(200)]
    foreground = background[:20]
    net = pd.DataFrame({
        'source': ['TF_hit'] * 10 + ['TF_miss'] * 10,
        'target': background[:10] + background[100:110],  # TF_hit sits inside the foreground
    })
    res = regulon_ora(foreground, background, net, min_targets=5).set_index('tf')
    assert res.loc['TF_hit', 'n_hit'] == 10 and res.loc['TF_miss', 'n_hit'] == 0
    assert res.loc['TF_hit', 'padj'] < 0.05 < res.loc['TF_miss', 'padj']
    # regulons smaller than min_targets, and TFs with no target in the background, drop out
    assert regulon_ora(foreground, background, net, min_targets=11).empty


if __name__ == '__main__':
    test_tf_activity_from_coefs_recovers_driver_tf()
    test_regulon_ora_recovers_enriched_tf()
    print('ok')
