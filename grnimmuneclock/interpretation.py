"""Interpret a trained aging clock: TF activity from its coefficients (ULM) or from a gene
set (ORA) against a regulatory network.
"""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact


def tf_activity_from_coefs(genes, coefs, net, min_targets=5):
    """ULM-based TF activity from a gene coefficient vector (e.g. clock ridge weights),
    treated as a one-sample profile, against a regulatory network.

    Parameters
    ----------
    genes : array-like of str
    coefs : array-like of float
    net : DataFrame with 'source' (TF), 'target' (gene) columns (+ optional 'weight')
    min_targets : int
        Minimum regulon size for a TF to be tested (decoupler `tmin`).

    Returns
    -------
    DataFrame with columns 'tf', 'score', 'padj', one row per testable TF.
    """
    try:
        import decoupler as dc
    except ImportError as e:
        raise ImportError("tf_activity_from_coefs requires decoupler: pip install decoupler") from e
    import anndata as ad

    adata = ad.AnnData(np.asarray([coefs], dtype=float),
                       obs=pd.DataFrame({'sample': [1]}),
                       var=pd.DataFrame(index=pd.Index(genes, name='genes')))
    dc.mt.ulm(adata, net, tmin=min_targets)
    score = adata.obsm['score_ulm'].iloc[0]
    padj = adata.obsm['padj_ulm'].iloc[0]
    return pd.DataFrame({'tf': score.index, 'score': score.values, 'padj': padj.values}).reset_index(drop=True)


def regulon_ora(foreground, background, net, min_targets=5):
    """Fisher overrepresentation of each TF regulon among `foreground`, against `background`.

    Coefficient-free: only set membership enters the test, so the magnitude and sign
    distortion that collinearity induces in a linear model's own weights cannot propagate
    here. Give it a directional foreground (e.g. clock genes that rise with age) or the
    enrichment is diluted by the opposite direction.

    Parameters
    ----------
    foreground : array-like of str
        Genes of interest; intersected with `background` before testing.
    background : array-like of str
        The universe the foreground was drawn from (e.g. all clock features). Regulons are
        restricted to it, so it must not be wider than the set foreground selection ran on.
    net : DataFrame with 'source' (TF), 'target' (gene) columns.
    min_targets : int
        Minimum regulon size, after restriction to `background`, for a TF to be tested.

    Returns
    -------
    DataFrame with 'tf', 'n_targets', 'n_hit', 'odds', 'pval', 'padj', sorted by 'pval'.
    """
    bg = set(background)
    fg = set(foreground) & bg
    rows = []
    for tf, sub in net.groupby('source'):
        targets = set(sub['target']) & bg
        if len(targets) < min_targets:
            continue
        n_hit = len(targets & fg)
        odds, pval = fisher_exact([[n_hit, len(targets) - n_hit],
                                   [len(fg) - n_hit, len(bg - fg - targets)]],
                                  alternative='greater')
        rows.append({'tf': tf, 'n_targets': len(targets), 'n_hit': n_hit, 'odds': odds, 'pval': pval})

    out = pd.DataFrame(rows, columns=['tf', 'n_targets', 'n_hit', 'odds', 'pval'])
    out['padj'] = false_discovery_control(out['pval'].values, method='bh') if len(out) else []
    return out.sort_values('pval', ignore_index=True)
