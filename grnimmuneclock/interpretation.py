"""Interpret a trained aging clock: out-of-sample permutation importance for its genes,
and TF activity from its coefficients (ULM) or from a gene set (ORA) against a regulatory
network.
"""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata, ttest_1samp, false_discovery_control, fisher_exact


def permutation_gene_importance(model, X, y, n_repeats=50, seed=0):
    """Out-of-sample permutation-importance p-value per feature, scored by Spearman rho.

    `model` must be a fitted Pipeline whose first step is a scaler and last step a linear
    model exposing `.coef_` (e.g. StandardScaler + Ridge). Such a model is linear, so a
    column permutation only perturbs one additive term of the prediction -- exploit that
    instead of an explicit per-feature refit loop.

    Parameters
    ----------
    model : sklearn Pipeline
    X, y : array-like
        Held-out data (NOT the training data).
    n_repeats : int
    seed : int

    Returns
    -------
    padj : np.ndarray, shape (n_features,)
        BH-adjusted p-value per feature (one-sided: permuting only hurts performance).
    tstat : np.ndarray, shape (n_features,)
        The t statistic behind `padj`. Negative for features whose permutation *improved*
        the score, i.e. no importance -- clip at 0 before using it as a magnitude.
    spearman_full : float
        Held-out Spearman rho of the unpermuted model.
    """
    linear = model.steps[-1][1]
    if not hasattr(linear, 'coef_'):
        raise ValueError("permutation_gene_importance needs a linear model (.coef_) as the pipeline's last step")

    X = X.toarray() if issparse(X) else np.asarray(X)
    y = np.asarray(y, dtype=float)
    Z = model.steps[0][1].transform(X)
    w = linear.coef_
    pred_full = linear.predict(Z)

    rank_y = rankdata(y)
    ry_c = rank_y - rank_y.mean()
    ry_ss = (ry_c ** 2).sum()
    rpf_c = rankdata(pred_full) - rankdata(pred_full).mean()
    spearman_full = (rpf_c * ry_c).sum() / np.sqrt((rpf_c ** 2).sum() * ry_ss)

    rng = np.random.default_rng(seed)
    importances = np.empty((Z.shape[1], n_repeats))
    for rep in range(n_repeats):
        perm_idx = rng.permutation(len(y))
        delta = (Z[perm_idx, :] - Z) * w[None, :]
        pred_perm = pred_full[:, None] + delta
        rp_c = rankdata(pred_perm, axis=0)
        rp_c -= rp_c.mean(axis=0, keepdims=True)
        corr_perm = (rp_c * ry_c[:, None]).sum(axis=0) / np.sqrt((rp_c ** 2).sum(axis=0) * ry_ss)
        importances[:, rep] = spearman_full - corr_perm

    res = ttest_1samp(importances, popmean=0, axis=1, alternative='greater')
    imp_p = np.nan_to_num(res.pvalue, nan=1.0)  # zero-variance importance -> no evidence
    padj = false_discovery_control(imp_p, method='bh')
    return padj, np.nan_to_num(res.statistic, nan=0.0), spearman_full


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
