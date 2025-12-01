"""
Plotting functions for GRNimmuneClock.

This module provides visualization tools for aging clock predictions,
including scatter plots, disease acceleration analysis, and perturbation effects.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, spearmanr
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from pandas.api.types import CategoricalDtype


# Default palettes
PALETTE_DISEASE = {
    'Healthy': '#56B4E9',
    'SLE': '#F0E442',
    'Mild': '#2ca02c',
    'Severe': '#e377c2'
}

PALETTE_GENDER = {
    'Male': '#4287f5',
    'Female': '#f542ce',
    'M': '#4287f5',
    'F': '#f542ce'
}


def plot_predicted_vs_actual(
    adata_or_df,
    actual_col='age',
    predicted_col='predicted_age',
    hue=None,
    palette=None,
    s=50,
    alpha=0.5,
    title=None,
    figsize=(4, 4),
    ax=None,
    show_metrics=True
):
    """
    Scatter plot of predicted age vs actual age.
    
    Parameters
    ----------
    adata_or_df : AnnData or DataFrame
        Data with actual and predicted ages
    actual_col : str
        Column name for actual age (default: 'age')
    predicted_col : str
        Column name for predicted age (default: 'predicted_age')
    hue : str, optional
        Column name for color grouping (e.g., 'sex', 'dataset')
    palette : dict, optional
        Color palette for hue groups
    s : int
        Marker size (default: 50)
    alpha : float
        Transparency (default: 0.5)
    title : str, optional
        Plot title
    figsize : tuple
        Figure size (default: (4, 4))
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    show_metrics : bool
        Show Spearman correlation and R² (default: True)
    
    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """
    # Extract data
    if hasattr(adata_or_df, 'obs'):  # AnnData
        df = adata_or_df.obs
    else:  # DataFrame
        df = adata_or_df
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    if palette is None and hue == 'sex':
        palette = PALETTE_GENDER
    
    sns.scatterplot(
        data=df,
        x=actual_col,
        y=predicted_col,
        hue=hue,
        palette=palette,
        s=s,
        alpha=alpha,
        ax=ax
    )
    
    # Identity line
    min_age = df[actual_col].min()
    max_age = df[actual_col].max()
    ax.plot(
        [min_age, max_age],
        [min_age, max_age],
        color='gray',
        linestyle='--',
        label='Perfect prediction',
        zorder=0
    )
    
    # Calculate metrics
    if show_metrics:
        y_true = df[actual_col].values
        y_pred = df[predicted_col].values
        
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 0:
            spearman = spearmanr(y_true, y_pred)[0]
            r2 = r2_score(y_true, y_pred)
            
            if title is None:
                title = f"Spearman: {spearman:.3f}, R²: {r2:.3f}"
            else:
                title = f"{title}\nSpearman: {spearman:.3f}, R²: {r2:.3f}"
    
    ax.set_xlabel("Actual Age")
    ax.set_ylabel("Predicted Age")
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.1, y=0.1)
    
    if title:
        ax.set_title(title, pad=15)
    
    return ax


def plot_age_acceleration_disease(
    adata_or_df,
    disease_col='condition',
    healthy_label='Healthy',
    disease_label='Disease',
    cell_type_col='cell_type',
    age_col='age',
    predicted_age_col='predicted_age',
    figsize=(4, 2.7),
    palette=None,
    title=None,
    ax=None
):
    """
    Plot age acceleration in disease vs healthy controls.
    
    Shows the difference in predicted age between disease and healthy states
    for each cell type, with statistical testing.
    
    Parameters
    ----------
    adata_or_df : AnnData or DataFrame
        Data with predictions
    disease_col : str
        Column name for disease/condition labels
    healthy_label : str
        Label for healthy/control samples
    disease_label : str
        Label for disease samples
    cell_type_col : str
        Column name for cell type
    age_col : str
        Column name for actual age
    predicted_age_col : str
        Column name for predicted age
    figsize : tuple
        Figure size
    palette : dict, optional
        Color palette for conditions
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """
    # Extract data
    if hasattr(adata_or_df, 'obs'):
        df = adata_or_df.obs.copy()
    else:
        df = adata_or_df.copy()
    
    # Filter to relevant conditions
    df = df[df[disease_col].isin([healthy_label, disease_label])]
    
    if len(df) == 0:
        raise ValueError(f"No data found for {healthy_label} or {disease_label}")
    
    # Calculate age acceleration (difference between disease and healthy)
    df_pivot = df.pivot_table(
        index=[age_col, cell_type_col],
        columns=disease_col,
        values=predicted_age_col
    )
    df_pivot = df_pivot.reset_index()
    df_pivot['age_acceleration'] = df_pivot[disease_label] - df_pivot[healthy_label]
    df_pivot = df_pivot[~df_pivot['age_acceleration'].isna()]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get cell types
    cell_types = sorted(df_pivot[cell_type_col].unique())
    
    # Stripplot
    sns.stripplot(
        ax=ax,
        data=df_pivot,
        y='age_acceleration',
        x=cell_type_col,
        order=cell_types,
        alpha=0.7,
        hue=age_col,
        palette='viridis',
        s=6
    )
    
    # Bar plot for medians
    medians = df_pivot.groupby(cell_type_col)['age_acceleration'].median()
    medians = medians.reindex(cell_types)
    bar_x = np.arange(len(cell_types))
    ax.bar(
        bar_x,
        medians,
        width=0.5,
        fill=False,
        edgecolor='black',
        linewidth=1.5,
        zorder=0.1,
        hatch='////'
    )
    
    # Statistical tests
    pvals = []
    for cell_type in cell_types:
        group = df[df[cell_type_col] == cell_type]
        group_healthy = group[group[disease_col] == healthy_label][predicted_age_col]
        group_disease = group[group[disease_col] == disease_label][predicted_age_col]
        
        if len(group_healthy) > 1 and len(group_disease) > 1:
            _, pval = ttest_ind(group_disease, group_healthy, equal_var=False)
        else:
            pval = 1.0
        
        pvals.append(pval)
    
    # FDR correction
    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
    
    # Add significance stars
    for i, pval_fdr in enumerate(pvals_fdr):
        star = '***' if pval_fdr < 0.001 else '**' if pval_fdr < 0.01 else '*' if pval_fdr < 0.05 else ''
        if star:
            y_max = df_pivot[df_pivot[cell_type_col] == cell_types[i]]['age_acceleration'].max()
            ax.text(
                i, y_max + 5, star,
                ha='center', va='bottom',
                fontsize=10, weight='bold',
                color='black'
            )
    
    ax.set_ylabel("Age acceleration (years)")
    ax.set_xlabel("")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(x=0.2, y=0.2)
    ax.legend(title='Actual age', bbox_to_anchor=(1.01, .9), loc='upper left', frameon=False)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    
    if title:
        ax.set_title(title, fontsize=12, weight='bold', pad=15)
    
    plt.tight_layout()
    return ax


def plot_feature_importance(
    clock,
    top_n=20,
    figsize=(6, 8),
    ax=None,
    color='steelblue'
):
    """
    Plot top features by importance (coefficient magnitude).
    
    Parameters
    ----------
    clock : AgingClock
        Trained aging clock instance
    top_n : int
        Number of top features to show (default: 20)
    figsize : tuple
        Figure size (default: (6, 8))
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    color : str
        Bar color (default: 'steelblue')
    
    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """
    # Get feature importances
    importance_df = clock.get_feature_importance(top_n=top_n)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal barplot
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['coefficient'].values, color=color, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Top {top_n} Features - {clock.cell_type}', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    return ax


def plot_perturbation_effect(
    df_control,
    df_treatment,
    treatment_name='Treatment',
    donor_col='donor_id',
    predicted_age_col='predicted_age',
    paired=True,
    figsize=(3, 3),
    ax=None
):
    """
    Plot the effect of a perturbation/treatment on predicted age.
    
    Parameters
    ----------
    df_control : DataFrame
        Control group data with predicted ages
    df_treatment : DataFrame
        Treatment group data with predicted ages
    treatment_name : str
        Name of the treatment (default: 'Treatment')
    donor_col : str
        Column name for donor/sample IDs
    predicted_age_col : str
        Column name for predicted age
    paired : bool
        Whether data is paired (same donors in both groups)
    figsize : tuple
        Figure size (default: (3, 3))
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib.axes.Axes
        The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Combine data
    df_control = df_control.copy()
    df_treatment = df_treatment.copy()
    df_control['condition'] = 'Control'
    df_treatment['condition'] = treatment_name
    
    df_all = pd.concat([df_control, df_treatment])
    
    if paired and donor_col in df_all.columns:
        # Paired plot with lines connecting donors
        df_pivot = df_all.pivot_table(
            index=donor_col,
            columns='condition',
            values=predicted_age_col
        )
        df_pivot = df_pivot.dropna()
        
        df_plot = df_pivot.reset_index().melt(
            id_vars=donor_col,
            value_vars=['Control', treatment_name],
            var_name='condition',
            value_name=predicted_age_col
        )
        
        # Create donor palette
        donors = df_plot[donor_col].unique()
        donor_palette = dict(zip(donors, sns.color_palette("husl", len(donors))))
        
        sns.lineplot(
            data=df_plot,
            x='condition',
            y=predicted_age_col,
            hue=donor_col,
            marker='o',
            alpha=0.6,
            ax=ax,
            palette=donor_palette,
            legend=True
        )
        
        # Paired t-test
        if len(df_pivot) > 1:
            from scipy.stats import ttest_rel
            _, pval = ttest_rel(df_pivot['Control'], df_pivot[treatment_name])
            mean_diff = df_pivot[treatment_name].mean() - df_pivot['Control'].mean()
        else:
            pval = 1.0
            mean_diff = 0
    else:
        # Unpaired plot
        sns.stripplot(
            data=df_all,
            x='condition',
            y=predicted_age_col,
            order=['Control', treatment_name],
            ax=ax,
            alpha=0.7,
            s=8
        )
        
        # Unpaired t-test
        if len(df_control) > 1 and len(df_treatment) > 1:
            _, pval = ttest_ind(
                df_treatment[predicted_age_col],
                df_control[predicted_age_col],
                equal_var=False
            )
            mean_diff = df_treatment[predicted_age_col].mean() - df_control[predicted_age_col].mean()
        else:
            pval = 1.0
            mean_diff = 0
    
    # Add significance annotation
    star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    y_max = df_all[predicted_age_col].max()
    y_bracket = y_max + 7
    h = 2
    
    ax.plot([0, 0, 1, 1], [y_bracket - h, y_bracket, y_bracket, y_bracket - h],
            lw=1.5, c='black')
    ax.text(
        0.5, y_bracket + 1,
        f"{mean_diff:.2f} yrs {star}\n(p={pval:.3g})",
        ha='center', va='bottom', fontsize=10
    )
    
    ax.set_xlabel("")
    ax.set_ylabel("Predicted Age")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.margins(x=0.1, y=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    return ax
