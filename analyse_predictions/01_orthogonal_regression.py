# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '01_orthogonal_regression', file_name ='logs/01_orthogonal_regression.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np
import re

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from rdkit import Chem
from rdkit.Chem import Descriptors
from analyse_predictions.regression import orthogonal_regression_scipy

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import textwrap

from scipy.stats import mannwhitneyu

# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr',False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True

# number of bootstrapping iterations for orthogonal regression
N_BOOTSTRAP = 1000


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Calculate the Concordance Correlation Coefficient (CCC) between two arrays.
    CCC = 1 → perfect agreement (points lie exactly on the 45° line)
    CCC = 0 → no agreement
    CCC < 0 → systematic disagreement

    Why it’s better than 𝑅2

    R2 only measures how well points follow a linear trend (precision), ignoring bias (systematic offset).
    CCC accounts for both the correlation and how far the line is from the ideal 45° line (accuracy).

    Parameters:
        y_true (array-like): Experimental values
        y_pred (array-like): Predicted values

    Returns:
        ccc (float): Concordance correlation coefficient
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


# read the smiles used for the predictions
smiles = pd.read_excel(r'data/structures/smiles.xlsx')

# read the acute experimental data, the mg/L refer to the original structures and need to be converted to the standardised structures
experimental_data_acute = pd.read_pickle(r'data/fish_acute/processed/REACH_short_term_fish_measurement.pickle')
# keep only quantitative AFT data, no exclusion reasons
msk = experimental_data_acute['scenario'].isin(['AFT_1a', 'AFT_1b'])  & experimental_data_acute['exclusion reasons'].isnull()
# take the most conservative value if multiple measurements are available for the same standardised SMILES
experimental_data_acute = (experimental_data_acute
                           .loc[msk]
                           .groupby(['smiles'], dropna=False)
                           [['effect concentration (mol/L, lower bound)', 'CAS number']]
                           .agg(**{'effect concentration (mol/L)': pd.NamedAgg(column='effect concentration (mol/L, lower bound)', aggfunc='min'),
                                   'CAS number': pd.NamedAgg(column='CAS number', aggfunc=lambda x: ', '.join(sorted(set(x.dropna()))))}
                                )
                           .reset_index()
                           # keep only the DSSTox smiles that could be standardised
                           .merge(smiles[['smiles', 'smiles (standardised)']], how='inner', left_on='smiles', right_on='smiles')
                           .assign(**{'molecular weight (standardised)': lambda df: df['smiles (standardised)'].apply(lambda x: Chem.MolFromSmiles(x)).apply(lambda x: Descriptors.MolWt(x) if x is not None else np.nan)})
                           #  compute the effect levels in mg/L for the standardised structures
                           .assign(**{'effect concentration (mg/L)': lambda df: df['effect concentration (mol/L)']*df['molecular weight (standardised)']*1000})
                           .drop('smiles', axis='columns')
                           )

# read the chronic experimental data, the mg/L refer to the original structures and need to be converted to the standardised structures
experimental_data_chronic = pd.read_pickle(r'data/fish_chronic/processed/REACH_long_term_fish_measurement.pickle')
# keep only quantitative FELS data, no exclusion reasons
msk = experimental_data_chronic['scenario'].isin(['FLS_1a', 'FLS_1b']) & experimental_data_chronic['exclusion reasons'].isnull()
# take the most conservative value if multiple measurements are available for the same standardised SMILES
experimental_data_chronic = (experimental_data_chronic
                           .loc[msk]
                           .groupby(['smiles'], dropna=False)
                           [['effect concentration (mol/L, lower bound)', 'CAS number']]
                           .agg(**{'effect concentration (mol/L)': pd.NamedAgg(column='effect concentration (mol/L, lower bound)', aggfunc='min'),
                                   'CAS number': pd.NamedAgg(column='CAS number', aggfunc=lambda x: ', '.join(sorted(set(x.dropna()))))}
                            )
                           .reset_index()
                           # keep only the DSSTox smiles that could be standardised
                           .merge(smiles[['smiles', 'smiles (standardised)']], how='inner', left_on='smiles', right_on='smiles')
                           .assign(**{'molecular weight (standardised)': lambda df: df['smiles (standardised)'].apply(lambda x: Chem.MolFromSmiles(x)).apply(lambda x: Descriptors.MolWt(x) if x is not None else np.nan)})
                           #  compute the effect levels in mg/L for the standardised structures
                           .assign(**{'effect concentration (mg/L)': lambda df: df['effect concentration (mol/L)'] * df['molecular weight (standardised)']*1000})
                           .drop('smiles', axis='columns')
                           )


# put all experimental results together
experimental_data = pd.concat([experimental_data_acute.assign(**{'study type': 'acute'}),
                                    experimental_data_chronic.assign(**{'study type': 'chronic'})], axis='index', ignore_index=True, sort=False)
log.info(f'number of standardised smiles with experimental data: ' + experimental_data.groupby('study type')['smiles (standardised)'].nunique().to_json())

# read all predictions
prediction_files = [r'data/predictions/vega/processed/predictions_vega.xlsx',
                    r'data/predictions/ecosar/processed/predictions_ecosar.xlsx',
                    r'data/predictions/isaferat/processed/predictions_isaferat_AD_no_extrapolation.xlsx',
                    r'data/predictions/isaferat/processed/predictions_isaferat_AD_extrapolation_all.xlsx',
                    r'data/predictions/isaferat/processed/predictions_isaferat_AD_extrapolation_structural_in_domain.xlsx',
                    # r'data/predictions/test/relax_fragment_constraint_true/processed/predictions_test.xlsx',
                    r'data/predictions/test/relax_fragment_constraint_false/processed/predictions_test.xlsx',
                    r'data/predictions/kate/processed/predictions_kate.xlsx',
                    # 'data/predictions/trident/processed/predictions_trident.xlsx',
                    ]
prediction_data = []
for prediction_file in prediction_files:
    prediction_data.append(pd.read_excel(prediction_file))
prediction_data = pd.concat(prediction_data, axis='index', ignore_index=True, sort=False)
# for ECOSAR, create a model entry when there is only one class other than neutral organics
single_class = (prediction_data
             .loc[prediction_data['platform']=='ECOSAR']
             .groupby(['platform', 'model name', 'mol ID'])['ECOSAR Class']
             .nunique()
             .rename('number of ECOSAR classes')
             .reset_index()
             .query('`number of ECOSAR classes`== 2')
            )
# for ECOSAR, create a model entry when there are multiple classes in addition to neutral organics
multiple_class = (prediction_data
             .loc[prediction_data['platform']=='ECOSAR']
             .groupby(['platform', 'model name', 'mol ID'])['ECOSAR Class']
             .nunique()
             .rename('number of ECOSAR classes')
             .reset_index()
             .query('`number of ECOSAR classes`> 2')
            )
# examine how often neutral organics is the most conservative prediction, only in domain predictions
tmp = prediction_data.merge(multiple_class[['platform', 'model name', 'mol ID']], on=['platform', 'model name', 'mol ID'])
tmp = tmp.loc[tmp['no effects at saturation'] == 'no']
tmp = tmp.assign(**{'is neutral organics': np.where(tmp['ECOSAR Class']=='Neutral Organics','neutral organics','other class')})
tmp = tmp.pivot_table(index='smiles (standardised)', columns=['is neutral organics', 'study type'], values='prediction', aggfunc='min')
tmp.columns = [' '.join(col).strip() for col in tmp.columns.values]
tmp = tmp.assign(**{'neutral organics is most conservative (acute)': np.where(tmp['neutral organics acute'] < tmp['other class acute'], 'yes', 'no'),
                    'neutral organics is most conservative (chronic)': np.where(tmp['neutral organics chronic'] < tmp['other class chronic'], 'yes', 'no')})
tmp.to_excel(r'data/predictions/ECOSAR_multiple_classes_neutral_organics_most_conservative.xlsx')

prediction_data = pd.concat([prediction_data,
                             prediction_data.merge(single_class[['platform', 'model name', 'mol ID']], on=['platform', 'model name', 'mol ID']).assign(**{'model name': lambda df: df['model name'] + ' (single class)'}),
                             prediction_data.merge(multiple_class[['platform', 'model name', 'mol ID']], on=['platform', 'model name', 'mol ID']).assign(**{'model name': lambda df: df['model name'] + ' (multiple classes)'})])
# keep the required columns
cols = ['platform', 'model name', 'model version', 'study type', 'mol ID',
       'smiles (standardised)', 'prediction status', 'training/validation set',
       'AD', 'predicted quantity', 'prediction', 'no effects at saturation',
       'notes']
prediction_data = prediction_data[cols]

# keep only the predictions for which:
# - the prediction succeeded
# - the prediction did not show no effects at saturation
msk = ((prediction_data['prediction status'] == 'succeeded')
       & (prediction_data['no effects at saturation'] == 'no')
       & prediction_data['predicted quantity'].str.contains(r'\((?:m|u)g/L\)'))
prediction_data = prediction_data.loc[msk]
# keep the most conservative prediction if multiple predictions are available for the same standardised SMILES
prediction_data = (prediction_data
                   .sort_values(['platform', 'model name', 'smiles (standardised)', 'study type', 'prediction'], ascending=[True, True, True, True, True])
                   .drop_duplicates(subset=[ 'platform', 'model name', 'smiles (standardised)', 'study type',], keep='first')
                   .reset_index(drop=True)
                   )

# read the charge states at pH 6, 7, 8
ACD_disccosiation_predictions = pd.read_excel(r'data/predictions/ACDPercepta/processed/ACDPercepta_dissociation_predictions.xlsx')
ACD_disccosiation_predictions = ACD_disccosiation_predictions[['smiles (standardised)', 'prediction status', 'charge at pH 6', 'charge at pH 7', 'charge at pH 8']]


# loop over all models
models = prediction_data[['platform', 'model name', 'study type']].drop_duplicates().reset_index()
model_predictive_performances = []
for model_idx, model_row in models.iterrows():
    platform = model_row['platform']
    model_name = model_row['model name']
    stydy_type = model_row['study type']
    model_predictive_performance = {'study type': stydy_type, 'platform': platform, 'model name': model_name}
    log.info(f'Processing model {platform}: {model_name} for study type {stydy_type}')

    model_predictions = (prediction_data.query('(`model name` == @model_name) & (platform == @platform)')
                         [['smiles (standardised)', 'training/validation set', 'AD', 'predicted quantity', 'prediction', 'study type']]
                         .drop_duplicates())

    # convert units if necessary
    if model_predictions['predicted quantity'].iloc[0] in ['LC50 (mg/L)', 'ChV (mg/L)', 'NOEC (mg/L)', 'EC10 (mg/L)']:
        pass
    elif model_predictions['predicted quantity'].iloc[0] == 'AC50 (ug/L)':
        model_predictions['prediction'] = model_predictions['prediction']*1.e-3
    else:
        raise ValueError(f'Unknown predicted quantity {model_predictions["predicted quantity"].iloc[0]} for model {model_name}')

    res = experimental_data.merge(model_predictions, how='inner', left_on=['smiles (standardised)', 'study type'], right_on=['smiles (standardised)', 'study type'])
    res['prediction'] = pd.to_numeric(res['prediction'], errors='coerce')
    res = res.query('prediction.notnull()').sort_values(by='prediction', ascending=False).reset_index(drop=True)
    res.insert(loc=0, column='model name', value=model_name)
    res.insert(loc=0, column='platform', value=platform)
    res.to_excel(f'data/predictions/predictions_vs_experimental_{platform}_{model_name.replace(" ", "_")}_{model_row["study type"]}.xlsx', index=False)

    # compute the fraction of predictions that are in domain, out of the training/validation set and deviate by more than 1 log unit, separately for over and under predictions
    msk = (res['AD'] == 'in domain') & (res['training/validation set'] == 'not in training/validation set')
    diff = res.loc[msk].assign(**{'diff': np.log10(res.loc[msk]['prediction']) - np.log10( res.loc[msk]['effect concentration (mg/L)'])})
    bins = [-np.inf, -2, -1, 1, 2, np.inf]
    labels = ['over protective by > 2 log units', 'over protective by 1-2 log units', 'within ±1 log unit', 'under protective by 1-2 log units', 'under protective by > 2 log units']
    diff['diff category'] = pd.cut(diff['diff'], bins=bins, labels=labels)
    diff_stats = (diff['diff category']
                  .value_counts(normalize=True)
                  .rename('proportion')
                  .reset_index()
                  .rename(columns={'index': 'diff category'})
                  .set_index('diff category')
                  .reindex(labels)
                  )
    model_predictive_performance.update({f'% {row.name} (in domain, not in training/validation set)': row['proportion'] for _, row in diff_stats.iterrows()})

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(nrows=1, ncols=1)
    xlim = (-4, 5)
    ylim = (-4, 5)

    #  in domain, not in training/validation predictions
    msk = (res['AD'] == 'in domain') & (res['training/validation set'] == 'not in training/validation set')
    if msk.sum()>0:
        x = np.log10(res.loc[msk, 'effect concentration (mg/L)'])
        y = np.log10(res.loc[msk, 'prediction'])
        scatter = ax.scatter(x, y, s=30, alpha=1, marker='o', edgecolor='k', facecolor='k', label=f'in domain predictions (n={len(x)})', zorder=1)
        reg = orthogonal_regression_scipy(x, y, n_bootstrap=N_BOOTSTRAP)
        xx = np.array(xlim)
        yy = reg['intercept']['median'] + reg['slope']['median']*xx
        ax.plot(xx, yy, 'k-', alpha=0.5, lw=1., label=f"slope={reg['slope']['median']:.2f} (95% CI: {reg['slope']['2.5%']:.2f} to {reg['slope']['97.5%']:.2f})\nintercept={reg['intercept']['median']:.2f} (95% CI: {reg['intercept']['2.5%']:.2f} to {reg['intercept']['97.5%']:.2f}), R2={reg['exact']['R2 ODR']: 0.2f}", zorder=3)
        model_predictive_performance.update({'n (in domain)': len(x), 'R2 (in domain)': reg['exact']['R2 ODR'], 'RMSE (in domain)': reg['exact']['RMSE'], 'MAE (in domain)': reg['exact']['MAE'],
                                             'CCC (in domain)': concordance_correlation_coefficient(x, y), 'orthogonal regression (in domain)': reg})

        # prepare the data for the GHS classification heatmap
        if stydy_type == 'acute':
            ghs_classes = [-np.inf, 0.1, 1, 10, 100, np.inf]  # mg/L
        elif stydy_type == 'chronic':
            ghs_classes = [-np.inf, 0.001, 0.01, 0.1, 1, np.inf]  # mg/L
        # count the number of points in each GHS classification bin for experimental and predicted values, rows have the true values and columns the predicted values
        confusion_matrix = pd.crosstab(pd.cut(10**x, bins=ghs_classes, right=False), pd.cut(10**y, bins=ghs_classes, right=False), dropna=False)
        model_predictive_performance.update({'confusion matrix (in domain)': confusion_matrix})


        # analyse the absolute error depending on the charge state at pH 6, 7, or 8, keep only molecules for which dissociation was modelled successfully
        absolute_error_vs_charge = (res
                                    .loc[msk, ['smiles (standardised)', 'effect concentration (mg/L)', 'prediction']]
                                    .merge(ACD_disccosiation_predictions.query('`prediction status`=="succeeded"'), on='smiles (standardised)', how='inner')
                                    .assign(**{'charged at neutral pH': lambda df: np.where((df['charge at pH 6'].abs() > 0.5) | (df['charge at pH 7'].abs() > 0.5) | (df['charge at pH 8'].abs() > 0.5), 'charged', 'not charged')})
                                    .assign(**{'absolute error (log10 units)': lambda df: (np.log10(df['prediction']) - np.log10(df['effect concentration (mg/L)'])).abs()})
                                    )
        log.info(f'From the {len(res)} in domain predictions not in training/validation set, states at neutral pH are {absolute_error_vs_charge["charged at neutral pH"].value_counts().to_json()}, and {len(res)-len(absolute_error_vs_charge)} have no dissociation prediction available')
        absolute_errors_ionised = absolute_error_vs_charge.loc[absolute_error_vs_charge['charged at neutral pH']=='charged', 'absolute error (log10 units)']
        log.info(f'Absolute errors (log10 units) for ionised at neutral pH: {absolute_errors_ionised.describe().to_json()}')
        absolute_errors_nonionised = absolute_error_vs_charge.loc[absolute_error_vs_charge['charged at neutral pH']=='not charged', 'absolute error (log10 units)']
        log.info(f'Absolute errors (log10 units) for non-ionised at neutral pH: {absolute_errors_nonionised.describe().to_json()}')
        stat, p = mannwhitneyu(absolute_errors_ionised, absolute_errors_nonionised, alternative="two-sided")
        log.info(f'Mann-Whitney U test for absolute errors of ionised vs non-ionised at neutral pH: statistic={stat}, p-value={p}')
        # .. compute the cliff delta from the U statistic
        n1 = len(absolute_errors_ionised)
        n2 = len(absolute_errors_nonionised)
        cliff_delta = (2*stat)/(n1*n2) - 1
        log.info(f'Cliff delta for absolute errors of ionised vs non-ionised at neutral pH: {cliff_delta:.2f}')
        # .. compute the cliff delta manually
        model_predictive_performance.update({'n non-ionised (in domain)': len(absolute_errors_nonionised),
                                             'n ionised (in domain)': len(absolute_errors_ionised),
                                             'MAE log10 non-ionised (in domain)': absolute_errors_nonionised.mean(),
                                             'MAE log10 ionised (in domain)': absolute_errors_ionised.mean(),
                                             'MAE log10 Mann-Whitney U statistic(in domain)': stat,
                                             'MAE log10 Mann-Whitney U p-value (in domain)': p,
                                             'MAE log10 Cliff delta (in domain)': cliff_delta,})
        # plot horizontal boxplot to compare absolute errors for ionised vs non-ionised at neutral pH, figure plot only if there are datapoints in both categories
        if (len(absolute_errors_ionised) > 0) and (len(absolute_errors_nonionised) > 0):
            fig_charge = plt.figure(figsize=(8, 6))
            ax_charge = fig_charge.subplots(nrows=1, ncols=1)
            order = np.sort(absolute_error_vs_charge['charged at neutral pH'].unique())
            sns.boxplot(data=absolute_error_vs_charge, x='absolute error (log10 units)', y='charged at neutral pH', ax=ax_charge, color='orange', order=order, showmeans=True,
                        medianprops=dict(color='black', linewidth=1),
                        meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=6)
                        )
            ax_charge.set_xlabel('absolute error (log10 units)', fontsize=10)
            ax_charge.set_ylabel('')
            ax_charge.set_title(platform + ', ' + model_name, fontsize=10)
            # remove left, top, right spines and offset bottom spine
            ax_charge.spines['left'].set_visible(False)
            ax_charge.spines['top'].set_visible(False)
            ax_charge.spines['right'].set_visible(False)
            ax_charge.spines['bottom'].set_position(('outward', 10))
            # annotate the median absolute error for each category, using vertical black text to the right of the median line
            for i, category in enumerate(order):
                median_absolute_error = absolute_error_vs_charge.loc[absolute_error_vs_charge['charged at neutral pH']==category, 'absolute error (log10 units)'].median()
                ax_charge.text(median_absolute_error-0.05 , i, f'AE median: {median_absolute_error:.2f}', ha='right', va='center', fontsize=8, color='black', rotation=90)
            # annotate the mean absolute error for each category, using vertical black text below the mean circle
            for i, category in enumerate(order):
                mean_absolute_error = absolute_error_vs_charge.loc[absolute_error_vs_charge['charged at neutral pH']==category, 'absolute error (log10 units)'].mean()
                ax_charge.text(mean_absolute_error , i-0.05, f'AE mean: {mean_absolute_error:.2f}', ha='center', va='bottom', fontsize=8, color='black', rotation=90)

            # add an annotation box with the study type
            ax_charge.text(0.6, 0.6, model_row['study type'], ha='left', va='top', fontsize=14, color='white', transform=ax_charge.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8), zorder=10)
            # add an annotation box with the number of data points for charged and not charged, as well as the p-value from the Mann-Whitney U test
            ax_charge.text(0.6, 0.5, f'not charged n={len(absolute_errors_nonionised)}\ncharged n={len(absolute_errors_ionised)}\np-value={p:.2e}, $\\delta$={cliff_delta:.2f}', ha='left', va='top', fontsize=10, color='black', transform=ax_charge.transAxes, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=10)
            fig_charge.tight_layout()
            fig_charge.savefig(f'figures/effect_ionisation/absolute_error_boxplot_charge_state_{platform}_{model_name.replace(" ", "_")}_{stydy_type}_charge_state.png', dpi=600)
            plt.close(fig_charge)

        # .. annotate the points for which toxicity is underestimated by more than 1 log unit (show only if less than 20 points)
        if (y - x > 1).sum() < 20:
            offset = 0
            for i in range(len(x)):
                if (y.iloc[i] - x.iloc[i]) > 1:
                    offset += 0.025
                    label = '-' if pd.isnull(cas_label := res.loc[msk].iloc[i]['CAS number']) else cas_label
                    # add the label if the prediction underestimates the experimental value by more than 1 log unit
                    ax.text(0.10, 0.98-offset, label, ha='right', va='center', fontsize=6, color='k', transform=ax.transAxes, zorder=2)
                    # annotate label at fixed position in the top-left
                    ax.annotate(
                        '',
                        xy=(x.iloc[i], y.iloc[i]),  # point being annotated
                        xytext=(0.105, 0.98-offset),  # relative position in axes coords (top-left)
                        textcoords="axes fraction",
                        ha='right', va='center',
                        fontsize=6, color='k',
                        arrowprops=dict(
                            arrowstyle="->",
                            color="grey",
                            lw=0.5,
                            shrinkA=0,  # shrink from label side
                            shrinkB=5  # shrink from point side (distance in points)
                        ),
                        zorder=1
                    )


    #  out of domain, not in training/validation predictions
    msk = (res['AD'] == 'out of domain') & (res['training/validation set'] == 'not in training/validation set')
    if msk.sum()>0:
        x = np.log10(res.loc[msk, 'effect concentration (mg/L)'])
        y = np.log10(res.loc[msk, 'prediction'])
        scatter = ax.scatter(x, y, s=30, alpha=0.5, marker='o', edgecolor='k', facecolor='orange', label=f'out of domain predictions (n={len(x)})', zorder=3)
        reg = orthogonal_regression_scipy(x, y, n_bootstrap=N_BOOTSTRAP)
        xx = np.array(xlim)
        yy = reg['intercept']['median'] + reg['slope']['median']*xx
        ax.plot(xx, yy, color='orange', linestyle='-', alpha=0.5, lw=1., label=f"slope={reg['slope']['median']:.2f} (95% CI: {reg['slope']['2.5%']:.2f} to {reg['slope']['97.5%']:.2f})\nintercept={reg['intercept']['median']:.2f} (95% CI: {reg['intercept']['2.5%']:.2f} to {reg['intercept']['97.5%']:.2f}), R2={reg['exact']['R2 ODR']: 0.2f}", zorder=2)
        model_predictive_performance.update({'n (out of domain)': len(x), 'r2 (out of domain)': reg['exact']['R2 ODR'], 'RMSE (out of domain)': reg['exact']['RMSE'], 'MAE (out of domain)': reg['exact']['MAE'],
                                             'CCC (out of domain)': concordance_correlation_coefficient(x, y), 'orthogonal regression (out of domain)': reg})

    #  training set predictions
    msk = res['training/validation set'] == 'training set'
    if msk.sum():
        x = np.log10(res.loc[msk, 'effect concentration (mg/L)'])
        y = np.log10(res.loc[msk, 'prediction'])
        scatter = ax.scatter(x, y, s=30, alpha=1, marker='o', edgecolor='r', facecolor='r', label=f'in training set (n={len(x)})', zorder=2)
        ax.set_xlabel('log10 (experimental LC50, mg/L)')
        ax.set_ylabel('log10 (predicted LC50, mg/L)')
        reg = orthogonal_regression_scipy(x, y, n_bootstrap=N_BOOTSTRAP)
        xx = np.array(xlim)
        yy = reg['intercept']['median'] + reg['slope']['median']*xx
        ax.plot(xx, yy, color='r', linestyle='-', alpha=0.5, lw=1., label=f"slope={reg['slope']['median']:.2f} (95% CI: {reg['slope']['2.5%']:.2f} to {reg['slope']['97.5%']:.2f})\nintercept={reg['intercept']['median']:.2f} (95% CI: {reg['intercept']['2.5%']:.2f} to {reg['intercept']['97.5%']:.2f}), R2={reg['exact']['R2 ODR']: 0.2f}", zorder=1)
        model_predictive_performance.update({'n (training set)': len(x), 'r2 (training set)': reg['exact']['R2 ODR'], 'RMSE (training set)': reg['exact']['RMSE'], 'MAE (training set)': reg['exact']['MAE'],
                                             'CCC (training set)': concordance_correlation_coefficient(x, y), 'orthogonal regression (training set)': reg})

    # axes labels
    ax.set_xlabel('log10 (experimental LC50, mg/L)', fontsize=10)
    ax.set_ylabel('log10 (predicted LC50, mg/L)', fontsize=10)
    # set the ticklabels to fontsize 10
    ax.tick_params(axis='both', which='major', labelsize=10)
    # set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # handle spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['left'].set_position(('outward', 10))
    # add legend, bottom left
    legend = ax.legend(frameon=True, loc='lower right', fontsize=9)

    # add the y=x line and 1 log unit errors
    x = np.linspace(-3.5, 5.5, 200)
    y_center = x
    y_upper = x + 1
    y_lower = x - 1
    ax.fill_between(x, y_lower, y_upper, color='lightgrey', alpha=0.5, zorder=0)
    ax.plot(x, y_center, color='grey', linestyle='--', alpha=0.7, lw=1.5, zorder=5)
    ax.set_title(platform + ', ' + model_name, fontsize=10)

    # add an annotation box with the study type
    ax.text(0.9, 0.5, model_row['study type'], ha='left', va='top', fontsize=14, color='white', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8), zorder=10)

    model_name = re.sub(r'[\s\(\)\-]+', '_', model_name)
    fig.savefig(fr'figures/orthogonal_regression/{platform}_{model_name}.png', dpi=600)

    model_predictive_performances.append(model_predictive_performance)


# export the model predictive performances
model_predictive_performances = pd.DataFrame(model_predictive_performances)
def flatten_json_columns(df, cols):
    out = df.copy()
    for col in cols:
        # normalize each JSON column
        norm = pd.json_normalize(out[col]).add_prefix(f"{col}.")
        # drop the original and join flattened
        out = out.drop(columns=[col]).join(norm)
    return out
model_predictive_performances = flatten_json_columns(model_predictive_performances, ['orthogonal regression (in domain)', 'orthogonal regression (out of domain)', 'orthogonal regression (training set)'])
model_predictive_performances = model_predictive_performances.sort_values(by=['study type', 'platform', 'model name'])
model_predictive_performances = model_predictive_performances.sort_values(by=['study type', 'platform', 'model name'], ascending=True).reset_index(drop=True)
model_predictive_performances.to_excel(r'data/predictions/model_predictive_performances_quantitative.xlsx', index=False)


# create heatmap with distribution of number of predictions in domain, out of domain and in training set that deviate by more than 1 log unit
heatmap_data = model_predictive_performances[['platform', 'model name',
                                              '% under protective by > 2 log units (in domain, not in training/validation set)',
                                              '% under protective by 1-2 log units (in domain, not in training/validation set)',
                                              '% within ±1 log unit (in domain, not in training/validation set)',
                                              '% over protective by 1-2 log units (in domain, not in training/validation set)',
                                              '% over protective by > 2 log units (in domain, not in training/validation set)']].set_index(['platform', 'model name'])
column_mapping = {
    '% under protective by > 2 log units (in domain, not in training/validation set)': 'under\nprotective\n>2 log units',
    '% under protective by 1-2 log units (in domain, not in training/validation set)': 'under\nprotective\n1–2 log units',
    '% within ±1 log unit (in domain, not in training/validation set)': 'within\n±1 log units',
    '% over protective by 1-2 log units (in domain, not in training/validation set)': 'over\nprotective\n1–2 log units',
    '% over protective by > 2 log units (in domain, not in training/validation set)': 'over\nprotective\n>2 log units'
}
# .. prepare heatmap data
heatmap_data = model_predictive_performances[['platform', 'model name'] + list(column_mapping.keys())].rename(columns=column_mapping)
heatmap_data = heatmap_data.set_index(['platform', 'model name'])
value_df = 100*heatmap_data[column_mapping.values()]   # only the percentage columns
# .. prepare bar plot data
n_in_domain = model_predictive_performances.set_index(['platform', 'model name'])['n (in domain)']
n_in_domain = n_in_domain.loc[value_df.index].fillna(0).astype(int)
annot = value_df.map(lambda x: f"{x:.00f} %")
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(
    2, 4,
    width_ratios=[0.75, 3.75, 6.0, 1.5],   # heatmap wide, bar plot narrow
    height_ratios=[30, 1],   # colorbar on bottom
    hspace=0.05, wspace=0.05
)
# .. create the heatmap
ax_hm = fig.add_subplot(gs[0, 2])
hm = sns.heatmap(
    value_df,
    cmap="Blues",
    annot=annot,
    annot_kws={"size": 8},
    fmt="s",
    linewidths=1.5,
    cbar=False,       # create custom colorbar below
    vmin=0,
    vmax=100,
    ax=ax_hm
)
ax_hm.set_xticks(0.5 + np.arange(len(value_df.columns)))
ax_hm.tick_params(axis="x", length=0)  # move x ticks away from heatmap
ax_hm.xaxis.set_ticks_position("top")
ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=0, ha="center", fontsize=8)
ax_hm.set_yticks([])
ax_hm.set_ylabel("")
# .. add model names in the 2nd grid cell as text labels
ax_names = fig.add_subplot(gs[0, 1])
ax_names.set_xlim(0, 1)
ax_names.set_ylim(0, len(value_df.index))
for i, (platform, model_name) in enumerate(value_df.index):
    ax_names.text(
        1, 0.5 + i,
        f"{model_name}",
        ha="right",
        va="center",
        fontsize=8
    )
ax_names.axis("off")  # hide the axes
ax_names.invert_yaxis()  # match heatmap row order
# .. add platform names in the 1st grid cell as text labels
ax_platform = fig.add_subplot(gs[0, 0])
ax_platform.set_xlim(0, 1)
ax_platform.set_ylim(0, len(value_df.index))
current_platform = None
for i, (platform, model_name) in enumerate(value_df.index):
    if current_platform is None or platform != current_platform:
        ax_platform.text(
            1, 0.5 + i,
            f"{platform}",
            ha="right",
            va="center",
            fontsize=8
        )
        current_platform = platform
ax_platform.axis("off")  # hide the axes
ax_platform.invert_yaxis()  # match heatmap row order
# .. colorbar below heatmap
ax_cb = fig.add_subplot(gs[1, 2])
cbar = fig.colorbar(
    hm.collections[0],
    cax=ax_cb,
    orientation="horizontal",

)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("")
ax_cb.tick_params(
    axis='x',      # or 'y' for vertical colorbar
    length=0       # set tick length to 0
)
# .. bar plot on the right
ax_bar = fig.add_subplot(gs[0, 3])
ax_bar.barh(
    y=0.5 + np.arange(len(n_in_domain)),
    width=n_in_domain.values,
    color="grey"
)
ax_bar.set_ylim(0, len(n_in_domain))
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_visible(False)
ax_bar.spines['bottom'].set_position(('outward', 5))
ax_bar.set_yticks([])  # remove duplicate labels
ax_bar.set_xlabel("# structures", fontsize=8)
ax_bar.tick_params(axis='x', labelsize=8)
ax_bar.invert_yaxis()  # match heatmap row order
pos_bar = ax_bar.get_position()
ax_bar.set_position([
    pos_bar.x0 + 0.01,  # shift right
    pos_bar.y0,
    pos_bar.width,
    pos_bar.height
])
for i, v in enumerate(n_in_domain):
    ax_bar.text(
        v + 10, 0.5 + i,
        str(v),
        ha="left",
        va="center",
        fontsize=8,
        color="black"
    )
# draw horizontal lines to separate platforms
current_platform = None
for i, (platform, model_name) in enumerate(value_df.index):
    if platform != current_platform:
        if current_platform is not None:
            ax_hm.hlines(i, *ax_hm.get_xlim(), colors='black', linewidth=0.5)
            ax_names.hlines(i, *ax_names.get_xlim(), colors='black', linewidth=0.5)
            ax_platform.hlines(i, *ax_platform.get_xlim(), colors='black', linewidth=0.5)
            ax_bar.hlines(i, *ax_bar.get_xlim(), colors='black', linewidth=0.5)
        current_platform = platform
# add an annotation box with the study type
ax_hm.set_xlim(-0.25, 5)
for study_type in ['acute', 'chronic']:
    x = -0.25
    y0 = model_predictive_performances.loc[model_predictive_performances['study type'] == study_type].index.min()+0.25
    y1 = model_predictive_performances.loc[model_predictive_performances['study type'] == study_type].index.max()+0.75
    height = y1 - y0
    width = 0.2   # adjust as needed
    box = patches.FancyBboxPatch(
        (x, y0),             # lower-left corner
        width,                       # box width
        height,                      # box height
        boxstyle="round,pad=0,rounding_size=0.1",    # same rounded style
        facecolor='black',
        edgecolor='none',
        alpha=1.0,
        zorder=10,
        transform=ax_hm.transData
    )
    ax_hm.add_patch(box)
    ax_hm.text(
    x + width/2,
    (y0 + y1) / 2,
    study_type,
    ha='center',
    va='center',
    fontsize=8,
    color='white',
    rotation=90,
    transform=ax_hm.transData,
    zorder=15
    )
plt.savefig("figures/orthogonal_regression/model_predictive_performances_in_domain_not_train_val_heatmap.png", dpi=600)


# heatmaps for the GHS classification confusion matrices
excluded_models = [#'96h-LC50 to fish (Opera MP), extr. AD excluded',
                   #'96h-LC50 to fish (Opera MP), extr. AD included',
                   '96h-LC50 to fish (Opera MP), extr. AD included (str. AD in domain)',

                   '96h-LC50 to fish (no MP), extr. AD excluded',
                   '96h-LC50 to fish (no MP), extr. AD included',
                   '96h-LC50 to fish (no MP), extr. AD included (str. AD in domain)',

                   #'32d-EC10 to fish (Opera MP), extr. AD excluded',
                   #'32d-EC10 to fish (Opera MP), extr. AD included',
                   '32d-EC10 to fish (Opera MP), extr. AD included (str. AD in domain)',

                   '32d-EC10 to fish (no MP), extr. AD excluded',
                   '32d-EC10 to fish (no MP), extr. AD included',
                   '32d-EC10 to fish (no MP), extr. AD included (str. AD in domain)'
                   ]
heatmaps = model_predictive_performances.loc[~model_predictive_performances['model name'].isin(excluded_models), ['platform', 'model name', 'study type', 'confusion matrix (in domain)']].dropna(subset='confusion matrix (in domain)')

# grid size
rows, cols = 5, 5

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

# flatten axes for easy iteration
axes_flat = axes.flatten()
for i_model, ax in enumerate(axes_flat):
    if i_model < len(heatmaps):
        data = heatmaps.iloc[i_model]['confusion matrix (in domain)']/sum(heatmaps.iloc[i_model]['confusion matrix (in domain)'].values.flatten())
        data_max = 0.4
        # create red and blue colormaps
        red_cmap = LinearSegmentedColormap.from_list("red_map", ["white", "red"])
        blue_cmap = LinearSegmentedColormap.from_list("blue_map", ["white", "blue"])

        # masks
        upper_mask = np.triu(np.ones_like(data), k=1).astype(bool)  # above diagonal
        lower_mask = np.tril(np.ones_like(data), k=-1).astype(bool)  # below diagonal

        # plot lower triangle (blue)
        ax.imshow(np.where(lower_mask, data, np.nan), cmap=blue_cmap, vmin=0, vmax=data_max)

        # plot upper triangle (red)
        ax.imshow(np.where(upper_mask, data, np.nan), cmap=red_cmap, vmin=0, vmax=data_max)

        # plot diagonal (optional: gray)
        ax.imshow(np.where(~(upper_mask | lower_mask), data, np.nan), cmap=blue_cmap, vmin=0, vmax=data_max)

        # add text annotations
        for (j, k), value in np.ndenumerate(data.values):
            if value > 0.5*data_max:
                ax.text(k, j, f"{100*value:.0f}%", ha='center', va='center', color='white', fontsize=6, transform=ax.transData)
            else:
                ax.text(k, j, f"{100*value:.0f}%", ha='center', va='center', color='black', fontsize=6, transform=ax.transData)

        # set the title
        title = textwrap.fill(f"{heatmaps.iloc[i_model]['platform']}, {heatmaps.iloc[i_model]['model name']}", width=30)
        ax.set_title(title, fontsize=7)

        # set the tick labels
        def fmt(v):
            """Format number by removing trailing .0"""
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)
        def interval_to_label(interval):
            left, right = interval.left, interval.right
            # (-inf, a)
            if left == float("-inf"):
                return f"<{fmt(right)}"
            # [a, inf)
            if right == float("inf"):
                return f">={fmt(left)}"
            # [a, b)
            return f"[{fmt(left)},{fmt(right)})"
        ghs_labels = [interval_to_label(iv) for iv in data.columns.categories.tolist()]
        ax.set_xticks(np.arange(len(ghs_labels)))
        ax.set_xticklabels(ghs_labels, fontsize=6, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='center')
        ax.set_yticks(np.arange(len(ghs_labels)))
        ax.set_yticklabels(ghs_labels, fontsize=6, fontweight='bold')
        # hide the ticks
        ax.tick_params(axis="both", which="both", length=0)
        # hide the border around the heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)

        # add a rectangle around each diagonal cell
        n = data.shape[0]  # number of rows/columns
        for i in range(n):
            rect = patches.Rectangle(
                (i - 0.5, i - 0.5),  # (x, y) bottom-left corner
             1,  # width
            1,  # height
                fill=False,
                edgecolor='black',
                linewidth=1
                )
            ax.add_patch(rect)
    else:
        ax.axis("off")   # Leave last cell(s) empty
plt.tight_layout()
# add the acute/chronic legend in the top left corner (needs to be done after tight_layout)
for i_model, ax in enumerate(axes_flat):
    if i_model < len(heatmaps):
        # add a black circle with white text in the top-left corner of the subplot to indicate an acute or chronic model.
        bbox = ax.get_position()
        rel_size = 0.06 * bbox.width
        x_offset = 0.08 * bbox.width  # move right if positive, left if negative
        y_offset = 0.08 * bbox.height  # move down if negative, up if positive
        cx = bbox.x0 - x_offset  # slightly left
        cy = bbox.y1 + y_offset  # slightly above
        # draw circle
        col = 'black' if heatmaps.iloc[i_model]['confusion matrix (in domain)'].values.sum()>=50 else 'gray'
        circle = patches.Circle(
            (cx, cy),
            radius=rel_size,
            transform=fig.transFigure,
            facecolor=col,
            edgecolor=col,
            zorder=50
        )
        fig.patches.append(circle)
        # add letter
        fig.text(
            cx, cy,
            'A' if heatmaps.iloc[i_model]['study type'] == 'acute' else 'C',
            color="white",
            ha="center",
            va="center",
            transform=fig.transFigure,
            fontsize=8,
            zorder=51
        )
fig.savefig("figures/orthogonal_regression/model_predictive_performances_in_domain_confusion_matrices.png", dpi=600)


# horizontal barplot with the Mann-Whitney U test p values for absolute errors of ionised vs non-ionised at neutral pH
mw_results = model_predictive_performances[['platform', 'model name', 'study type',
                                            'n non-ionised (in domain)',
                                            'n ionised (in domain)',
                                            'MAE log10 non-ionised (in domain)',
                                            'MAE log10 ionised (in domain)',
                                            'MAE log10 Mann-Whitney U statistic(in domain)',
                                            'MAE log10 Mann-Whitney U p-value (in domain)',
                                            'MAE log10 Cliff delta (in domain)']].dropna(subset=['MAE log10 Mann-Whitney U p-value (in domain)'])
fig = plt.figure(figsize=(12, 6))
ax = fig.subplots(nrows=1, ncols=1)
# sort by p-value
mw_results = mw_results.sort_values(by='MAE log10 Mann-Whitney U p-value (in domain)', ascending=True).reset_index(drop=True)
# horizontal bar plot
bars = ax.barh(
    y=np.arange(len(mw_results)),
    width=-np.log10(mw_results['MAE log10 Mann-Whitney U p-value (in domain)']),
    color='orange',
    edgecolor='none',
    zorder=2
)
ax.set_yticks(np.arange(len(mw_results)))
ax.set_yticklabels([f"{row['platform']}, {row['model name']}" for _, row in mw_results.iterrows()], fontsize=8)
ax.set_xlabel('-log10 (Mann-Whitney U p-value)', fontsize=10)
# add vertical line at p=0.05
ax.axvline(-np.log10(0.05), color='red', linestyle='--', lw=1, zorder=1)
ax.text(-np.log10(0.05)+0.05, len(mw_results), 'p=0.05', color='red', fontsize=8, va='center', zorder=1)
# annotate bars with p-value, cliff delta, and % ionised
for i, row in mw_results.iterrows():
    ax.text(
        -np.log10(row['MAE log10 Mann-Whitney U p-value (in domain)']) + 0.1,
        i,
        f"p={row['MAE log10 Mann-Whitney U p-value (in domain)']:.2e}, charged: {row['n ionised (in domain)']:.0f} out of {row['n ionised (in domain)'] + row['n non-ionised (in domain)']:.0f}",
        fontsize=8,
        va='center',
        zorder=2
    )
# remove the spines at the top, left and right, move bottom and left outwards
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_position(('outward', 10))
# make the y-ticks zero length
ax.tick_params(axis='y', length=0)
fig.tight_layout()
plt.savefig("figures/effect_ionisation/absolute_error_mann_whitney_u_charge_state.png", dpi=600)