# setup logging
import logger
import logging
log = logger.setup_applevel_logger(logger_name = '05_radial_dendrogram_plot', file_name ='logs/05_radial_dendrogram_plot.log', level_stream=logging.INFO, level_file=logging.DEBUG)

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat
from scipy.cluster import hierarchy

from cheminformatics.fingerprint_descriptor_distances import compute_morgan_fingerprint

import matplotlib
%matplotlib
# matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import radialtree as rt
from radialtree.radialtree import radialTreee

from rdkit import Chem
from rdkit.Chem import Descriptors


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

# read in the standardised SMILES
smiles_set = (pd.read_excel(rf'data/structures/smiles.xlsx', sheet_name='smiles (source)')
              [['smiles', 'smiles (standardised)']]
              .drop_duplicates())

for endpoint in ['acute', 'chronic']:
    for toxicity_band_basis in ['mass', 'molar']:
        log.info(f"Processing fish {endpoint} toxicity data with toxicity band basis: {toxicity_band_basis}")

        if endpoint == 'acute':
            # read the acute fish toxicity data
            REACH_short_term_fish = pd.read_excel(fr'data/fish_acute/processed/REACH_short_term_fish_measurement.xlsx')

            # keep scenarios AFT 1a and AFT 1b only
            msk = REACH_short_term_fish['scenario'].isin(['AFT_1a', 'AFT_1b']) & REACH_short_term_fish['exclusion reasons'].isnull()
            tox_data = REACH_short_term_fish.loc[msk].copy()


        elif endpoint == 'chronic':
            # read the acute fish toxicity data
            REACH_long_term_fish = pd.read_excel(fr'data/fish_chronic/processed/REACH_long_term_fish_measurement.xlsx')

            # keep scenarios FLS 1 and FLS 2 only, no exclusion reasons
            msk = REACH_long_term_fish['scenario'].isin(['FLS_1a', 'FLS_1b']) & REACH_long_term_fish['exclusion reasons'].isnull()
            tox_data = REACH_long_term_fish.loc[msk].copy()


        # add in the standardised SMILES and calculate its molecular weight
        tox_data = tox_data.merge(smiles_set, how='inner', left_on='smiles', right_on='smiles')
        tox_data['molecular weight (standardised)'] = tox_data['smiles (standardised)'].apply(lambda smiles: Descriptors.MolWt(Chem.MolFromSmiles(smiles)))

        # aggregate the data to keep the most conservative effect concentration for each standardised SMILES
        tox_data = (tox_data.groupby(['smiles (standardised)', 'molecular weight (standardised)']).agg({'effect concentration (mol/L, lower bound)': lambda x: x.min(),
                                                                                                        'CAS number': lambda x: ', '.join(x.dropna().drop_duplicates().to_list()) if len(x.dropna())>0 else '-',
                                                                                                        'scenario': lambda x: ', '.join(x.dropna().drop_duplicates().to_list())})
                    .rename(columns={'effect concentration (mol/L, lower bound)': 'effect concentration (mol/L)'}).reset_index()
                    )
        tox_data['effect concentration (mg/L)'] = tox_data['effect concentration (mol/L)']*tox_data['molecular weight (standardised)']*1.e3
        log.info(f"Number of unique structures with {endpoint} fish toxicity data: {tox_data.shape[0]}")

        #
        # REACH_short_term_fish = pd.read_excel(fr'output/{iteration}/data/REACH/short_term_fish/REACH_short_term_fish_study.xlsx')
        # msk = (REACH_short_term_fish['quality score'] == 1) & (REACH_short_term_fish['effect concentration (mol/L, lower bound)'] >= 1.e-9)
        # tox_data = REACH_short_term_fish.loc[msk]
        # # .. keep one tox value for each smiles (standardised)
        # tox_data = tox_data.groupby('smiles (standardised)')[['effect concentration (mol/L, lower bound)', 'effect concentration (mg/L, lower bound)', 'CAS number']].agg({'effect concentration (mol/L, lower bound)': 'min',
        #                                                                                                                        'effect concentration (mg/L, lower bound)': 'min',
        #                                                                                                                        'CAS number': lambda s: ', '.join(s.dropna().drop_duplicates().to_list()) if len(s.dropna())>0 else '-'}).reset_index()
        # .. map toxicity bands
        if toxicity_band_basis == 'molar':
            # Toxicity Band	LC₅₀ (mol/L)
            # Very toxic	< 1×10⁻⁵ mol/L
            # Toxic	1×10⁻⁵ – 1×10⁻⁴ mol/L
            # Harmful	1×10⁻⁴ – 1×10⁻³ mol/L
            # Practically non-toxic	> 1×10⁻³ mol/L
            tox_mapping = {
                '< 1 µM': [-np.inf, 1e-6],
                '1 - 10 µM': [1e-6, 1e-5],
                '10 - 100 µM': [1e-5, 1e-4],
                '100 - 1000 µM': [1e-4, 1e-3],
                '> 1 mM': [1e-3, np.inf]
            }
            tox_data['toxicity band'] = pd.cut(tox_data['effect concentration (mol/L)'], bins=[-np.inf] + [v[1] for v in tox_mapping.values()], labels=list(tox_mapping.keys()), right=False)
        elif toxicity_band_basis == 'mass' and endpoint == 'acute':
            tox_mapping = {
                '<0.1 mg/L': [-np.inf, 0.1],
                '0.1 - 1 mg/L': [0.1, 1.],
                '1 - 10 mg/L': [1., 10.],
                '10 - 100 mg/L': [10., 100.],
                '>100 mg/L': [100., np.inf]
            }
            tox_data['toxicity band'] = pd.cut(tox_data['effect concentration (mg/L)'], bins=[-np.inf] + [v[1] for v in tox_mapping.values()], labels=list(tox_mapping.keys()), right=False)
            log.info(f'toxicity bands (acute, mass) {pd.cut(tox_data['effect concentration (mg/L)'], bins =[-np.inf, 0.0001, 0.001, 0.01, 0.1, 1., 10., 100., np.inf], right=False).value_counts().sort_index().to_json()}')
        elif toxicity_band_basis == 'mass' and endpoint == 'chronic':
            tox_mapping = {
                '<0.01 mg/L': [-np.inf, 0.01],
                '0.01 - 0.1 mg/L': [0.01, 0.1],
                '0.1 - 1 mg/L': [0.1, 1.],
                '1 - 10 mg/L': [1., 10.],
                '>10 mg/L': [10., np.inf]
            }
            tox_data['toxicity band'] = pd.cut(tox_data['effect concentration (mg/L)'], bins=[-np.inf] + [v[1] for v in tox_mapping.values()], labels=list(tox_mapping.keys()), right=False)
            log.info(f'toxicity bands (chronic, mass) {pd.cut(tox_data['effect concentration (mg/L)'], bins =[-np.inf, 0.0001, 0.001, 0.01, 0.1, 1., 10., 100., np.inf], right=False).value_counts().sort_index().to_json()}')

        else:
            raise ValueError(f"Unknown toxicity band basis: {toxicity_band_basis}. Use 'molar' or 'mass'.")

        # convert the SMILES to RDKit molecules
        def smiles_to_mol(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                return mol
            except Exception as e:
                log.error(f"Error converting SMILES to molecule: {e}")
                return None
        tox_data['mol'] = tox_data['smiles (standardised)'].apply(smiles_to_mol)


        # compute the Morgan fingerprints
        radius = 2
        nBits = 2048
        tox_data['fingerprint'] = tox_data['mol'].apply(compute_morgan_fingerprint, radius=radius, nBits=nBits)

        # order the data by toxicity band
        # .. make toxicity band a categorical variable with the correct order
        tox_data['toxicity band'] = pd.Categorical(tox_data['toxicity band'], categories=list(tox_mapping.keys()), ordered=True)
        tox_data = tox_data.sort_values(by='toxicity band', ascending=True).reset_index(drop=True)

        # tox_data = tox_data[:30]


        # compute the Tanimoto distance matrix
        distance_matrix_lower_triang = GetTanimotoDistMat(tox_data['fingerprint'].to_list())
        # .. convert lower triangle to full square form
        # distance_matrix = squareform(distance_matrix_lower_triang)
        # distance_matrix_upper_triang = distance_matrix[np.triu_indices(len(tox_data), k=1)]
        distance_matrix = np.zeros((len(tox_data), len(tox_data)))
        ind = np.tril_indices(len(tox_data), -1)
        distance_matrix[ind] = distance_matrix_lower_triang
        distance_matrix = distance_matrix + distance_matrix.T
        distance_matrix_upper_triang = distance_matrix[np.triu_indices(len(tox_data), k=1)]


        # create the radial plot
        labels = tox_data['CAS number'].to_list()
        toxicity_bands = tox_data['toxicity band'].to_list()
        # Compute and plot the dendrogram, optimal_ordering ensures that distance between successive leaves is minimal
        Y = hierarchy.linkage(distance_matrix_upper_triang, method='ward', optimal_ordering=True)
        Z2 = hierarchy.dendrogram(Y, labels=labels, no_plot=True)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.subplots()
        colormap = dict(zip(tox_mapping.keys(), [
                                             (128/255, 0, 0, 1),
                                             (255/255, 0, 0, 1),
                                             (255/255, 128/255, 0, 1),
                                             (153/255, 153/255, 102/255, 1),
                                             (51/255, 204/255, 51/255, 1)]))
        colorlabels = {'toxicity band': [colormap[toxicity_band] for toxicity_band in toxicity_bands]}
        colorlabels_legend = {'toxicity band':{"colors":colormap.values(),
                                           "labels": colormap.keys()}}
        # .. create a colormap with only black
        black_to_black = LinearSegmentedColormap.from_list("black_to_black", ['black', 'black'], N=256)
        if endpoint == 'acute':
            fontsize = 1
        else:
            fontsize = 2.5
        rd = radialTreee(Z2, ax=ax, colorlabels=colorlabels, colorlabels_legend=None, pallete=black_to_black, fontsize=fontsize)
        # fig.tight_layout()
        from matplotlib.lines import Line2D
        # .. make the default rectangle used as legend invisible
        rec = plt.getp(fig, 'children')[0]
        plt.setp(rec, visible=False)
        # .. define custom legend entries
        custom_lines = [
            Line2D([0], [0], color=list(colormap.values())[0], lw=4),
            Line2D([0], [0], color=list(colormap.values())[1], lw=4),
            Line2D([0], [0], color=list(colormap.values())[2], lw=4),
            Line2D([0], [0], color=list(colormap.values())[3], lw=4),
            Line2D([0], [0], color=list(colormap.values())[4], lw=4),
        ]
        legend_labels = list(colormap.keys())
        fig.legend(
            handles=custom_lines,
            labels=legend_labels,
            loc='center',
            fontsize=8,
            bbox_to_anchor=(0.72, 0.125),  # (x, y) in figure coordinates (0 to 1)
            frameon=False
        )
        fig.savefig(fr'figures\fish_{endpoint}_dendrogram_radial_plot_{toxicity_band_basis}.png', dpi=1200, bbox_inches='tight')


        fig = plt.figure(figsize=(10, 6))
        ax = fig.subplots()# Compute and plot the dendrogram.
        Y = hierarchy.linkage(distance_matrix_upper_triang, method='ward', optimal_ordering=True)
        Z2 = hierarchy.dendrogram(
            Y,
            # no_plot=True,
            ax=ax,
            color_threshold=0.6,
            labels=labels,
            leaf_font_size=fontsize,
            leaf_rotation=90,
        )
        # set line width to 0.5 for all dendrogram branches
        for icoord, dcoord, color in zip(Z2['icoord'], Z2['dcoord'], Z2['color_list']):
            ax.plot(icoord, dcoord, color, linewidth=0.25)
        fig.savefig(fr'figures\fish_{endpoint}_dendrogram_horizontal_plot_{toxicity_band_basis}.png', dpi=1200, bbox_inches='tight')

