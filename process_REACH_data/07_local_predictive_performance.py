import logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/07_local_predictive_performance.log', level_stream=logging.INFO)

import matplotlib
matplotlib.use('tkagg')  # use TkAgg backend for interactive plotting
%matplotlib


from rdkit import Chem, DataStructs
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.sparse import csr_matrix

from cheminformatics.fingerprint_descriptor_distances import compute_morgan_fingerprint

import pandas as pd

import numpy.linalg as la

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import umap
import math
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

import textwrap

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

# read the original and standardised smiles
structures = pd.read_excel(rf'data\structures\smiles.xlsx', sheet_name='smiles (source)')
structures = structures.groupby('smiles (standardised)', dropna=True)['CAS number'].agg('first').reset_index()
smiles_set = structures['smiles (standardised)'].dropna().reset_index(drop=True).to_list()


# prepare molecules and fingerprints
def mol_from_smiles(s):
    m = Chem.MolFromSmiles(s)
    return m
mols = [mol for s in smiles_set if (mol := mol_from_smiles(s)) is not None]
nmol = len(mols)


# compute Morgan fingerprints
radius = 2  # Morgan radius
nBits = 2048  # Morgan fingerprint size
fps = [compute_morgan_fingerprint(mol, radius=radius, nBits=nBits) for mol in mols]
log.info(f"{len(mols)} valid molecules produced from {len(smiles_set)} smiles")
# build the Tanimoto similarity matrix (dense), both lower and upper triangles are filled
# this is also known as affinity matrix and the diagonal is one and not zero
S = np.eye(nmol, dtype=np.float32)
for i in range(nmol):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
    S[i, i+1:] = sims
    S[i+1:, i] = sims
# we also compute the distance matrix for silhouette score computation later
D = 1.0 - S

# --- Compute normalized Laplacian L = I - D^(-1/2) * S * D^(-1/2)
# (use dense version here; for large nmol, use scipy.sparse.linalg.eigsh instead)
def laplacian_eigs(affinity, k=None):
    affinity = affinity.copy()
    # make the diagonal zero (self-loops)
    np.fill_diagonal(affinity, 0.0)
    n = affinity.shape[0]
    d = affinity.sum(axis=1)
    # convert to 1D array if needed, note that self loop is excluded by setting the diagonal to 0
    d_sqrt_inv = np.diag(1.0 / np.sqrt(d + 1e-12))
    L = np.eye(n) - d_sqrt_inv @ affinity @ d_sqrt_inv
    # this should be equivalent to:
    # from scipy.sparse import csgraph
    # csgraph.laplacian(affinity, normed=True)
    if k is None or k >= n:  # full spectrum
        eigvals = la.eigvalsh(L)  # symmetric -> faster
    else:  # partial spectrum (for large n)
        from scipy.sparse.linalg import eigsh
        L_sparse = csr_matrix(L)
        eigvals, _ = eigsh(L_sparse, k=k, which='SM')
    return np.sort(eigvals)
# get full spectrum
eigvals = laplacian_eigs(S)
print("Sorted Laplacian eigenvalues:")
print(eigvals[:20], "...")  # show first 20

# find the number of clusters from the eigenvalues
# Estimate the 'elbow'/'knee' of the curve using the kneed algorithm.
from kneed import KneeLocator
kn = KneeLocator(range(len(eigvals)), eigvals,
                 curve='concave', direction='increasing',
                 interp_method='interp1d', )
n_clusters = kn.knee
log.info('Estimated number of clusters (knee): %d', n_clusters)

# plot the eigenvalues and the knee
neigen_max = 150  # limit to first 200 for clarity
fig = plt.figure(figsize=(6, 6))
ax = fig.subplots()
# without edge color
ax.scatter(range(1, neigen_max + 1), eigvals[:neigen_max], marker='o', label='Eigenvalues',
        color='black', s=50, edgecolors='None', alpha=0.5)
# keep only the bottom spine
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('$k$', fontsize=10)
ax.set_ylabel(r'$\lambda$', fontsize=10)
# vertical line at the knee
ax.axvline(kn.knee, color='red', linestyle='--', linewidth=0.5, label=f'knee at {kn.knee}')
# add a text annotation for the knee, make it vertical and parallel to the vertical line
ax.text(kn.knee - 1, 0.5, f'knee at $k$={kn.knee}', color='red', fontsize=10, rotation=90, va='center', ha='right')
# separate the x and y axes
ax.spines['bottom'].set_position(('outward', 10))
ax.spines['left'].set_position(('outward', 10))
# draw an arrow at the top right corner of the graph and write under it "eignevalues in a descending order"
ax.annotate('', xy=(130, 0.9), xytext=(150, 0.9),
            arrowprops=dict(arrowstyle='<-', color='black', lw=0.5))
ax.text(80, 0.90, r'$\lambda$ in ascending order', fontsize=10, color='black', va='center', ha='left')
fig.tight_layout()
fig.savefig(rf'figures\clustering\spectral_clustering_eigenvalues.png', dpi=600)



clustering_results = []
for n_clusters in [30, 40, 50, 60, 80, 100, 200, 300, 500]:
    log.info(f'Performing spectral clustering with n_clusters={n_clusters}')
    # spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='cluster_qr',
        n_init=100,
        random_state=42
    )
    # this expect the affinity matrix and hence the diagonal to be one
    labels_spectral = sc.fit_predict(S)
    # compute the silhouette score, this expects the distance matrix
    silhouette_score_spectral = silhouette_score(D, labels_spectral)

    # k-means clustering for comparison
    from sklearn.cluster import KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=5,
        random_state=42
    )
    labels_kmeans = kmeans.fit_predict(np.array([fp.ToList() for fp in fps]))
    # compute the silhouette score, this expects the distance matrix
    silhouette_score_kmeans = silhouette_score(D, labels_kmeans)

    # k-medoids clustering for comparison
    from sklearn_extra.cluster import KMedoids
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        method="alternate", # "pam"
        init="k-medoids++", # "build"
        random_state=42
    )
    # this expects a distance matrix, it is equivalent to using the fingerprints and the metric="jaccard"
    labels_kmedoids = kmedoids.fit_predict(D)
    medoid_indices = kmedoids.medoid_indices_
    # compute the silhouette score, this expects the distance matrix
    silhouette_score_kmedoids = silhouette_score(D, labels_kmedoids)

    # hierarchical clustering for comparison
    from sklearn.cluster import AgglomerativeClustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='single',
    )
    labels_hierarchical = hierarchical.fit_predict(D)
    # compute the silhouette score, this expects the distance matrix
    silhouette_score_hierarchical = silhouette_score(D, labels_hierarchical)


    clustering_result = {
        'number of clusters': n_clusters,
        'silhouette_score_spectral': silhouette_score_spectral,
        'silhouette_score_kmeans': silhouette_score_kmeans,
        'silhouette_score_kmedoids': silhouette_score_kmedoids,
        'silhouette_score_hierarchical': silhouette_score_hierarchical,
        'spectral (labels)': labels_spectral,
        'kmeans (labels)': labels_kmeans,
        'kmedoids (labels)': labels_kmedoids,
        'kmedoids (medoid indices)': medoid_indices,
        'hierarchical (labels)': labels_hierarchical,
    }
    log.info(f'n_clusters={n_clusters}, '
             f'silhouette score (spectral)={silhouette_score_spectral:.3f}, '
             f'silhouette score (kmeans)={silhouette_score_kmeans:.3f}, '
             f'silhouette score (kmedoids)={silhouette_score_kmedoids:.3f}, '
             f'silhouette score (hierarchical)={silhouette_score_hierarchical:.3f}')
    clustering_results.append(clustering_result)
clustering_results = pd.DataFrame(clustering_results)
# boxplot with the cluster sizes for each method and number of clusters. The different number of clusters are shown in different subplots of the same figure
# the boxplots are annotated with the silhuette score
fig = plt.figure(figsize=(8, 8))
axs = fig.subplots(nrows=2, ncols=2)
axs = axs.flatten()
for ax_i, method in enumerate(['spectral (labels)', 'kmeans (labels)', 'kmedoids (labels)', 'hierarchical (labels)']):
    data = []
    for i, n_clusters in enumerate(clustering_results['number of clusters']):
        labels = clustering_results.loc[clustering_results['number of clusters'] == n_clusters, method].values[0]
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = pd.DataFrame({'cluster': unique, 'size': counts})
        cluster_sizes['number of clusters'] = n_clusters
        data.append(cluster_sizes)
    data = pd.concat(data, ignore_index=True)
    sns.boxplot(data=data, x='number of clusters', y='size', ax=axs[ax_i], color='orange', fliersize=2)
    axs[ax_i].set_xlabel('number of clusters', fontsize=10)
    axs[ax_i].set_ylabel('cluster size', fontsize=10)
    axs[ax_i].set_title(f'{method.replace(" (labels)", "")}', fontsize=10)
    # hide the left and top spine and move the other two outwards
    axs[ax_i].spines['top'].set_visible(False)
    axs[ax_i].spines['right'].set_visible(False)
    axs[ax_i].spines['bottom'].set_position(('outward', 10))
    axs[ax_i].spines['left'].set_position(('outward', 10))
    # annotate the median with the median value
    for i, n_clusters in enumerate(clustering_results['number of clusters']):
        median = data.loc[data['number of clusters'] == n_clusters, 'size'].median()
        axs[ax_i].text(i, median, f'{median:.0f}', ha='center', va='bottom', fontsize=6, color='black')
    # add the silhuette score above the boxplots
    # for i, n_clusters in enumerate(clustering_results['number of clusters']):
    #     median = data.loc[data['number of clusters'] == n_clusters, 'size'].median()
    #     silhouette_score_value = clustering_results.loc[clustering_results['number of clusters'] == n_clusters, f'silhouette_score_{method.split(" ")[0]}'].values[0]
    #     axs[ax_i].text(i, -10, f's={silhouette_score_value:.2f}', ha='center', va='top', fontsize=6, color='black')
fig.tight_layout()
fig.savefig(rf'figures\clustering\cluster_size_boxplot.png', dpi=600)









# set the number of clusters
n_clusters = 60
for method in ['spectral', 'kmeans', 'kmedoids']:
    log.info(f'visualising clusters for method {method} with n_clusters={n_clusters}')
    labels = clustering_results.loc[clustering_results['number of clusters'] == n_clusters, method+' (labels)'].values[0]
    # visualise the clusters
    for label in np.unique(labels):
        log.info(f'visualising cluster {label}: {np.sum(labels==label)} structures')
        plt.interactive('off')
        log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

        mols_cluster = [mols[i] for i in range(len(mols)) if labels[i] == label]
        cas_numbers_cluster = [structures.loc[structures['smiles (standardised)'] == smiles_set[i], 'CAS number'].values[0]
                               for i in range(len(mols)) if labels[i] == label]

        n_cols = 8
        n_rows = math.ceil(len(mols_cluster)/n_cols)
        fig = plt.figure(figsize=(8, 8), dpi=600)
        fig.subplots_adjust()
        axs = fig.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw = {'wspace':0.01, 'hspace':0.01})
        h = fig.text(0.1, 0.95, f'cluster {label}', fontdict={'fontsize': 4})
        for i_mol, ax in enumerate(axs.flatten()):
            if i_mol < len(mols_cluster):
                plt.setp(ax, frame_on=False)
                plt.setp(plt.getp(ax, 'xaxis'), visible=False)
                plt.setp(plt.getp(ax, 'yaxis'), visible=False)
                cas_number = cas_numbers_cluster[i_mol]
                mol_title = f"{cas_number}"
                h = ax.text(0.5, 0.1, mol_title, ha="center", va="top", rotation=0, size=2.0, bbox=None, transform=ax.transAxes, zorder=2)
                im = Chem.Draw.MolToImage(mols_cluster[i_mol])
                ax.imshow(im, zorder=1)
            else:
                plt.setp(ax, visible=False)
        Path(rf'figures\clustering\clusters\{method}').mkdir(parents=True, exist_ok=True)
        fpath = Path(rf'figures\clustering\clusters\{method}\cluster_{label}.png')
        fig.savefig(fname = fpath, dpi=600)
        plt.close(fig)
        log.info(f'saved figure {str(fpath)} created')
        plt.interactive('on')
        log.info(f'matplotlib interactivity set to {plt.isinteractive()}')


# visualise the clusters using t-SNE or UMAP
# in both cases we reduce the dimension of the fingerprints to n_components by using truncated SVD
n_components = 250  # number of components for SVD
from sklearn.decomposition import TruncatedSVD
# reduce the dimensionality of the fingerprints
svd = TruncatedSVD(n_components=250, random_state=42)
# fit the SVD model to the fingerprints
svd.fit(np.array([fp.ToList() for fp in fps]))
# transform the fingerprints to the reduced space
fps_reduced = svd.transform(np.array([fp.ToList() for fp in fps]))
# report the percentage of variance explained by the first n_components
log.info(f'Percentage of variance explained by the first {n_components} components: {svd.explained_variance_ratio_.sum():.3f}')


# # compute the silhouette score
# # .. compute the pairwise distances using the precomputed Tanimoto similarity
# D_fps_reduced = pairwise_distances(fps_reduced, metric='euclidean')
# silhouette_avg = silhouette_score(D_fps_reduced, labels, metric='precomputed')
# log.info(f'{n_clusters} clusters: silhouette score is {silhouette_avg:.3f}')
#

# build the t-SNE model to visualise the clusters
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
)
X_2d_tsne = tsne.fit_transform(fps_reduced)

# build the UMAP model to visualise the clusters
from sklearn.metrics import pairwise_distances

reducer = umap.UMAP(random_state=2, n_neighbors=20, min_dist=0.1, n_epochs=500, n_components=2, metric='euclidean')
reducer.fit(fps_reduced)
X_2d_umap = reducer.transform(fps_reduced)

# store the reduced embeddings and labels
reduced_embeddings = (pd.DataFrame(labels, columns=['cluster'], index=smiles_set)
                      .merge(pd.DataFrame(X_2d_tsne, columns=['x1 (tsne)', 'y1 (tsne)'], index=smiles_set),
                             left_index=True, right_index=True)
                      .merge(pd.DataFrame(X_2d_umap, columns=['x1 (umap)', 'y1 (umap)'], index=smiles_set),
                             left_index=True, right_index=True)
                      .reset_index()
                      .rename({'index': 'smiles (standardised)'}, axis='columns')
                      )
reduced_embeddings.to_excel(rf'data\clustering\spectral_clustering_reduced_embeddings.xlsx', index=False)

# visualise the clusters in 2D using t-SNE and UMAP
for method in ['spectral', 'kmeans', 'kmedoids']:
    log.info(f'visualising clusters within chemical space for method {method} with n_clusters={n_clusters}')
    labels = clustering_results.loc[clustering_results['number of clusters'] == n_clusters, method+' (labels)'].values[0]

    for reduction_method, X_2d in zip(['tsne', 'umap'], [X_2d_tsne, X_2d_umap]):
        # plot a scatter plot of the UMAP projection with clusters colored by labels
        fig = plt.figure(figsize=(10, 10))
        nrows = round(math.sqrt(n_clusters))
        ncols = math.ceil(n_clusters / nrows)
        axs = fig.subplots(nrows=nrows, ncols=ncols)
        for i, ax in enumerate(axs.flat):
            if i < n_clusters:
                n_structures = np.sum(labels == i)
                msk = labels == i
                scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgrey',  s=4, alpha=0.7, edgecolors='none', zorder=1)
                scatter = ax.scatter(X_2d[msk, 0], X_2d[msk, 1], c='orange', s=4, alpha=0.5, edgecolors='none', zorder=2)
                ax.text(0.1, 0.2, f'#{i}\n{n_structures} structures', fontsize=5, fontweight='normal', color='black',
                        transform=ax.transAxes, va='center', ha='left', zorder=3)

                # highlight_textprops = [{'fontsize': 6, 'fontweight': 'normal', 'color': 'black'},
                #                        {'fontsize': 4, 'fontweight': 'bold', 'color': 'black'}]
                # ax_text(s=f'<#{i}>\n<{n_structures} structures>', x=0.2, y=0.2, ax=ax, va='top', ha='left',
                #         highlight_textprops=highlight_textprops, transform=ax.transAxes, zorder=3)
            ax.axis('off')  # turn off unused subplots
        fig.tight_layout()
        fig.savefig(rf'figures\clustering\{method}_clustering_{reduction_method}_clusters.png', dpi=600)




# compute the local predictive performance per cluster
model_predictive_performance_files = (list(Path('data/predictions/').glob('predictions_vs_experimental_*acute.xlsx')) +
                                      list(Path('data/predictions/').glob('predictions_vs_experimental_*chronic.xlsx'))
                                      )
for method in ['kmeans', 'kmedoids', 'spectral']:
    log.info(f'Processing model predictive performance for clustering method: {method}')
    labels = clustering_results.loc[clustering_results['number of clusters'] == n_clusters, method+' (labels)'].values[0]
    structures_labels = pd.DataFrame({'smiles (standardised)': smiles_set, 'cluster': labels})

    local_predictive_performances = []
    global_predictive_performaces = []
    for model_predictive_performance_file in model_predictive_performance_files:
        log.info('Processing model predictive performance file: %s', str(model_predictive_performance_file))
        df_perf = (pd.read_excel(model_predictive_performance_file)
                   .merge(structures_labels, left_on='smiles (standardised)', right_on='smiles (standardised)', how='left')
                   )
        df_perf.to_excel(rf'data\clustering\{method}\{Path(model_predictive_performance_file).stem}_with_clusters.xlsx', index=False)
        # check that there is no missing cluster label
        n_missing_labels = df_perf['cluster'].isna().sum()
        assert n_missing_labels == 0, f'There are {n_missing_labels} structures with missing cluster labels in the predictive performance data'

        # in domain predictions, not in the training or validation sets
        for cluster in structures_labels['cluster'].unique():
            log.info('Processing cluster %d', cluster)
            log.info('number of structures in cluster: %d', np.sum(structures_labels['cluster'] == cluster))

            msk_in_domain_not_train_val = (df_perf['AD'] == 'in domain') & (df_perf['training/validation set'] == 'not in training/validation set') & (df_perf['cluster'] == cluster)
            log.info(f'Number of in domain, not in training/validation set structures in cluster {cluster}: {np.sum(msk_in_domain_not_train_val)}')

            # compute the RMSE and MAE for the cluster if there are at least 5 structures
            if np.sum(msk_in_domain_not_train_val) < 1:
                log.info(f'Cluster {cluster}: less than 5 structures in domain and not in training/validation set, skipping performance metrics computation')
                local_predictive_performance = {'file': str(model_predictive_performance_file),
                                                'platform': df_perf['platform'].unique()[0],
                                                'model name': df_perf['model name'].unique()[0],
                                                'cluster': cluster,
                                                'n_structures_in_cluster': np.sum(structures_labels['cluster'] == cluster),
                                                'n_structures_in_domain_not_train_val': np.sum(msk_in_domain_not_train_val),
                                                'study type': df_perf['study type'].unique()[0],
                                                'RMSE': np.nan,
                                                'MAE': np.nan}
            else:
                y = np.log10(df_perf.loc[msk_in_domain_not_train_val, 'prediction'].to_numpy())
                x = np.log10(df_perf.loc[msk_in_domain_not_train_val, 'effect concentration (mg/L)'].to_numpy())
                RMSE = np.sqrt(np.mean((y - x) ** 2))
                MAE = np.mean(np.abs(y - x))
                log.info(f'Cluster {cluster}: RMSE = {RMSE:.3f}, MAE = {MAE:.3f}')
                local_predictive_performance = {'file': str(model_predictive_performance_file),
                                                'platform': df_perf['platform'].unique()[0],
                                                'model name': df_perf['model name'].unique()[0],
                                                'cluster': cluster,
                                                'n_structures_in_cluster': np.sum(structures_labels['cluster'] == cluster),
                                                'n_structures_in_domain_not_train_val': np.sum(msk_in_domain_not_train_val),
                                                'study type': df_perf['study type'].unique()[0],
                                                'RMSE': RMSE,
                                                'MAE': MAE}
            local_predictive_performances.append(local_predictive_performance)

        # overall predictive performance across all clusters
        overall_msk_in_domain_not_train_val = (df_perf['AD'] == 'in domain') & (df_perf['training/validation set'] == 'not in training/validation set')
        y_overall = np.log10(df_perf.loc[overall_msk_in_domain_not_train_val, 'prediction'].to_numpy())
        x_overall = np.log10(df_perf.loc[overall_msk_in_domain_not_train_val
        , 'effect concentration (mg/L)'].to_numpy())
        RMSE_overall = np.sqrt(np.mean((y_overall - x_overall) ** 2))
        MAE_overall = np.mean(np.abs(y_overall - x_overall))
        log.info(f'Overall predictive performance across all clusters: RMSE = {RMSE_overall:.3f}, MAE = {MAE_overall:.3f}')
        global_predictive_performance = {'file': str(model_predictive_performance_file),
                                         'platform': df_perf['platform'].unique()[0],
                                         'model name': df_perf['model name'].unique()[0],
                                         'cluster': 'overall',
                                         'n_structures_in_cluster': len(structures_labels),
                                         'n_structures_in_domain_not_train_val': np.sum(overall_msk_in_domain_not_train_val),
                                         'study type': df_perf['study type'].unique()[0],
                                         'RMSE': RMSE_overall,
                                         'MAE': MAE_overall}
        global_predictive_performaces.append(global_predictive_performance)
    local_predictive_performances = pd.DataFrame(local_predictive_performances).sort_values(by=['file', 'cluster'], ascending=True)
    global_predictive_performances = pd.DataFrame(global_predictive_performaces).sort_values(by=['file'], ascending=True)


    # plot a heatmap of the local predictive performance (RMSE) per cluster for all models
    msk = local_predictive_performances['study type'] == 'acute' # keep only acute studies
    # .. remove ECOSAR multiple classes
    tmp = local_predictive_performances.loc[msk & (~local_predictive_performances['model name'].str.contains('(?i)multiple classes'))]
    tmp = tmp.assign(model_name_platform=local_predictive_performances['platform']+', '+local_predictive_performances['model name'])
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[0.8, 3], hspace=0.025)  # less space between plots
    # .. boxplot (top)
    ax_box = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        x="cluster",
        y="RMSE",
        data=tmp,
        ax=ax_box,
        color="lightgrey",
        width=0.8,  # make the boxes thinner
        medianprops={"color": "black", "linewidth": 1.5}  # black median line
    )
    ax_box.set_xticks([])  # hide x labels for boxplot (will use heatmap labels)
    # yticks every 0.5 using a locator
    ax_box.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax_box.set_xlabel("")
    ax_box.set_ylabel("RMSE", fontsize=8)
    # ax_box.tick_params(length=0)
    # remove top, right, bottom spines, make left spine outward
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)
    ax_box.spines['bottom'].set_visible(False)
    ax_box.spines['left'].set_position(('outward', 10))
    # ax_box.set_ylim(0, 1)  # cap RMSE for consistency with heatmap
    # .. heatmap (bottom)
    ax_heat = fig.add_subplot(gs[1, 0])
    rmse_pivot = tmp.pivot(index="model_name_platform", columns="cluster", values="RMSE")
    # .. annotations
    annot_pivot = tmp.pivot(index="model_name_platform", columns="cluster", values="n_structures_in_domain_not_train_val")
    sns.heatmap(
        rmse_pivot,
        ax=ax_heat,
        annot=annot_pivot.astype("Int64"),  # integer annotations
        fmt="d",
        cmap="RdYlGn_r",  # green (low) → red (high)
        vmin=0,  # RMSE = 0 is pure green
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        cbar_kws={
            "label": "RMSE (capped at 1)",
            "orientation": "horizontal",
            "fraction": 0.05, # width of colorbar relative to figure
            "pad": 0.1, # space between heatmap and colorbar
            "shrink": 0.5 # half height
          }
    )
    # .. show all x and y labels
    ax_heat.set_xticks(np.array(range(rmse_pivot.shape[1]))+0.5)
    ax_heat.set_xticklabels(rmse_pivot.columns, fontsize=8, rotation=90)
    ax_heat.set_yticks(np.array(range(rmse_pivot.shape[0]))+0.5)
    # break the y labels to two lines if they are too long, using textwrap, with a maximum width of 30 characters
    yticklabels = rmse_pivot.index.tolist()
    yticklabels = [textwrap.fill(label, width=40) for label in yticklabels]
    ax_heat.set_yticklabels(yticklabels, fontsize=8)
    # .. remove tick marks
    ax_heat.tick_params(length=0)
    ax_heat.set_xlabel("cluster")
    ax_heat.set_ylabel('')
    # Adjust spacing manually instead of tight_layout()
    fig.subplots_adjust(
        top=1,  # space for boxplot
        bottom=0.05,
        left=0.2,
        right=0.975,
        hspace=0.025  # vertical space between boxplot and heatmap
    )
    plt.show()
    fig.savefig(rf'figures\clustering\rmse\{method}_local_predictive_performance_heatmap_acute.png', dpi=600)


    # plot the local predictive performance
    for model_predictive_performance_file in model_predictive_performance_files:
        df = local_predictive_performances.loc[local_predictive_performances['file'] == str(model_predictive_performance_file)]
        model_name = (df['platform'] + ', ' + df['model name']).values[0]
        global_predictive_performance = global_predictive_performances.loc[global_predictive_performances['file'] == str(model_predictive_performance_file)]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df))
        bar_width = 0.7
        # --- Bar plots (left axis) ---
        ax1.bar(
            x,
            df["n_structures_in_cluster"],
            width=bar_width,
            label="number of structures in cluster",
            color='lightgrey',
            alpha=0.5,
            zorder=1
        )
        ax1.bar(
            x,
            df["n_structures_in_domain_not_train_val"],
            width=bar_width,
            label="in AD, not in training/validation set",
            color='orange',
            alpha=0.5,
            zorder=1
        )
        ax1.set_xlabel("cluster")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["cluster"])
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.set_ylabel("RMSE")
        # plot all clusters
        rmse_mask = ~df["RMSE"].isna() & (df["n_structures_in_domain_not_train_val"] >= 1)
        ax2.plot(
            x[rmse_mask],
            df.loc[rmse_mask, "RMSE"],
            color="black",
            linewidth=1,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            zorder=2,
            label="RMSE (n<5)"
        )
        # plot clusters with n_structures_in_domain_not_train_val >= 5
        rmse_mask = ~df["RMSE"].isna() & (df["n_structures_in_domain_not_train_val"] >= 5)
        ax2.scatter(
            x[rmse_mask],
            df.loc[rmse_mask, "RMSE"],
            marker="o",
            s=6,
            facecolor="black",
            edgecolor="black",
            zorder=2,
            label="RMSE (n≥5)"
        )
        # make the left y axis logarithmic
        ax1.set_yscale('log')
        # hide top spine and offet the rest
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_position(('outward', 10))
        ax1.spines['bottom'].set_position(('outward', 10))
        ax2.spines['right'].set_position(('outward', 10))
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        # make the x-tick labels rotated 90 degrees, and smaller font size
        plt.setp(ax1.get_xticklabels(), rotation=90, fontsize=8)
        ax1.set_ylim(0.8, 1000)
        # increase the second y axis limit from the current maximum by 0.2, the minimum should be as now
        ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1] + 0.2)
        # make the y-axis tick labels be at 1, 10, 100, 1000
        ax1.set_yticks([1, 10, 100, 1000])
        ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_ylabel('number of structures')
        ax1.set_xlabel('cluster')
        ax1.set_title(model_name)
        # draw a horizonal line for the overall RMSE
        ax2.axhline(global_predictive_performance["RMSE"].values[0], color='black', linestyle='--', linewidth=0.5, label='overall RMSE')
        # annotate the overall RMSE line
        ax2.text(len(df)-1, global_predictive_performance["RMSE"].values[0] + 0.05, f'overall RMSE: {global_predictive_performance["RMSE"].values[0]:.3f}', fontsize=8, color='black', ha='right', va='bottom')
        # add legends for the two y-axes
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        fig.canvas.draw()
        # loop over minor ticks < 1 and make their length 0 after the figure is drawn because of the logarithmic scale
        for tick in ax1.yaxis.get_minor_ticks():
            yval = tick.get_loc()
            if yval < 1:
                tick.tick1line.set_markersize(0)
                tick.tick2line.set_markersize(0)
        Path(rf'figures\clustering\rmse\{method}').mkdir(parents=True, exist_ok=True)
        fig.savefig(rf'figures\clustering\rmse\{method}\{Path(model_predictive_performance_file).stem}.png', dpi=600)
        plt.close(fig)


    # draw a box plot of the local RMSEs for each acute model
    msk = local_predictive_performances['RMSE'].notnull() & (local_predictive_performances['study type'] == 'acute')
    df = local_predictive_performances.loc[msk]
    df['model name'] = df['platform'] + ', ' + df['model name']
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    sns.boxplot(
        data=df,
        ax=ax,
        x="RMSE",
        y="model name",
        orient="h",
        showmeans=True,
        boxprops=dict(facecolor="orange", edgecolor="orange"),
        medianprops=dict(color="black", linewidth=2),
        meanprops=dict(
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6
        ),
        whiskerprops=dict(color="orange"),
        capprops=dict(color="orange")
    )
    ax.set_xlabel("RMSE")
    # custom legend
    legend_elements = [
        Patch(facecolor="orange", edgecolor="orange", label="IQR"),
        Line2D([0], [0], color="black", lw=2, label="cluster median RMSE"),
        Line2D([0], [0], marker="o", color="black", linestyle="-", markersize=6, label="cluster mean RMSE")
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True
    )
    # remove all spines except the bottom that is moved outward
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_ylabel('')
    fig.tight_layout()
    # remove y tick marks
    ax.yaxis.set_ticks_position('none')
    fig.savefig(rf'figures\clustering\rmse\{method}\boxplot_acute_models_local_rmse.png', dpi=600)
    plt.close(fig)