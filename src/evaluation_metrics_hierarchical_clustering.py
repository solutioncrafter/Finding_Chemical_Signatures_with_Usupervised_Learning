import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def calculate_wcss(X, labels):
    """
    Calculate Within-Cluster Sum of Squares (WCSS) for the given data
    and labels.

    Parameters:
    X (numpy.ndarray): Standardized data points.
    labels (numpy.ndarray): Cluster labels for the data points.

    Returns:
    float: The WCSS value.
    """
    wcss = 0
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += ((cluster_points - centroid) ** 2).sum()
    return wcss


def compute_metrics(dataframe, cluster_number=range(2, 10)):
    """
    Compute clustering metrics for different numbers of clusters.

    Parameters:
    dataframe (pandas.DataFrame): The input data.
    cluster_number (range): The range of cluster numbers to evaluate.

    Returns:
    tuple: Containing lists of WCSS,
                silhouette scores,
                Calinski-Harabasz scores,
                and Davies-Bouldin scores.
    """
    knn_graph = kneighbors_graph(dataframe, 10, include_self=False)

    wcss = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []

    scaler = StandardScaler()
    dataframe_standardized = scaler.fit_transform(dataframe)

    for i in cluster_number:
        clustering = AgglomerativeClustering(
            n_clusters=i, metric='euclidean', linkage='ward',
            connectivity=knn_graph)
        clustering.fit_predict(dataframe_standardized)

        try:
            wcss.append(calculate_wcss(dataframe_standardized,
                                       clustering.labels_))
        except Exception:
            print(f"WCSS score omitted for point {i}")
            wcss.append(np.nan)

        try:
            davies_bouldin = davies_bouldin_score(dataframe_standardized,
                                                  clustering.labels_)
            davies_bouldin_scores.append(davies_bouldin)
        except Exception:
            print(f"Davies-Bouldin score omitted for point {i}")
            davies_bouldin_scores.append(np.nan)

        try:
            silhouette_avg = silhouette_score(dataframe_standardized,
                                              clustering.labels_)
            silhouette_scores.append(silhouette_avg)
        except Exception:
            print(f"Silhouette score omitted for point {i}")
            silhouette_scores.append(np.nan)

        try:
            calinski_harabasz = calinski_harabasz_score(dataframe_standardized,
                                                        clustering.labels_)
            calinski_harabasz_scores.append(calinski_harabasz)
        except Exception:
            print(f"Calinski-Harabasz score omitted for point {i}")
            calinski_harabasz_scores.append(np.nan)

    return (wcss, silhouette_scores, calinski_harabasz_scores,
            davies_bouldin_scores)


def plot_hierarchical_clustering_evaluation_curves(
        dataframe, cluster_number_evaluation, cluster_number=range(2, 10)):
    """
    Plot evaluation curves for hierarchical clustering metrics.

    Parameters:
    dataframe (pandas.DataFrame): The input data.
    cluster_number_evaluation (int): The cluster number at which to draw a line
    cluster_number (range): The range of cluster numbers to evaluate.
    """
    (wcss, silhouette_scores, calinski_harabasz_scores,
     davies_bouldin_scores) = compute_metrics(dataframe, cluster_number)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 2], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cluster_number, wcss, label='WCSS', color='red', marker='x')
    ax1.set_ylabel('WCSS', color='red')
    ax1.axvline(x=cluster_number_evaluation, color='black', linestyle='--')

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(cluster_number, silhouette_scores, label='Silhouette',
             color='blue', marker='o')
    ax2.set_ylabel('Silhouette\nScore', color='blue')
    ax2.axvline(x=cluster_number_evaluation, color='black', linestyle='--')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(cluster_number, calinski_harabasz_scores,
             label='Calinski-Harabasz',
             color='green', marker='s')
    ax3.set_ylabel('Calinski-Harabasz\nScore', color='green')
    ax3.axvline(x=cluster_number_evaluation, color='black', linestyle='--')
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(cluster_number, davies_bouldin_scores, label='Davies-Bouldin',
             color='darkviolet', marker='P')
    ax4.set_ylabel('Davies-Bouldin\nScore', color='darkviolet')
    ax4.axvline(x=cluster_number_evaluation, color='black', linestyle='--')
    ax4.set_xlabel('Number of Clusters')
    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.show()
