import pandas as pd
from helpers import interact_with_fs, validate_parameters
from Interfaces.algorithm import Algorithm
from sklearn.cluster import KMeans


class Clusterer(Algorithm):
    """The Clusterer is responsible for clustering the rows of a prediction matrix.
    This is done upon fit of BanditMF()

    Attributes
    ----------
    _clustered_matrix: pd.DataFrame
        The clustered matrix, created upon fit. size num_clusters x num_items

    DEFAULT_NUM_CLUSTERS: int
        Default value for num_clusters parameter

    Parameters
    ----------
    num_clusters: int
        The number of clusters to create
    """

    _clustered_matrix: pd.DataFrame


    # PARAMETERS
    num_clusters: int
    DEFAULT_NUM_CLUSTERS = 3

    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters

    @validate_parameters
    def fit(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        For now, use KMeans clustering. In the future: experiment with different
        possibilities
        Parameters
        ----------
        matrix
            The user x item prediction matrix to cluster. Result of MFU
        Returns
        -------
        pd.DataFrame
            self._clustered_matrix

        """
        self._kmeans(matrix)
        return self._clustered_matrix

    @interact_with_fs([(f"clustered_matrix", "_clustered_matrix")])
    def _kmeans(self, matrix: pd.DataFrame) -> None:
        """
        Run Kmeans clustering on a user-item prediction matrix
        Parameters
        ----------
        matrix: pd.DataFrame
            The matrix to cluster
        """
        clusters = KMeans(
            n_clusters=self.num_clusters,
            n_init=20,
            verbose=False,
        ).fit(matrix)
        matrix["cluster"] = clusters.labels_
        # Group on cluster, compute average. Result: num_clusters x num_items matrix
        clustered_matrix = matrix.groupby("cluster").mean()
        self._clustered_matrix = clustered_matrix
