import pandas as pd
from Baselines.base_baseline import BaseBaseline
from helpers import X_y_to_matrix
from numpy import ndarray
from sklearn.utils import check_X_y


class PopularityBasedRecommender(BaseBaseline):
    """A baseline recommender that recommends based on popularity:

    fit() requires the training matrix. Upon predict(), the recommender predicts the
    most popular item, according to global average rating, that was not found in
    training nor history

    Attributes
    ----------
    _global_averages: pd.Series
        List of global average for each item. Length n_items. Sorted on descending
        rating
    """

    _global_averages: pd.DataFrame

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "PopularityBasedRecommender":
        """
        Fit, eg set global averages
        Parameters
        ----------
        X: pd.DataFrame
            2 column frame: user,item
        y: pd.DataFrame
            rating column

        Returns
        -------
            self

        """
        check_X_y(X, y, ensure_2d=True, ensure_min_features=2, accept_sparse=False)
        matrix = X_y_to_matrix(X, y)
        avg_ratings = matrix.mean(axis=0)
        avg_ratings.sort_values(ascending=False, inplace=True)
        self._global_averages = avg_ratings
        return self

    def recommend(self, n: int, candidates: ndarray) -> pd.Series:
        """
        Recommend n items for user a user. The user does not need to be provided,
        since predictions are not personal
        Parameters
        ----------
        n: int
            The number of items to recommend
        candidates: np.array
            The list of candidates to select from
        Returns
        -------
        pd.Series
            List of recommendations, sorted descending on prediction value, length n.
             The predicted values are actually the rank in the list, to ensure the
             ranking order

        """
        predictions = self._global_averages
        predictions = predictions[predictions.index.isin(candidates)]
        predictions = predictions[:n]
        predictions[:] = list(range(len(predictions), 0, -1))

        return predictions
