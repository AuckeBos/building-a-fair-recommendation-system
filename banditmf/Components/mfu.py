import numpy as np
import pandas as pd
from Components.Rewarders.base_rewarder import BaseRewarder
from helpers import (
    frame_to_matrix,
    interact_with_fs,
    matrix_to_frame,
    validate_parameters,
)
from Interfaces.algorithm import Algorithm
from Interfaces.predictor import Predictor
from Interfaces.rewardable import Rewardable
from lenskit.algorithms.als import BiasedMF


class MatrixFactorizationUnit(Rewardable, Predictor, Algorithm):
    """MatrixFactorizationUnit is used to rank items for non-cold users.
    Upon fit, it performs matrix factorization and saves the user x item prediction
    matrix. Upon predict, the row for the user is returned, which contains a
    prediction for each rating. Upon reward, the cumulative regret is increased. If
    it exceeds a limit, the MFU refits. Refitting is the same as the initial fit.

     Attributes
     ----------
     predictions_matrix : pd.DataFrame
        The predictions matrix of n_user rows and n_items columns. It is
        computed via _factorize(), using the BiasedMF matrix factorization
        algorithm. For non-cold users, BanditMF uses this matrix to rank items

    _original_data: pd.DataFrame
        The matrix of original ratings, eg the matrix on which the current
        predictions are based.

    _cumulative_regret: float
        Set to 0 upon fit(). At each reward(), compute diff between the prediction
        and the actual rating for that user,item. This diff is the regret. The
        cumulative regret is used to decide whether we must_refit()

    DEFAULT_REFIT_AT: float
        Default value for refit_at parameter.

    Parameters
    ----------
    refit_at: float
        Threshold that defines when the MFU must be refit. Should be proportional to
        Rewarder.MAX_RATING. If the cumulative regret exceeds this limit, the MFU refits
    """

    predictions_matrix: pd.DataFrame

    _original_data: pd.DataFrame
    _cumulative_regret: float

    # PARAMETERS
    refit_at: float
    DEFAULT_REFIT_AT = 10 * BaseRewarder.MAX_RATING

    def __init__(self, refit_at: float):
        self.refit_at = refit_at

    @validate_parameters
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "MatrixFactorizationUnit":
        """
        Fit the matrix factorization, eg
        - Reset the rewards and regrets
        - Run matrix factorization, save predictions

        Parameters
        ----------
        X: pd.DataFrame
            The matrix with known ratings, user x item
        y: Not used, added for sklearn compatibility
        Returns
        -------
            self
        """
        self._original_data = X
        self._cumulative_regret = 0.0
        self._factorize()
        return self

    def predict(self, user: int) -> pd.Series:
        """
        Predicting the ratings for a user means simply selecting the row in the
        predictions matrix
        Parameters
        ----------
        user: int
            The user to predict ratings for
        Returns
        -------
        pd.Series:
            List of predicted ratings, maps item id to predicted rating

        """
        predicted_ratings = self.predictions_matrix.loc[user]
        return predicted_ratings

    def reward(self, user: int, item: int, rating: float) -> None:
        """
        Save a reward. Eg
        - Update cumulative regret with the difference between the actual and the
        predicted rating

        Parameters
        ----------
        user: int
            The user to reward for
        item: int
            The item that was recommended
        rating: float
            The actual rating for the user,item tuple
        """
        prediction = self.predictions_matrix.loc[user, item]
        self._cumulative_regret += abs(rating - prediction)

    def refit(self, matrix: pd.DataFrame) -> None:
        """
        Refit using the newest rating matrix, same as fit()
        """
        self.fit(matrix)

    def must_refit(self) -> bool:
        """
        Return true if the mfu must refit. Refitting means rerunning the matrix
        factorization based on the most up-to-date data. This data is the original
        ratings plus all the ratings we have received during the reward() calls.

        We must refit if the cumulative regret is greater than refit_at.
        Returns
        -------
            True if must refit, else false

        """
        return self._cumulative_regret > self.refit_at

    def user_exists(self, user: int) -> bool:
        """
        Check whether the user exists in the predictions matrix
        Parameters
        ----------
        user: int
            The user to check
        Returns
        -------
        bool
            True if the user exists
        """
        return user in self.predictions_matrix.index

    @interact_with_fs(
        [
            ("factorization", "predictions_matrix"),
        ]
    )
    def _factorize(self) -> None:
        """
        Factorize the original sparse user x item ratings matrix into matrices
        - p: User x Feature matrix
        - q: Item x Feature matrix
        Using BiasedMF

        Compute R_hat (predictions) = p x q.T
        """
        algorithm = BiasedMF(50)
        algorithm.fit(matrix_to_frame(self._original_data))

        p = algorithm.user_features_
        q = algorithm.item_features_
        unbiased_prediction_matrix = pd.DataFrame(
            np.dot(p, q.T),  # type: ignore
            index=self._original_data.index,
            columns=self._original_data.columns,
        )
        # Convert to frame such that we can compute the biased predictions, convert back
        unbiased_predictions = matrix_to_frame(unbiased_prediction_matrix)
        predictions = algorithm.bias.inverse_transform(unbiased_predictions)

        predictions_matrix = frame_to_matrix(predictions)

        self.predictions_matrix = predictions_matrix
