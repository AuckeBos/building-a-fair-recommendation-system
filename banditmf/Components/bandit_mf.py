from logging import ERROR, getLogger
from typing import Dict, Optional, Union

import pandas as pd
from Components.casino import Casino
from Components.clusterer import Clusterer
from Components.mfu import MatrixFactorizationUnit
from Components.Rankers.base_ranker import BaseRanker
from Components.Rankers.default_ranker import DefaultRanker
from helpers import X_y_to_matrix, validate_parameters
from Interfaces.algorithm import Algorithm
from Interfaces.rewardable import Rewardable
from numpy import ndarray
from sklearn.utils import check_X_y


class BanditMf(Rewardable, Algorithm):
    """BanditMF Recommendation system.

    BanditMF combines MatrixFactorization with MAB
    Please refer to the thesis for a detailed explanation of attributes and parameters

    Note that the attributes under the 'Parameters' section are forwarded to the other
    components via create_components. These parameters can be provided in the
    __init__ of the BanditMF algorithm

    This RS uses MatrixFactorization when a non-cold user requests a recommendation,
    the Casino (set of MABs) is used  for cold users. Both the MFU (
    MatrixFactorizationUnit) and the Casino are predictors; they should be able to
    providean ordered list of recommendations. The Ranker ranks this list, subject to
    fairness constraints.

    Attributes
    ----------
    _mfu: MatrixFactorizationUnit
        Created on create_components, fit on fit(). Used to predict items to non-cold
            users
    _casino: Casino
        Created on create_components, fit on fit(). Used to predict items to cold users
    _clusterer: Clusterer
        Created on create_components, fit on fit(). Used in fit() to cluster rows in
        the prediction matrix of the mfu. The clusters are arms for the MABs in the
        casino, one MAB per user, one arm per cluster

    _ranker: Ranker. Default is DefaultRanker, can be overridden through property
    setter. The ranker is used after rating prediction to rerank items based on some
    algorithm.

    _recommender_types: Dict[int, str]
        Maps each user to either Casino or MFU. It indicates which recommender was
        used for the latest recommendation to that user. If no recommendation has yet
        been done for the user, the user doesn't exist in this dict. Used in reward().

    _data: pd.DataFrame
        Complete matrix of all known ratings in the train set, eg those in X,
        y of fit() + and those retrieved via reward(). In format user x item matrix.
        Initialized in fit(), updated on reward()

    Parameters
    ----------
    initial_cu_threshold: int
        Param for the Casino. Refer to the its description
    certainty_window: int
        Param for the Casino. Refer to the its description
    refit_at: float
        Param for the MFU. Refer to the its description
    num_clusters: int
        Param for the Clusterer. Refer to the its description
    """

    # ATTRIBUTES
    _mfu: MatrixFactorizationUnit
    _casino: Casino
    _clusterer: Clusterer
    _ranker: BaseRanker
    _data: pd.DataFrame
    _recommender_types: Dict[int, str]

    # PARAMETERS
    initial_cu_threshold: Optional[int] = None
    certainty_window: Optional[int] = None
    refit_at: Optional[float] = None
    num_clusters: Optional[int] = None

    def __init__(
        self,
        initial_cu_threshold: Optional[int] = None,
        certainty_window: Optional[int] = None,
        refit_at: Optional[float] = None,
        num_clusters: Optional[int] = None,
    ):
        """
        Set the recommenders HyperParameters
        """
        self.initial_cu_threshold = initial_cu_threshold
        self.certainty_window = certainty_window
        self.refit_at = refit_at
        self.num_clusters = num_clusters

    @validate_parameters
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "BanditMf":
        """
        Fit BanditMF:
        - Create components
        - Set Sklearn compatibility parameters
        - Save data
        - fit components

        Parameters
        ----------
        X: pd.DataFrame
            Shape (n_ratings, n_features), Features will be stripped
            down to the first 2: must be user,item columns
        y: pd.Series
            Shape (n_ratings,) containing the ratings of the user,item tuples

        Returns
        -------
            self

        """
        # Disable warning logs
        getLogger("lenskit").setLevel(ERROR)
        self._create_components()
        # Ensure sklearn compatibility
        check_X_y(X, y, ensure_2d=True, ensure_min_features=2, accept_sparse=False)
        self.n_features_in_ = 2
        self._recommender_types = {}

        matrix = X_y_to_matrix(X, y)
        self._data = matrix
        # Fit MF and Casino
        self._mfu.fit(matrix)
        clusters = self._clusterer.fit(self._mfu.predictions_matrix)
        self._casino.fit(clusters, matrix=matrix)
        return self

    def recommend(
        self,
        user: int,
        n: int,
        candidates: ndarray = None,
        enforce_is_cold: bool = None,
    ) -> pd.Series:
        """
        Recommend n items to a user:
        - Select the recommender that is required for this user
        - Save the recommender in self_recommender_types, needed in reward()
        - Predict using the recommender
        - Drop the items that are no candidates
        - Rerank the items, return them
        Parameters
        ----------
        user: int
            The user to recommend for
        n: int
            The number of items to recommend
        candidates: np.array
            The list of candidates to select from. If not provided, all items are
            candidates
        enforce_is_cold: bool
            If provided, use this value for is_cold, hence forcing a specific
            recommender. If not provided, use recommender based on is_cold(user)
        Returns
        -------
        pd.Series:
            List of recommendations, sorted descending on prediction value, length n

        """
        recommender = self._choose_recommender(user, enforce_is_cold)
        self._recommender_types[user] = recommender.__class__.__name__

        predictions = recommender.predict(user)

        if candidates is not None:
            predictions = predictions[predictions.index.isin(candidates)]
        result = self._ranker.predict(predictions, n)
        return result

    def reward(self, user: int, item: int, rating: float) -> bool:
        """
        Reward BanditMF for a recommendation that was made earlier:
        - Set the new rating in self._data, is used on refit later
        - Select the recommender that was latest used to recommend to this user
        - Reward the recommender
        - If either the casino or the mfu tells us to refit, refit both
        Parameters
        ----------
        user: int
            The user
        item: int
            The item
        rating: float
            The rating in [0, Rewarder.MAX_RATING]
        Returns
        -------
        bool
            True if the Casino and the MFU have been refit, else false

        """
        self._data.loc[user, item] = rating
        predictor = (
            self._casino
            if self._recommender_types[user] == self._casino.__class__.__name__
            else self._mfu
        )
        predictor.reward(user, item, rating)
        must_refit = self._mfu.must_refit() or (
            predictor == self._casino and self._casino.stopping_criteria_met(user)
        )
        if must_refit:
            self._mfu.refit(self._data)
            self._casino.refit()
            return True
        else:
            return False

    def is_cold(self, user: int) -> bool:
        """
        Check whether a user is cold. This is checked through the casino. E.g. if the
        the casino has achieved certainty for the user, the user is non-cold. If the
        casino has never seen the user, the user is cold.

        #A: If the user existed during fit() (is in mfu), but it does not exist in
        the casino, the user was already warm during fit (eg had more known ratings
        than casino.initial_cu_threshold). Thus the user is not cold now. This check
        is added because in this case, the casino would say that the user is cold,
        while it is not. Note that this check would never succeed during an
        evaluation, because evaluations extract a user completely from the training
        set, hence every cold user in the testset occurs in the Casino. However,
        during production mode, this check will succeed for users.
        Parameters
        ----------
        user: int
            The user to check

        Returns
        -------
            True if is cold, else false
        """
        # A
        if self._mfu.user_exists(user) and not self._casino.user_exists(user):
            return False
        return self._casino.is_cold(user)

    @property
    def ranker(self) -> BaseRanker:
        return self._ranker

    @ranker.setter
    def ranker(self, ranker: BaseRanker) -> None:
        self._ranker = ranker

    def _create_components(self) -> None:
        """
        Ran at the start of fit(), create the components needed for recommendation,
        and provide their hyperparameters
        """
        self._casino = Casino(
            initial_cu_threshold=self.initial_cu_threshold,
            certainty_window=self.certainty_window,
        )
        self._mfu = MatrixFactorizationUnit(refit_at=self.refit_at)
        self._clusterer = Clusterer(num_clusters=self.num_clusters)
        self._ranker = DefaultRanker()

    def _choose_recommender(
        self, user: int, enforce_is_cold: bool = None
    ) -> Union[Casino, MatrixFactorizationUnit]:
        """
        Used in recommend(). Choose the predictor for the user:
        - If enforce_is_cold is not None, set is_cold to the value. Else set it
        through self.is_cold(user)
        - Return the Casino if the user is cold, else the MFU
        Parameters
        ----------
        user: int
            The user
        enforce_is_cold: bool
            If provided, use this value for is_cold, hence forcing a specific
            recommender. If not provided, use recommender based on is_cold(user)
        Returns
        -------
            self._casino if the user is cold, else self._matrix_facotrization

        """
        if enforce_is_cold is not None:
            is_cold = enforce_is_cold
        else:
            is_cold = self.is_cold(user)
        return self._casino if is_cold else self._mfu
