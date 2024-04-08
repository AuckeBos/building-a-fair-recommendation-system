from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from Components.clusterer import Clusterer
from Components.history_aware_mab import HistoryAwareMab
from Components.Rewarders.base_rewarder import BaseRewarder
from helpers import validate_parameters
from Interfaces.algorithm import Algorithm
from Interfaces.predictor import Predictor
from Interfaces.rewardable import Rewardable
from mabwiser.mab import LearningPolicy


class Casino(Rewardable, Predictor, Algorithm):
    """The Casino is used as recommender for cold users.
    Upon fit, it creates one MAB for each user. Each MAB has one arm for each cluster
    that the Clusterer created.  Upon predict, one cluster from the MAB of the user
    is selected, using the UCB1 algorithm. The cluster (arm) is actually a list of
    predictions, one for each item. This list is returned. Upon reward, the arm for
    that MAB is rewarded based on the actual rating.

    The goal of the MAB is to find to which cluster the user belongs. If that cluster
    is found, the MAB is done, hence the user is non-cold. The MABs certainty is
    computed through certainty_achieved.

    Attributes
    ----------
    _clusters: pd.DataFrame
        Frame of clusters, one row per cluster, one column per item. Values are
        average ratings of all users in the cluster for that item

    _multi_armed_bandits: Dict[int, HistoryAwareMab]
        Dict containing one MAB for each user. Created on fit(). If a recommendation
        for a new user is later created, the MAB for that user is created at that moment

    _global_average_rating_: float
        Global average rating of all clusters. Initialized upon fit(). Used as
        initial reward for each arm, such that all arms have an equal probability

    DEFAULT_INITIAL_CU_THRESHOLD: int
        Default value for initial_cu_threshold parameter

    DEFAULT_CERTAINTY_WINDOW: int
        Default value for certainty_window parameter

    Parameters
    ----------
    initial_cu_threshold: int
        Upon fit, we create an MAB for each user with fewer ratings than this parameter.
        During training, we don't use this var anymore. We switch to MF as soon as
        the MAB is certain of a cluster. However we need it upon fit, since we need
        to determine which users to select for the Casino and which for the MFU
    certainty_window: int
        The window used to check whether an MAB is certain about the cluster for the
        user. If the MAB selects the same cluster for certainty_window timesteps in a
        row, the certainty is achieved.


    """

    _global_average_rating_: float
    _clusters: pd.DataFrame
    _multi_armed_bandits: Dict[int, HistoryAwareMab]

    # PARAMETERS
    initial_cu_threshold: int
    DEFAULT_INITIAL_CU_THRESHOLD = 35

    certainty_window: int
    DEFAULT_CERTAINTY_WINDOW = 2 * Clusterer.DEFAULT_NUM_CLUSTERS

    def __init__(self, initial_cu_threshold: int, certainty_window: int):
        self.initial_cu_threshold = initial_cu_threshold
        self.certainty_window = certainty_window

    @validate_parameters
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs: Any) -> "Casino":
        """
        Fitting the MAB means:
        - Save the clusters
        - For each cold user, create an MAB
        Parameters
        ----------
        X: pd.DataFrame
            The data to fit on. This is the clustered matrix, as created via the
            Clusterer. Shape (NUM_CLUSTERS, I)
        y: np.array
            Not used, added to adhere to base class

        Keyword Args
        ------------
        <matrix> pd.DataFrame:
            User x Item matrix, eg the unclustered X. Used to select the users from
            that are initially cold
        Returns
        -------
            self

        """
        if "matrix" not in kwargs:
            raise KeyError("Casino requires the original matrix")
        matrix = kwargs["matrix"]
        self._clusters = X
        self._global_average_rating_ = np.mean(X.mean())
        self._multi_armed_bandits = {}
        # Select all users with fewer ratings than initial_cu_threshold
        users = matrix[matrix.count(axis=1) < self.initial_cu_threshold].index.tolist()
        for user in users:
            self._create_mab_for_user(user)
        return self

    def predict(self, user: int) -> pd.Series:
        """
        The predictions for a user are the predictions for the selected cluster.
        The cluster is selected through the MAB of the user
        Parameters
        ----------
        user: int
            The user to return predictions for

        Returns
        -------
            pd.Series
                The predictions, maps item id to predicted rating
        """
        # If user is new, create empty mab
        if user not in self._multi_armed_bandits:
            self._create_mab_for_user(user)
        mab = self._multi_armed_bandits[user]
        selected_cluster_index = mab.predict()
        selected_cluster = self._clusters.iloc[selected_cluster_index]
        return selected_cluster

    def reward(self, user: int, item: int, rating: float) -> None:
        """
        Save reward into the mab of a user. The cluster to save the reward on is the
        cluster that was pulled latest
        Parameters
        ----------
        user: int
            The user we recommended to
        item: int
            The item that was recommended
        rating: float
            The actual rating of the u,i tuple. Converted to reward through Rewarder

        """
        reward = BaseRewarder.rating_to_reward(rating)
        mab = self._multi_armed_bandits[user]
        mab.reward(reward)

    def refit(self) -> None:
        """
        Refit the MAB. Ran after reward() in BanditMF, if must_refit/

        Todo: Implement. Idea:
            After refit, keep same clusters (eg same user ids in each
            cluster), but update the predicted values by re-computing the mean over the
            users. This way, the arm history can remain, but predictions will change
            after refit, such that we are using a somewhat up-to-date casino
        """
        pass

    def stopping_criteria_met(self, user: int) -> bool:
        """
        Check whether the MAB of a user has achieved certainty
        Parameters
        user: int
            The user to check

        Returns
        -------
        True if an MAB exists for the user, and the mab has achieved certainty
        """
        # If no MAB exists, certainty is not achieved
        if not self.user_exists(user):
            return False
        mab = self._multi_armed_bandits[user]
        return mab.certainty_achieved()

    def predicted_cluster(self, user: int) -> Union[str, int]:
        """
        The predicted cluster for a user is the cluster that was the favorite arm at
        the latest timestep. Should only be called if certainty achieved, else the
        returned cluster is not perse the predicted one, but simply the currently
        most favorite one
        Parameters
        ----------
        user: int
            The user to get the prediction for
        Returns
        -------
        Union[str, int]
            The favorite arm

        """
        mab = self._multi_armed_bandits[user]
        return mab.latest_favorite_arm()

    def is_cold(self, user: int) -> bool:
        """
        A user is cold if the stopping critrea is not met
        Parameters
        ----------
        user: int
            The user to check for
        Returns
        -------
            True if not criteria met. Note that if no MAB exists for the user (eg we
            never recommended anything yet to the user), this returns true

        """
        return not self.stopping_criteria_met(user)

    def user_exists(self, user: int) -> bool:
        """
        Check whether the user exists, eg whether we have an MAB for it
        Parameters
        ----------
        user: int
            The user to check
        Returns
        -------
        bool
            True if the user exists
        """
        return user in self._multi_armed_bandits

    def _create_mab_for_user(self, user: int) -> None:
        """
        Run for each cold user in fit(). Create an MAB:
        - Set the arms to the row index of self._clusters: one arm per row
        - The reward for each arm is the global average
        - Create HistoryAwareMAB, with one pull for each arm, with rewards global avg
        rating
        - Save the mab in self._multi_armed_bandits
        Parameters
        ----------
        user: int
            The user to create an MAB for
        """
        arms = self._clusters.index.tolist()
        # Initial fit: equal prob for all arms
        reward = self._global_average_rating_ / BaseRewarder.MAX_RATING
        initial_decisions = arms
        initial_rewards = [reward] * len(arms)
        mab = HistoryAwareMab(
            arms=arms,
            learning_policy=LearningPolicy.UCB1(alpha=1),
            certainty_window=self.certainty_window,
        )
        mab.partial_fit(initial_decisions, initial_rewards)
        self._multi_armed_bandits[user] = mab
