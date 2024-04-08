import typing
from copy import deepcopy
from typing import Any, List, Union

import mabwiser.mab
from helpers import keys_of_max_val
from mabwiser.ucb import _UCB1


class HistoryAwareMab(mabwiser.mab.MAB):
    """Custom MAB class that keeps track of its prediction history
    Attributes
    ----------
    pull_history: List
        Upon predict, add the predicted arm to this list, hence keep track of the
        history of pulls
    pull_count_per_arm: Dict
        Summed number of pulls per arm, indexed on arm name
    favorite_arms: List
        Contains the favorite arm at time t, where t is the index in the list. The
        favorite arm is the arm with the most number of pull at that time.
    state_history: List[UBC1]
        History of states. After each prediction, we save a copy of the current UCB1,
        which holds values like current mean and UCBs
        Todo: remove this, for now used in Evaluations

    certainty_window:
        The window to use when doing a certainty check. If we are certain about the
        cluster of certainty_window time steps, we are done

    latest_pull:
        The arm that was selected during the latest predict(). Used in the
        self.reward() to select the arm to save reward for. Note: This means that
        one must save a reward for each pull, else this latest_pull value might be
        the wrong arm

    """

    pull_history: List
    pull_count_per_arm: dict
    favorite_arms: List
    state_history: List[_UCB1]
    certainty_window: int
    latest_pull: Any

    @typing.no_type_check
    def __init__(self, arms, learning_policy, certainty_window: int):
        """
        Override init to initialize the attributes
        Parameters
        ----------
        arms
        learning_policy
        certainty_window
        """
        super().__init__(arms, learning_policy)
        self.certainty_window = certainty_window
        self.pull_history = []
        self.favorite_arms = []
        self.state_history = []
        self.pull_count_per_arm = {arm: 0 for arm in arms}

    @typing.no_type_check
    def predict(self, contexts=None):
        """
        Override function to save the arm that was pulled
        """
        result = super().predict(contexts)
        # For now: throw error if predicted more than one item
        if isinstance(result, list):
            raise TypeError("Error: Predicted more than 1 arm")
        # Save the pull
        self.latest_pull = result
        return result

    def reward(self, reward: float) -> None:
        """
        Reward the MAB. The cluster that is rewarded is self.latest_pull
        - Save pull in history
        - Run partial fit
        Parameters
        ----------
        reward: float
            The reward
        """
        selected_cluster = self.latest_pull
        # Save the pull
        self.pull_history.append(selected_cluster)
        self.pull_count_per_arm[selected_cluster] += 1
        # Compute favorite arm: arm with most pulls, -1 if more than 1
        favorite_arms = keys_of_max_val(self.pull_count_per_arm)
        favorite_arm = favorite_arms[0] if len(favorite_arms) == 1 else -1
        self.favorite_arms.append(favorite_arm)
        self.state_history.append(deepcopy(self._imp))
        self.partial_fit([selected_cluster], [reward])

    def certainty_achieved(self, before: int = None) -> bool:
        """
        Return true if we are certain that the user of this MAB belongs to a cluster
        We are certain if one arm has been the favorite for certainty_window times in a
        row
        Parameters
        ----------
        before: int
            If provided, check whether certainty was achieved before this point in
            time (index), eg we subset favorite_arms up to and including this index.

        Returns
        -------
        bool
            True if we are certain of the cluster of the user

        """
        if before is not None:
            favorite_arms = self.favorite_arms[: before + 1]
        else:
            favorite_arms = self.favorite_arms
        # Cannot be certain if have not pulled enough times
        if len(favorite_arms) < self.certainty_window:
            return False
        recent_favorites = favorite_arms[-self.certainty_window :]
        distinct_favorites = list(set(recent_favorites))
        # Return true if there is only a single favorite. (-1 means  no favorite)
        return len(distinct_favorites) == 1 and distinct_favorites[0] != -1

    def latest_favorite_arm(self) -> Union[str, int]:
        return self.favorite_arms[-1]
