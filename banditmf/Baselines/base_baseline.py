from abc import abstractmethod
from typing import Optional, no_type_check

import pandas as pd
from Interfaces.algorithm import Algorithm
from numpy import ndarray


class BaseBaseline(Algorithm):
    """Abstract base class for baseline recommenders

    Attributes
    ----------
    random_state: Optional[int]
        seed. Used if provided
    """

    random: Optional[int]

    def __init__(self, seed: Optional[int] = None):
        """
        Init sets the seed.
        Parameters
        ----------
        seed: Optional[int]
            seed
        """
        self.random_state = seed

    @no_type_check
    @abstractmethod
    def fit(self, *args) -> "BaseBaseline":
        """
        Fit the baseline
        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def recommend(self, n: int, candidates: ndarray) -> pd.Series:
        """
        Recommend n items for user a user. The user does not need to be provided,
        since predictions of baselines are not personal.
        Parameters
        ----------
        n: int
            The number of items to recommend
        candidates: np.array
            The list of candidates to select from
        Returns
        -------
        pd.Series
            List of recommendations.
            The predicted values are actually the rank in the list. Hence the
            predicted ratings are 1..n

        """
        pass
