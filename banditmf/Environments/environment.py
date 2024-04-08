from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from Components.bandit_mf import BanditMf
from Components.CandidateSelectors.base_candidate_selector import BaseCandidateSelector
from Components.CandidateSelectors.unrestricted_candidate_selector import (
    UnrestrictedCandidateSelector,
)
from Components.Rankers.base_ranker import BaseRanker


class Environment:
    """
    An Environment contains a history (training set) and a recommender (BanditMF). It
    exposes functions to recommend() and reward(). It also exposes its historical
    recommendations, including the rewards, via result()

    Attributes
    ----------
    history: Tuple[pd.DataFrame, pd.Series]
        The training set on which the recommender will be fit in __init__(). Hence is
        the X,y tuple of shapes (2, n_ratings), (n_ratings,)
        
    recommender: BanditMf
        The recommender. Is created and fit() upon __init__.
        
    candidate_selector: CandidateSelector
        The CS that is used during recommend(). By default uses the UnrestrictedCS
    
    _recommendations: List[Tuple[int, int, float]]
        Initialized empty on __init__(). Upon each recommendation, save the 
        prediction for the u,i. Can be retrieved at any point via result()
        
    _enforce_is_cold: Optional[bool]
        This value is provided to recommender.recommend(). If set, the recommender 
        will thus always see users as (non)cold

    """
    history: Tuple[pd.DataFrame, pd.Series]
    recommender: BanditMf
    candidate_selector: BaseCandidateSelector
    _recommendations: List[Tuple[int, int, float]]
    _enforce_is_cold: Optional[bool] = None

    def __init__(
        self,
        history: Tuple[pd.DataFrame, pd.Series],
        hyper_parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the environment. Requires a historical set of ratings (training set),
        and a dict of hyperparameters. It will create a recommender, and fit it on
        the training set.
        Parameters
        ----------
        history:  Tuple[pd.DataFrame, pd.Series]
            Train set in format (X, y) on which BanditMF is fit
        hyper_parameters: Dict[str, Any]
            The hyperparameters to be passed to BanditMf
        """
        hyper_parameters = hyper_parameters or {}
        self.history = history
        self.recommender = BanditMf(**hyper_parameters).fit(*history)
        # Extract unique items through X
        items = history[0]["item"].unique()
        self.candidate_selector = UnrestrictedCandidateSelector().fit(items)
        self._recommendations = []

    def recommend(self, user: int, n: int = None) -> pd.Series:
        """
        Recommend n items to the user. Restrict recommendations through
        self.candidate_selector. Also save recommendations in self._recommendations
        Parameters
        ----------
        user: int
            The user to recommend items to
        n: int
            The number of items to recommend. Defaults to 1
        Returns
        -------
            pd.Series
                Predictions (item -> predicted_value)

        """
        n = n or 1
        candidates = self.candidate_selector.candidates(user)
        recommendations = self.recommender.recommend(
            user, n, candidates, self._enforce_is_cold
        )

        # Save the recommendation for each item
        for item, value in recommendations.iteritems():
            self._recommendations.append((user, item, value))

        return recommendations

    def reward(self, user: int, item: int, rating: float) -> bool:
        """
        Reward the recommender. The reward is defined in terms of a rating in range
        (0, Rewarder.MAX_RATING). Rewarding means:
        - reward() on recommender
        - select() on the candidate selector
        Parameters
        ----------
        user: int
            The user that was recommended to
        item: int
            The item that was recommended
        rating: float
            The true rating of the item
        Returns
        -------
        bool
            True if the recommender was refit
        """
        was_refit = self.recommender.reward(user, item, rating)
        self.candidate_selector.select(user, item)
        return was_refit

    def result(
        self, as_frame: bool = True
    ) -> Union[List[Tuple[int, int, float]], pd.DataFrame]:
        """
        Retrieve the recommendations that have been made up until now in this
        Environment
        Parameters
        ----------
        as_frame: bool
            If true (default), return as user,item,prediction frame. Else as list of
            tuples

        Returns
        -------
        user,item,predictions recommendations. As list or as frame
        """
        if not as_frame:
            return self._recommendations
        frame = pd.DataFrame(
            self._recommendations, columns=["user", "item", "prediction"]
        )
        return frame

    def set_candidate_selector(self, selector: BaseCandidateSelector) -> "Environment":
        """
        Set the candidate selector. This selector will then be used in recommend().
        Note that the selector must have been fit()

        Parameters
        ----------
        selector: BaseCandidateSelector
            The fit selector
        Returns
        -------
        self
        """
        self.candidate_selector = selector
        return self

    def set_ranker(self, ranker: BaseRanker) -> "Environment":
        """
        Set fitted ranker on self.recommender.

        Parameters
        ----------
        ranker: BaseRanker
            The fit ranker
        Returns
        -------
        self
        """
        self.recommender.ranker = ranker
        return self

    def set_enforce_is_cold(self, enforce_is_cold: bool) -> "Environment":
        """
        Set enforce_is_cold value on self
        Parameters
        ----------
        enforce_is_cold: bool
            The value to set
        Returns
        -------
        self

        """
        self._enforce_is_cold = enforce_is_cold
        return self
