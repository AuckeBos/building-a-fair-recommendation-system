from abc import abstractmethod

import pandas as pd
from numpy.typing import ArrayLike


class BaseCandidateSelector:
    """Base class for candidate selectors
    
    A CandidateSelector used when recommending items by BanditMF. The candidate 
    selector is able to tell BanditMF which items are recommendation
    candidates for each user. This class is an abstract class; specific 
    implementations exist. They must define the fit() function.
    
    The candidates() function is defined in this base class, it uses _candidates.

    Attributes
    ----------
    _candidates: pd.DataFrame
        Integer User x Item matrix. Values are 1 if are a candidate, else 0
    """

    _candidates: pd.DataFrame

    @abstractmethod
    def fit(self, **kwargs: dict) -> "BaseCandidateSelector":
        """
        Fit the selector. Eg create an int matrix of user x item, values are 1 if
        are candidates, eg if rating is known, else 0. Save in self._candidates

        Returns
        -------
            self
        """
        raise NotImplementedError("CandidateSelector is an abstract class")

    @abstractmethod
    def select(self, user: int, item: int) -> None:
        """
        Must be ran whenever an item is selected for a user. Might update _candidates
        to unset an item as a candidate
        Parameters
        ----------
        user: int
            The user that was selected for
        item: int
            The item that was selected
        """
        raise NotImplementedError("CandidateSelector is an abstract class")

    def candidates(self, user: int) -> ArrayLike:
        """
        The candidates for a user are all items that have value 1 in self._candidates
        Parameters
        ----------
        user: int
            The user to get candidates for

        Returns
        -------
        np.array:
            The names of the items that are candidates
        """
        all_items = self._candidates.loc[user]
        candidates = all_items[all_items == 1].index.to_numpy()
        if not candidates.any():
            raise KeyError(f"No candidates found for user {user}")
        return candidates
