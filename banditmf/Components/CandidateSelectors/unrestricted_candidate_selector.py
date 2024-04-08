from typing import Any

from Components.CandidateSelectors.base_candidate_selector import BaseCandidateSelector
from numpy import ndarray


class UnrestrictedCandidateSelector(BaseCandidateSelector):
    """
    The UnrestrictedCandidateSelector has no restrictions. Hence each item is always a
    candidate
    
    Attributes
    ----------
    _items: np.array
        The list of all items in the system. Provided on fit()
    """
    _items: ndarray

    def fit(self, items: ndarray) -> "UnrestrictedCandidateSelector":
        """
        Simply save all items
        Parameters
        ----------
        items: ndarray
            List of all items
        Returns
        -------
        self

        """
        self._items = items
        return self

    def select(self, user: int, item: int) -> None:
        """
        Selecting an item has no effect: The candidates are unrestricted
        """
        pass

    def candidates(self, _: Any) -> ndarray:
        """
        All items are always candidates
        Parameters
        ----------
        _: Any
            Not used, added to adhere to base class
        Returns
        -------
        ndarray:
            self._items

        """
        return self._items
