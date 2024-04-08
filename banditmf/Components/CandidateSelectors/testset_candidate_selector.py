import pandas as pd
from Components.CandidateSelectors.base_candidate_selector import BaseCandidateSelector


class TestsetCandidateSelector(BaseCandidateSelector):
    """
    In the TestsetCandidateSelector an item is a candidate for a user if that item has a
    rating in the testset, and the item has not yet been selected. 
    
    Attributes
    ----------
    _candidates: pd.DataFrame
        integer matrix, shape users x items. Each cell is 1 if the item is a 
        candidate for the user, else it is 0. Is initialized upon fit
    """

    def fit(self, testset: pd.DataFrame) -> "TestsetCandidateSelector":
        """
        Convert the rating matrix into a boolean matrix. Values are true if the
        rating exists in the testset. Upon select(), the cell is set to 0,
        as the item is no longer a candidate for the user
        Parameters
        ----------
        testset: pd.DataFrame
            User x Item testing matrix
        Returns
        -------
            self
        """
        matrix_as_bool = testset > 0
        candidates = matrix_as_bool.astype(int)
        self._candidates = candidates
        return self

    def select(self, user: int, item: int) -> None:
        """
        Whenever an item is selected for a user, that item is not a candidate
        anymore. Hence update the value to 0 in the matrix
        """
        self._candidates.loc[user, item] = 0
