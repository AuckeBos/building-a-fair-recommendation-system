from abc import abstractmethod

import pandas as pd


class BaseRanker:
    """
    Abstract class. Specific rankers should extend this class. Provides function
    interface
    """

    @abstractmethod
    def predict(self, items: pd.Series, n: int = 1) -> pd.Series:
        """
        Rank items according to ranking algorithm, select the top n
        Parameters
        ----------
        items: pd.Series
            The items to rank. Maps item to predicted rating
        n: int
            The number of items to select after ranking
        Returns
        -------
            ps.Series: The top n items based on the ranking procedure
        """
        pass
