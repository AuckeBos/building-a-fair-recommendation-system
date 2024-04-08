import pandas as pd
from Components.Rankers.base_ranker import BaseRanker


class DefaultRanker(BaseRanker):
    """Default ranker. Simply ranks items on descending predicted rating"""

    def rank(self, items: pd.Series) -> pd.Series:
        """
        Rank the items by simply sorting on descending predicted rating
        Parameters
        ----------
        items: pd.Series
            The items to rank. The index are the item names, the values are the
            predicted ratings
        Returns
        -------
            The re-ranked items

        """
        return items.sort_values(ascending=False)

    def predict(self, items: pd.Series, n: int = 1) -> pd.Series:
        """
        Rank items, select top n
        Parameters
        ----------
        items: pd.Series
            The (unordered) items to rank
        n: int
            The number of items to predict
        Returns
        -------
            The top n items of the reranked list

        """
        reranked = self.rank(items)
        return reranked[:n]
