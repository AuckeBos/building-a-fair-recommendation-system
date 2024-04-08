import math
import random
from typing import List

import pandas as pd
from Components.Rankers.base_ranker import BaseRanker
from DataLoaders.data_loader import DataLoader
from helpers import X_y_to_matrix


class FairRanker(BaseRanker):
    """The FairRanker is used to fairly re-rank a list of predictions.

    After ranking, the items are ordered on descending predicted rating,
    but it adheres to fairness constraints. The predicted ratings of the
    items are changed to n .. 1 (instead of in [0, 5.0], such that we are sure the new
    ranking is used  when computing scores: The item with the higest predicted value
    is recommended first, which is the first item in fair ranking (the one with
    predicted rating 'n')

    Attributes
    ----------
    _p: float
        Must be in [0.0, 1.0]. Probability of success on the Bernouli trial. Provided
        on init
    _categories: pd.Series
        Of length n_items. Maps each item to the category it belongs to. The category
        of an item must be either 0 (disadvantaged) or 1 (advantaged). Set on fit()

    """

    _p: float
    _categories: pd.Series

    def __init__(self, p: float):
        self._p = p

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FairRanker":
        """
        Extract the categories of the items of the dataset
        Parameters
        ----------
        X: pd.DataFrame
            2 column (user,item) frame (must contain all items that can ever be
            recommended, hence at least the testset)
        y: rating column of same length
        Returns
        -------
            self

        """
        self._categories = DataLoader.get_categories(X_y_to_matrix(X, y), "popularity")
        return self

    def set_p(self, p: float) -> "FairRanker":
        self._p = p
        return self

    def predict(self, items: pd.Series, n: int = 1) -> pd.Series:
        """
        Rank the items, select top n. Use a generative process:
        - Initialize result to empty list
        - Rerank items on decreasing predicted rating (highest rating first)
        - While the list is nog long enough, select one item, base on:
        - Bernouli(p). If does not succeed, the next item is simply the first item
        - If succeeds, select the category with the least exposure; select the first
            item of that category
        - Add item to list, remove from items, update exposure of categories
        Parameters
        ----------
        items: pd.Series
            The unordered items
        n: int
            The number of items to predict

        Returns
        -------
            The top n items of the reranked list

        """
        # Order on descending predicted rating
        possible_items = items.sort_values(
            ascending=False, inplace=False
        ).index.tolist()
        selected_items: List[int] = []
        # [unprotected_exposure, protected_exposure]
        exposures = [0.0, 0.0]
        # Generative process, continue untill large enough or no items remain
        while len(selected_items) < n and len(possible_items) > 0:
            # Select first item by default
            item = possible_items[0]
            select_fair = random.random() <= self._p
            # If must select fair, select first item that matches category
            if select_fair:
                # Category is one with least exposure
                category = exposures.index(min(exposures))
                # Select first item of category
                for possible_item in possible_items:
                    if self._categories[possible_item] == category:
                        item = possible_item
                        break
            else:  # Else use default (first) item, retrieve category
                category = self._categories[item]
            # Select item
            selected_items.append(item)
            # Update exposure of the category of the item
            exposures[category] += 1 / math.log2(len(selected_items) + 1)
            # Drop the item from the possible items
            possible_items.remove(item)
        # Convert items to series, the values are n .. 0: First item highest rank
        result = pd.Series(
            index=selected_items, data=range(len(selected_items), 0, -1), name=0
        )
        return result

    @property
    def p(self) -> float:
        return self._p
