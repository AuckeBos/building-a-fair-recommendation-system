from typing import no_type_check

import numpy as np
import pandas as pd
from Baselines.base_baseline import BaseBaseline
from numpy import ndarray


class RandomRecommender(BaseBaseline):
    """A baseline recommender that recommends random items"""

    @no_type_check
    def fit(self, *args) -> "RandomRecommender":
        """A RandomRecommender does not need to be fit"""
        return self

    def recommend(self, n: int, candidates: ndarray) -> pd.Series:
        """
        Recommend n items by randomy selecting them from the candidates
        Parameters
        ----------
        n: int
            The number of items to recommend
        candidates: np.array
            The list of candidates to select from
        Returns
        -------
        pd.Series
            List of recommendations. Eg randomly selected n items from candidates
             The predicted values are actually the rank in the list. Hence the
             predicted ratings are 1..n

        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        items = np.random.permutation(candidates)[:n]
        predictions = range(1, n + 1)
        return pd.Series(predictions, items)
