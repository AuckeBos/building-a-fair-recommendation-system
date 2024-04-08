from abc import abstractmethod

import pandas as pd


class Predictor:
    """
    A predictor in this context is not the same as in sklearn: In sklearn a Predictor
    consumes X and outputs y_pred. Here we consume a user and predict the ratings for
    each item in the environment
    """

    @abstractmethod
    def predict(self, user: int) -> pd.Series:
        """
        Predict the ratings for a user. The result is a list of predictions, one for
        each item
        Parameters
        ----------
        user: int
            The user to predict for

        Returns
        -------
        pd.Series:
            List of predictions, one for each item. unordered
        """
