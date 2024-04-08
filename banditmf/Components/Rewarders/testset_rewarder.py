import pandas as pd
from Components.Rewarders.base_rewarder import BaseRewarder


class TestsetRewarder(BaseRewarder):
    """The testset rewarder provides rewards during simulation.

    IMPORTANT
    ---------
    This rewarder uses the testset to extract rewards from ratings. Therefor each
    user,item tuple that is rewarded, must occur in the testset. Therefor the
    TestsetCandidateSelector must be used CandidateSelector, when using this rewarder.

    The TestsetRewarder defines the reward as the known rating for the (user,item)
    combination. Hence it needs to be fit on the testset, and must be used in
    combination with a TestsetCandidateSelector. If another selector is used,
    it could happen that no rating exists in the testset for a certain user,
    item recommendation, in that case an KeyError will be raised.

    Attributes
    ----------

    _matrix: pd.DataFrame
        Complete testset User x Item rating matrix. Provided on fit()
    """

    _matrix: pd.DataFrame

    def fit(self, matrix: pd.DataFrame) -> "TestsetRewarder":
        """
        Fit the rewarder, eg save the matrix
        Parameters
        ----------
        matrix: pd.DataFrame
            Testset matrix. Shape (n_users x n_items)

        Returns
        -------
            self

        """
        self._matrix = matrix
        return self

    def get_reward(self, user: int, item: int) -> float:
        """
        The reward is read from the _matrix:
        - Read the rating in the matrix on index u,i
        - Convert the rating to a reward
        """
        rating = self._matrix.loc[user, item]
        reward = self.rating_to_reward(rating)
        return reward
