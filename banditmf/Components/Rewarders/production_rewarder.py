from abc import abstractmethod

from Components.Rewarders.base_rewarder import BaseRewarder


class ProductionRewarder(BaseRewarder):
    """
    The production rewarder provides a stub that should be used when ran in production.

    Attributes
    ----------

    MAX_RATING: float
        The max rating in the dataset
    """

    MAX_RATING: float = 5.0

    @abstractmethod
    def fit(self, **kwargs: dict) -> "ProductionRewarder":
        """
        Fit the rewarder on training data. it should extract the items that are
        available for rewarding, and save them
        Returns
        -------
            self
        """

    @abstractmethod
    def get_reward(self, user: int, item: int) -> float:
        """
        Get the reward for a recommendation. Should parse user behaviour and extract
        a rating
        Parameters
        ----------
        user: int
            The user that was recommended to
        item: int
            The item that was recommended
        Returns
        -------
            The reward. In range [0, 1]

        """
