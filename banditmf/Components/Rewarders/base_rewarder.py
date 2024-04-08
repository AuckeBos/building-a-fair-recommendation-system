from abc import abstractmethod


class BaseRewarder:
    """
    Baseclass for rewarders. Provides base functionality.

    A Rewarder provides rewards for a recommendation. A recommendation is defined
    by a (user, item) tuple. This abstract class must be implemented via a concrete
    class, which should provide get_reward().

    Attributes
    ----------

    MAX_RATING: float
        The max rating in the dataset. For now set to 5.0 (in line with Movielens)
        Todo: Read from dataset instead
    """

    MAX_RATING: float = 5.0

    def fit(self, **kwargs: dict) -> "BaseRewarder":
        """
        Rewarders generally need to be fit upon the testset

        Returns
        -------
            self
        """
        return self

    @abstractmethod
    def get_reward(self, user: int, item: int) -> float:
        """
        Get the reward for a recommendation
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
        raise NotImplementedError("BaseRewarder is an abstract class")

    def get_rating(self, user: int, item: int) -> float:
        """
        Get the rating for a recommendation, by converting the reward to a rating
        Parameters
        ----------
        user: int
            The user that was recommended to
        item: int
            The item that was recommended
        Returns
        -------
            The rating. In range [0, MAX_RATING]

        """
        return self.reward_to_rating(self.get_reward(user, item))

    @classmethod
    def reward_to_rating(cls, reward: float) -> float:
        """
        The reward for a rating is that rating divided by the max rating, such that
        it is normalize to [0, 1]
        """
        return reward * cls.MAX_RATING

    @classmethod
    def rating_to_reward(cls, rating: float) -> float:
        """
        Inverse of reward_to_rating
        """
        return rating / cls.MAX_RATING
