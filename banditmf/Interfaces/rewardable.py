from abc import abstractmethod


class Rewardable:
    """
    A rewardable must be able to save rewards. The reward function should be called
    whenever the system receives a reward of a user for an item.
    """

    @abstractmethod
    def reward(self, user: int, item: int, rating: float) -> None:
        """
        Save the reward for a user-item combination
        Parameters
        ----------
        user: int
            The user
        item: int
            The item
        rating: float
            The rating in (0, 5)
        """
        pass
