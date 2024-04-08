from random import shuffle
from typing import List, Optional, Tuple

import pandas as pd
from Environments.base_simulation import BaseSimulation
from helpers import average_ndcgs, matrix_to_frame
from numpy.typing import NDArray
from sklearn.utils import check_random_state


class BatchSimulation(BaseSimulation):
    """
    A BatchSimulation is a Simulation that uses the users in a testset to 
    simulate recommendations and rewards. It is used in Evaluations.
    
    This simulation runs by iterating over the stream of user ids, which is the 
    (shuffled or sorted) column 'user' in the testset. For each item in the stream it 
    requests a recommendation and rewards the item that was recommended. 
    
    - simulate_one() Recommends 1 item for that user, and rewards the recommender 
        based on the true rating
    - computes_scores() Computes ndcg@k for each k in self.K_VALUES for each user in 
    the stream. Averaged over each user this gives a list of averaged scores, 
    one for each k in self.K_VALUES
    - generate_stream() Generates a stream of users
    
    Attributes
    ----------        
    stream: List[Tuple[int]]
        The stream of users to simulate. During simulation, users are popped(). Hence 
        the simulation has finished when users.empty()
         
    K_VALUES: List[float]
        The values of k to compute ndcg@k for        
    """

    stream: List[Tuple[int]]

    K_VALUES = list(range(5, 205, 5))

    def simulate_one(self) -> Tuple[int, int]:
        """
        Simulating an item means:
        - recommend one item for the user
        - Reward the recommender for the item the recommended
        """
        user = self.stream.pop(0)[0]
        was_cold = self.recommender.is_cold(user)
        recommendations = super().recommend(user)
        item = recommendations.index[0]
        rating = self.rewarder.get_rating(user, item)
        self.reward(user, item, rating)
        return user, was_cold

    def compute_scores(self) -> NDArray[float]:
        """
        Compute nDCG@k for each k in ks. Compute for each user separately,
        then average over the users

        Returns
        -------
        List[float]
            List of ndcg@k. Size len(ks)
        """
        self.assert_finished()

        # Use cache
        if self._scores is not None:  # type: ignore
            return self._scores  # type: ignore

        actual = matrix_to_frame(self.test_matrix)
        predictions = self.result()  # type: pd.DataFrame
        scores = average_ndcgs(actual, predictions, self.K_VALUES)
        self._scores = scores
        return scores

    def generate_stream(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        do_shuffle: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        The stream of this simulator is the list of user ids, eg the column 'user' of X.
        """
        stream = []
        for user, rating_count in X.groupby("user").size().iteritems():
            stream.extend([(user,)] * rating_count)
        if do_shuffle:  # If shuffle
            check_random_state(seed).shuffle(stream) if seed is not None else shuffle(
                stream
            )
        else:  # Else sort
            stream.sort()
        self.stream = stream
