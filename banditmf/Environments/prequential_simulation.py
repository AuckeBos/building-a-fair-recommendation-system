from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from Environments.base_simulation import BaseSimulation
from helpers import X_y_to_frame
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score
from sklearn.utils import shuffle


class PrequentialSimulation(BaseSimulation):
    """
    A PrequentialSimulation is a Simulation that uses the ratings in a testset
    to simulate  recommendations and rewards. It is used to run Evaluations.
    
    This simulation runs by iterating over the u,i,r rows in a testset.
    For each row, it asks the recommender to recommend K items. Hence it receives a 
    ranking of K items; it is saved in self.rankings. Then it reward(u,i,r)'s the 
    recommender with the current row, and continues. Hence the recommendations are 
    only requested and saved, not used to reward() the recommender with. This 
    history is later used to compute_scores() with. 
    
    IMPORTANT:
    The Environment class contains an attribute _recommendations, which is returned 
    in result(). This attribute has no useful value anymore, because each item that 
    is recommended() is added. In this simulation, each stream item results in 10 
    recommendations, thus the _recommendations list contains too many 
    recommendations. Therefore result() is overridden, which raises a ValueError
    
    - simulate_one() recommends k items, saves those (along with the user and the 
        candidates), then rewards on the rating of the current stream item
    - computes_scores() Computes the ndcg@self.k for each item in self.ratings, 
        eg for each tuple in the stream. Note that for each user, self.k rankings cannot 
        be scored, namely the last k ones. At these points the length of the recs is 
        smaller than k, as there are only fewer than k candidates left. Hence ndcg@k 
        cannot be computed. If self.ensure_score_length, 0 is appended in those cases, 
        else nothing
    - generate_stream() Generates a stream of user,item,rating tuples out of the testset
    
    Attributes
    ----------
    stream: List[Tuple[int, int, float]]
        The stream of ratings to simulate. Values are user,item,rating.
    
    rankings: List[Tuple[int, np.array, pd.Series, boolean]]
        List of rankings. Each ranking consists of 4 elements: The user for which it 
        was created, The list of candidates at the time the ranking was created,
        the ranking of length self.k, boolean was_cold: true if the user was cold 
        just before this recommendation.
    
    k: int
        k for ndcg@k. Default is 10
        
    ensure_score_length: bool
        Default false. If true, ensure the length of scores is equal to the
            length of self.rankings. To do so, the scores contain '0' values at the
            timesteps where the ndcg@k could not be computed. These are the points
            were the length of the reclist (hence the length of the candidates) was
            smaller then self.k. This is the case self.k times for each user (namely 
            the last k items in the stream for that user).
    """
    stream: List[Tuple[int, int, float]]
    rankings: List[Tuple[int, ndarray, pd.Series, bool]]
    k: int = 10
    ensure_score_length: bool = False

    def simulate_one(self) -> Tuple[int, int]:
        """
        Simulating this simulator means:
        - Pop u,i,r tuple
        - Recommend self.k items
        - Save the result (ranking) in self.rankings
        - Reward the recommender with the u,i,r tuple of the stream
        """
        user, item, rating = self.stream.pop(0)
        was_cold = self.recommender.is_cold(user)
        candidates = self.candidate_selector.candidates(user)
        # Recommend k items, such that we can compute ndcg@K
        recommendations = self.recommend(user, self.k)
        self.rankings.append((user, candidates, recommendations, was_cold))
        self.reward(user, item, rating)
        return user, was_cold

    def compute_scores(self) -> NDArray[float]:
        """
        Compute nDCG@k for each ranking in self.rankings

        #A:
        To ndcg_score(), y_true and y_pred should be of equal size. But y_pred is
        always of len self.K. Hence we extend it with 0's for the unexisting (not
        recommended) items. Since we only compute ndcg@k, these won't be taken into
        account anyway.

        Returns
        -------
        NDArray[float]
            ndcg@k at each timestep, of length self.stream
        """
        # Use cache
        if self._scores is not None:  # type: ignore
            return self._scores  # type: ignore
        scores = []
        self.assert_finished()
        for user, candidates, recommendations, _ in self.rankings:
            # Reclist too small: One of the last k items in the testset for the user
            if len(recommendations) < self.k:
                if self.ensure_score_length:  # If ensure length, save ndcg of 0
                    scores.append(0)
                continue
            # Receive all actual ratings of all candidates
            ratings = self.test_matrix.loc[user, candidates]
            frame = (
                ratings.to_frame()
                .join(recommendations, lsuffix="_left", rsuffix="_right")
                .fillna(0)
            )  # A
            # y_true is first col, y_pred is second col
            ndcg = ndcg_score([frame.iloc[:, 0]], [frame.iloc[:, 1]], k=self.k)
            scores.append(ndcg)
        np_scores = np.array(scores)
        self._scores = np_scores
        return np_scores

    def generate_stream(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        do_shuffle: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Merge X,y into one frame, shuffle rows if desired. The stream is a list of the
        user,item,rating in the testset
        """
        frame = X_y_to_frame(X, y)
        # Shuffle set if required
        if do_shuffle:
            frame = shuffle(frame, random_state=seed)
        else:  # Else sort
            frame.sort_values("user")
        stream = list(frame.to_records(index=False))
        self.stream = stream

    def simulate(self) -> Iterator[int]:
        """
        Override simulate() to reset rankings
        """
        self.rankings = []
        return super().simulate()

    def result(self, **kwargs: Any) -> None:
        """
        Since self._recommendations is not valid in this class, raise error
        """
        raise ValueError("The PrequentialSimulation has no result(), only rankings")
