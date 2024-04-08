from abc import ABC, abstractmethod
from math import floor
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Components.Rewarders.base_rewarder import BaseRewarder
from Components.Rewarders.testset_rewarder import TestsetRewarder
from Environments.environment import Environment
from helpers import X_y_to_matrix
from mlflow import set_tag
from numpy.typing import NDArray
from tqdm import tqdm


class BaseSimulation(Environment, ABC):
    """
    The BaseSimulation is an abstract class, specific implementations exist. Note 
    that a simulation is a specific Environment, namely one that simulates 
    recommend() and reward()
    
    A Simulation provides functionality to simulate a recommender for a testset. It 
    must expose functions:
    - simulate_one() to simulate one item of the stream
    - computes_scores() to compute the score of the RS after simulation has completed
    - generate_stream() to generate the stream of tuples out of a test set, 
        this stream is iterated while simulating
    
    This base class provides interface functions and some functionality that is used 
    in all simulations.
    
    Attributes
    ----------
    rewarder: Rewarder
        The rewarder that is used to reward() the recommender during simulation. By 
        default uses the TestsetRewarder
    
    stream: List[Tuple]
        Stream of items in the testset. Tuples can take different formats, depending 
        on the implementation of the simulation
    
    recommendation_counts: Dict[int, int]
        Maps user to the number of recommendations we have simulated, 0 for each user
        before the simulation is ran
        
    refit_moments: Dict[int, int]
        Maps user to the moment in time (eg nr of recs) it was switched from cold to 
        warm. Maps to -1 if not refit before end of simulation
        
    users: List[int]
        List of unique users in the user stream. Set in fit()
    
    test_matrix: pd.DataFrame
        Test matrix. set on fit()
            
    _scores: NDArray[float]
        Cache of scores. If scores are computed once. Save them, don't recompute 
        again. See compute_scores()
    
    log_to_mlflow: bool
        If true, log every 10% of simulation to mlflow. Default is False
    """

    rewarder: BaseRewarder
    stream: List[Tuple]

    recommendation_counts: Dict[int, int]
    refit_moments: Dict[int, int]

    users: List[int]
    test_matrix: pd.DataFrame

    _scores: Optional[NDArray[float]] = None
    log_to_mlflow: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        rewarder: BaseRewarder = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        log_to_mlflow: bool = False,
        **kwargs: Dict[str, Any]
    ) -> "BaseSimulation":
        """
        Fit the simulator, eg provide the data that is needed to run the simulation
        Parameters
        ----------
        X: pd.DataFrame
            Testset data to simulate. Columns (user, item), shape (2, n_ratings)
        y: pd.Series
            Ratings, shape (n_ratings,)
        rewarder: BaseRewarder
            The rewarder to user, use the TestsetRewarder if not provided
        shuffle: bool
            Whether to shuffle the stream. If false, will sort (eg one user will be
            simulated for all its ratings in X before continuing to the next user)
        seed: Optional[int]
            If set, use this as seed for randomizing the stream. Has no effect if
            shuffle=False
        log_to_mlflow: Optional[bool]
            Set attribute on self. If true, will log to mlflow during simulate()
        **kwargs: Dict[str, Any]
            Optional parameters for specific implementations of this class. Own
            attributes (self.__dict__) are set with these values

        Returns
        -------
            self
        """
        self.test_matrix = X_y_to_matrix(X, y)
        self.rewarder = rewarder or TestsetRewarder().fit(self.test_matrix)
        self.generate_stream(X, y, shuffle, seed)
        self.users = list(X.user.unique())
        self.log_to_mlflow = log_to_mlflow

        # Initialize recommendation counts and refit moments
        self.recommendation_counts = {}
        self.refit_moments = {}
        for user in self.users:
            self.recommendation_counts[user] = 0
            self.refit_moments[user] = -1

        for k, v in kwargs.items():
            self.__setattr__(k, v)
        return self

    def simulate(self) -> Iterator[int]:
        """
        Run the simulation:
        - Keep track of position. Log to mlflow every 10% of stream (if desired)
        - Loop over stream
        - Run self.simulate_one, and keep track of rec counts and refit moments
        - Compute scores when done
        Yields
        ------
        Iterator[int]
            The users that were recommended to
        """
        # Keep track of position, to log progress to mlflow. Log every 10%
        stream_size = len(self.stream)
        update_interval = floor(stream_size / 10)
        i = 0
        with tqdm(total=stream_size, ascii=True, ncols=50) as pbar:
            while self.stream:
                i += 1
                # simulate_one() is implemented in specific implementations
                user, was_cold = self.simulate_one()
                self.recommendation_counts[user] += 1
                # Cold switch
                if was_cold and not self.recommender.is_cold(user):
                    self.refit_moments[user] = self.recommendation_counts[user]
                pbar.update()
                if self.log_to_mlflow and i % update_interval == 0:
                    set_tag("progress", str(pbar))
                yield user
        self.compute_scores()

    @abstractmethod
    def simulate_one(self) -> Tuple[int, bool]:
        """
        Simulate one item of the stream
        Must be overridden by specific simulation. Must:
        - Pop the item from the stream
        - Recommend for the current stream item
        - Reward th recommender
        - Save something for the current iteration, if desired
        - Return User

        Returns
        -------
        User, was_cold
        """

    @abstractmethod
    def compute_scores(self) -> NDArray[float]:
        """
        Compute the scores of the simulation. Before compute, check if is set in
        self.scores. If so return it. Before return, set self.scores to prevent
        recompute
        Returns
        -------
        NDArray[float]
            Scores
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        do_shuffle: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Must be implemented by specific implementations
        Must generate the stream of tuples. Shuffle if desired, else sort
        Parameters
        ----------
        X: pd.DataFrame
            Columns (user, item), shape (2, n_ratings)
        y: pd.Series
            values. shape (n_ratings,)
        do_shuffle: Optional[bool]
            If true, shuffle stream, else sort. Defaults to false
        seed: Optional[int]
            If not None and shuffle is true, this is seed

        """
        pass

    @classmethod
    def get_refit_info(cls, refit_moments: List[int]) -> Tuple[float, float]:
        """
        Get the statistics about when and how often the MFU was refit during this
        simulation
        Parameters
        ----------
        refit_moments: List[int]
            List of refit moments. -1 if not refit, else the number of
            recommendations that was needed before the user went from cold to warm.
            For a simulation instance, can be retrieved via
            list(self.refit_moments.values())
        Returns
        -------
        Tuple[float, float]
            avg_moment: The average moment (in terms of number of recommendations
            done) at which a user went from cold to warm
            frac_refit_achieved: The fraction of users that got warm before the
            simulation finished

        """
        moments = np.array(refit_moments)
        if not moments[moments > -1].any():
            return np.NAN, 0
        avg_moment = np.mean(moments[moments > -1])
        num_not_refit = len(moments[moments == -1])
        num_total = len(moments)
        frac_refit_achieved = 1 - num_not_refit / num_total
        return avg_moment, frac_refit_achieved  # type: ignore

    def result(
        self, as_frame: bool = True
    ) -> Union[List[Tuple[int, int, float]], pd.DataFrame]:
        """
        Retrieve the recommendations of the simulation, if the simulation is
        finished. See documentation of super function
        Raises
        ------
        RuntimeError
            If the simulation has not yet finished

        """
        self.assert_finished()
        return super().result(as_frame)

    def assert_finished(self) -> None:
        """
        Raises
        ------
        RuntimeError
            If not finished
        """
        if not self.finished:
            raise RuntimeError(
                "Simulation result accessed before simulation was finished"
            )

    @property
    def finished(self) -> bool:
        """
        A simulation is finished if the stream is empty
        Returns
        -------
        bool
            True if finished

        """
        return not self.stream
