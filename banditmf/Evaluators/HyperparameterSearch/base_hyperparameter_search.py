from abc import ABC, abstractmethod
from math import floor
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.Rewarders.base_rewarder import BaseRewarder
from DataLoaders.data_loader import DataLoader
from Environments.base_simulation import BaseSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, create_logger, log
from joblib import Parallel, delayed
from mlflow import log_params, log_text, set_experiment, start_run
from skopt import Optimizer
from skopt.space import Categorical, Integer
from sqlalchemy import Numeric


class BaseHyperparameterSearch(BaseEvaluator, ABC):
    """
    The BaseHyperparameterSearch defines the base functionality of a search. TypeA
    and TypeB override some functions to alter behaviour. They must at least implement:

    - split: Create train,test splits
    - format_train_result: Get the results from a list of simulated simulations
    - log_result: Log the result of format_train_result to mlflow

    A search is ran as follows:
    - Create RUN_COUNT train,test splits
    - Create an skopt optimizer. It will optimize using GaussianProcessRegressor
    - Start an mlflow parent run
    - Do USER_COUNT times the following:
        - Sample BATCH_SIZE sets of hyperparameters from the optimizer. For each set:
            - train(parameters). Eg:
                - For each train,test in the splits:
                    - create BaseSimulation, run it
                    - compute scores
                - Average the scores over all splits, return the averaged result
        - For each result:
            - Log to mlflow as a child run
            - tell() the optimizer the loss
    - Retrieve the best result(), save in parent mlflow run

    We use parallelization for efficiency. We use it as follows:
    - We run train() at BATCH_SIZE threads simultaneously. During train(),
    we simulate() for RUN_COUNT train,test splits, with one thread for each split.

    Attributes
    ----------
    N_COMBINATIONS: int
        The total number of combinations we will test in the search. Note that
        N_COMBINATIONS should be a multiple of BATCH_SIZE, since we split it in
        batches. If N_RUNS % BATCH_SIZE =/= 0, it will be floor()'d
    BATCH_SIZE: int
        Number of train()'s to run simultaneously
    X: pd.DataFrame
        The complete set of training data. Will create tr/val split on these
    y: pd.Series
        Labels for X
    splits:  List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
        List of train,test splits. Do not contain the indices, but the actual data
    PARAMETER_SPACE: Dict[str, Any]
        The parameter space to search. Note that values are in each run updated  via
        format_parameters, such that BanditMF accepts them
    SIMULATOR_CLASS: class
        Class of the simulator to use, Differs per specific Search
    SIMULATOR_KWARGS: Dict
        Kwargs to provide to fit() of the simulator
    """

    N_COMBINATIONS = 1
    BATCH_SIZE = 1

    X: pd.DataFrame
    y: pd.Series

    splits: List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]

    PARAMETER_SPACE = {
        "num_clusters": Integer(2, 6),
        "certainty_window": Integer(1, 5),  # Will be  * num_clusters
        "refit_at": Categorical([5, 10, 15]),  # Will be * MAX_RATING
    }
    SIMULATOR_CLASS: BaseSimulation.__class__
    SIMULATOR_KWARGS: Dict = {}

    @abstractmethod
    def split(self) -> None:
        """
        Create the train,test splits, save in self.splits
        """
        pass

    @classmethod
    @abstractmethod
    def format_train_result(
        cls,
        simulations: List[BaseSimulation],
    ) -> Dict[str, Any]:
        """
        Format the return value of the train() function
        Parameters
        ----------
        simulations: List[BaseSimulation]
            List of simulated simulations
        Returns
        -------
        Dict[str, Any]
            Contains at lest:
            - At index 'metrics' a dict mapping metric names to values:
            - At index 'loss' the loss value
        """
        pass

    @classmethod
    @abstractmethod
    def log_result(cls, train_result: Dict[str, Any]) -> None:
        """
        Log the result of train() to mlflow
        Parameters
        ----------
        train_result: Dict[str, Any]
            The result of train(). In format of format_train_result()
        """
        pass

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        """
        The evaluate function is the same for all hyperparam searches
        Returns
        -------

        """
        self.log_params()
        # Create train,test splits
        self.X, self.y = DataLoader.movielens(random_state=self.RANDOM_STATE).train
        self.split()
        # Save result history. 'params' => train(params)
        result_history = {}
        optimizer = Optimizer(dimensions=list(self.PARAMETER_SPACE.values()))
        # Number of batches to run
        n_batches = floor(self.N_COMBINATIONS / self.BATCH_SIZE)
        logger = create_logger(n_batches)
        log(
            logger,
            f"Starting hypersearch: {n_batches} * {self.BATCH_SIZE} = {self.N_COMBINATIONS}",
        )
        for i in range(n_batches):
            search_points = optimizer.ask(self.BATCH_SIZE)
            # Drop points that are already searched
            search_points = [p for p in search_points if repr(p) not in result_history]
            # Format params
            parameter_set = [self.format_parameters(p) for p in search_points]
            results = Parallel(n_jobs=-1)(
                delayed(self.train)(params) for params in parameter_set
            )
            for x, y in zip(search_points, results):
                result_history[repr(x)] = y
                optimizer.tell(x, y["loss"])
            log(logger, "", i + 1)
        # Get best result, save to parent run of mlflow
        best_result = result_history[repr(optimizer.get_result().x)]
        self.log_result(best_result)

    def train(self, parameters: Dict[str, Union[int, float]]) -> Dict[str, Any]:
        """
        The training function to minimize. For each tr,te split, run the simulation
        async.

        Each run returns:
            - scores: ndcg@k for each k
            - parameters: The used parameters
            - refit_moments: Number of recommendations that was needed before the
            user was refit, for each user. Else -1
        - Average result over all splits. These are metrics params and loss

        Parameters
        ----------
        parameters: Dict[str, Union[int, float]]
            parameters

        Returns
        -------
            Dict[str, Any]
                See format_train_result

        """
        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            simulations = Parallel(n_jobs=-1)(
                delayed(self.simulate_one_split)(tr, te, parameters)
                for tr, te in self.splits
            )
            train_results = self.format_train_result(simulations)
            train_results["parameters"] = parameters
            self.log_result(train_results)
            return train_results

    def simulate_one_split(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
        parameters: Dict[str, Any],
    ) -> BaseSimulation:
        """
        Run the simulation for a single train,test split
        Parameters
        ----------
        train: Tuple[pd.DataFrame, pd.Series] X,y
        test: Tuple[pd.DataFrame, pd.Series] X,y
        parameters: Dict[str, Any] Hyper parameters
        Returns
        -------
        BaseSimulation
            The simulated simulation

        """
        cs = TestsetCandidateSelector().fit(X_y_to_matrix(*test))
        simulation = (
            self.SIMULATOR_CLASS(train, parameters)
            .set_candidate_selector(cs)
            .fit(
                *test,
                seed=self.RANDOM_STATE,
                **self.SIMULATOR_KWARGS,
                log_to_mlflow=True,
            )
        )
        list(simulation.simulate())
        return simulation

    @classmethod
    def format_parameters(cls, parameters: List[int]) -> Dict[str, Numeric]:
        """
        Reformat the hyperparameters. They are retrieved via the gp_minimize() of
        skopt.
        Here we reformat them such that BanditMF accepts them. This means:
        - Convert to named dict. Use ordering of self.PARAMETER_SPACE
        - certainty_window = certainty_window * num_clusters. We want it to be
        relative to num clusters.
        - refit_at = refit_at * MAX_RATING. We want it to be relative to the max rating
        Parameters
        ----------
        parameters: Dict[str, int]
            The parameters, as set via PARAMETER_SPACE.

        Returns
        -------
            Dict[str, Union[int, float]]
                The parameters reformatted
        """
        reformated = dict(zip(cls.PARAMETER_SPACE.keys(), parameters))
        reformated["certainty_window"] *= reformated["num_clusters"]
        reformated["refit_at"] *= BaseRewarder.MAX_RATING
        return reformated

    def log_params(self) -> None:
        super().log_params()
        log_params(
            {
                "n_combinations": self.N_COMBINATIONS,
                "batch_size": self.BATCH_SIZE,
            }
        )
        log_text(str(self.PARAMETER_SPACE), "parameter_space.txt")
