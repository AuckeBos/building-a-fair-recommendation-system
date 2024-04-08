from typing import List, Tuple

import numpy as np
import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.casino import Casino
from Components.clusterer import Clusterer
from Components.mfu import MatrixFactorizationUnit
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Environments.batch_simulation import BatchSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, object_to_mlflow, set_ids_to_subsets
from joblib import Parallel, delayed
from mlflow import log_metric, log_params, set_experiment, start_run
from sklearn.model_selection import GroupShuffleSplit


class BatchSimulationEvaluator(BaseEvaluator):
    """
    The BatchSimulationEvaluator runs a BatchSimulation RUN_COUNT times, where each 
    run has  a testset of USER_COUNT users. After each simulation, it computes 
    nDCG@k for all k in K_VALUES. Scores are saved in 'runs'
    
    The nDCGs are computed as follows. After the simulation is finished, the results 
    are split per user. For each user this results in a ranking of items. For each of 
    these rankings the ndcg@k is computed for all k in K_VALUES. The nDCGs are averaged
    over all users and then over all splits. The result is avg nDCG@k for each k
    
    This evaluation is used in the thesis:
    - Section 4.4.3.1 discusses the BatchSimulation, which is the one
    used in this evaluator.
    
    Attributes
    ----------
    K_VALUES: int
        The k values for which to compute nDCG@k
    """
    RUN_COUNT = 1
    USER_COUNT = 1

    K_VALUES = list(range(5, 105, 5))

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        self.log_params()
        X, y = DataLoader.movielens().train
        generator = create_generator(
            GroupShuffleSplit, n_splits=self.RUN_COUNT, test_size=self.USER_COUNT
        )
        simulations = Parallel(n_jobs=-1)(
            delayed(self.run_once)(*set_ids_to_subsets(X, y, train, test))
            for train, test in generator.split(X, y, X.user, X.item.to_numpy())
        )
        object_to_mlflow(simulations, "simulations")
        self.log_aggregated_result(simulations)

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> BatchSimulation:
        """
        Run the evaluation for 1 testset. Eg loop over the simulation results,
        log status to stdout, return simulator
        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
        BatchSimulation
            The finished simulation
        """
        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            simulator = (
                BatchSimulation(train)
                .set_candidate_selector(
                    TestsetCandidateSelector().fit(X_y_to_matrix(*test))
                )
                .fit(*test, shuffle=True)
            )
            for _ in simulator.simulate():
                print(self._get_status(simulator), end="\r")
            self.log_result(simulator)
            return simulator

    def log_result(self, simulation: BatchSimulation) -> None:
        """
        Log the result of a simulation to mlflow:
        -
        Parameters
        ----------
        simulation: BatchSimulation
            The simulated simulation
        """
        scores = simulation.compute_scores()
        refit_moments = list(simulation.refit_moments.values())
        avg_refit_moment, frac_refit_achieved = BatchSimulation.get_refit_info(
            refit_moments
        )
        for k, ndcg in zip(self.K_VALUES, scores):
            log_metric(key="nDCG", value=ndcg, step=k)
        log_metric("avg_refit_at", avg_refit_moment)
        log_metric("fraction_refit", frac_refit_achieved)

    def log_aggregated_result(self, simulations: List[BatchSimulation]) -> None:
        """
        Log all aggregated info of runs
        Parameters
        ----------
        simulations: List[BatchSimulation]
            List of simulation simulations
        """
        scores = [simulation.compute_scores() for simulation in simulations]
        refit_moments = [m for sim in simulations for m in sim.refit_moments.values()]

        avg_scores = np.mean(np.array(scores), axis=0)
        for k, ndcg in zip(self.K_VALUES, avg_scores):
            log_metric(key="nDCG", value=ndcg, step=k)
        avg_moment, frac_refit_achieved = BatchSimulation.get_refit_info(refit_moments)
        log_metric("avg_refit_at", avg_moment)
        log_metric("fraction_refit", frac_refit_achieved)

    def log_params(self) -> None:
        super().log_params()
        log_params(
            {
                "Initial cold threshold": Casino.DEFAULT_INITIAL_CU_THRESHOLD,
                "Certainty window": Casino.DEFAULT_CERTAINTY_WINDOW,
                "Refit MF threshold": MatrixFactorizationUnit.DEFAULT_REFIT_AT,
                "Num clusters": Clusterer.DEFAULT_NUM_CLUSTERS,
            }
        )

    def _get_status(self, simulator: BatchSimulation) -> str:
        """
        Get status as string. Status contains the number of preds for each user
        """
        data = [
            f"User {user}: {c}" for user, c in simulator.recommendation_counts.items()
        ]
        status = " - ".join(data)
        return status
