from typing import Any, Dict, List

import numpy as np
from DataLoaders.cross_validation_generator import create as create_generator
from Environments.batch_simulation import BatchSimulation
from Evaluators.HyperparameterSearch.base_hyperparameter_search import (
    BaseHyperparameterSearch,
)
from helpers import set_ids_to_subsets
from mlflow import log_metric, log_metrics, log_param, log_params
from sklearn.model_selection import GroupShuffleSplit


class HyperparameterSearchTypeA(BaseHyperparameterSearch):
    """
    The HyperparameterSearch of type A optimizes the ndcg@k for different ks. The
    ndcg is computed over the rankings deduced after a complete simulation. It uses
    the BatchSimulation.

    The score is as follows: The evaluation computes ndcg@K for each K in K_VALUES,
    for each user in the testset. The used rankings for the ndcg is the list of
    recommendations as a result of the simulation for that user. The
    ndcgs are averaged over all users, and then over all splits. Hence for each run
    we have ndcg@K for each K. The final score is the average of the ndcg@k for a
    specific k, namely K_FOR_NDCG_FOR_LOSS. The loss is 1 - this value.

    This evaluation is used in the thesis:
    - Section 4.4.3.1 discusses the BatchSimulation, which is the one used
    in this evaluator
    - Section 4.4.4.1 discusses the result of this parameter searched. The tables are
    generated from the data that is logged to mlflow.

    Attributes
    ----------
    K_FOR_NDCG_FOR_LOSS: int
        We use 1 - ndcg@k as loss function to optimize using skopt.
        k = K_FOR_NDCG_FOR_LOSS. The value must be one of
        BatchSimulation.K_VALUES, otherwise its value would not have been
        computed by the simulation
    """

    N_COMBINATIONS = 1
    BATCH_SIZE = 1
    RUN_COUNT = 1
    USER_COUNT = 1
    K_FOR_NDCG_FOR_LOSS = 100

    SIMULATOR_CLASS = BatchSimulation

    def split(self) -> None:
        """
        Create the train,test splits:
        - use create_generator to create a train,test generator
        - Create idxs using the generator
        - Convert idxs to actual subsets of the data
        - Save in self.splits
        """
        splits = list(
            create_generator(
                GroupShuffleSplit,
                test_size=self.USER_COUNT,
                n_splits=self.RUN_COUNT,
                random_state=self.RANDOM_STATE,
            ).split(self.X, self.y, self.X.user, self.X.item.to_numpy())
        )
        self.splits = [set_ids_to_subsets(self.X, self.y, tr, te) for tr, te in splits]

    @classmethod
    def format_train_result(cls, simulations: List[BatchSimulation]) -> Dict[str, Any]:
        """
        Format the train results of a batch of simulations
        Returns
        -------
        Dict[str, Any]
            Contains:
                - At index 'metrics' a dict:
                - ndcg_at_k -> value for each k
                - avg_ndcg -> value
                - loss -> 1 - ndcg@K_FOR_NDCG_FOR_LOSS
                - avg_refit_at -> The avg moment a user went from cold to warm
                - fraction_refit -> Fraction of users that went from cold to warm
            - At index 'loss' the loss

        """
        # The train score is simply the average of the score of each simulation
        scores = np.mean(
            np.array([simulation.compute_scores() for simulation in simulations]),
            axis=0,
        )
        refit_moments = [
            m for simulation in simulations for m in simulation.refit_moments.values()
        ]
        metrics = {
            f"ndcg_at_{k}": score for k, score in zip(BatchSimulation.K_VALUES, scores)
        }
        metrics["avg_ndcg"] = np.mean(scores)
        metrics["loss"] = 1 - metrics[f"ndcg_at_{cls.K_FOR_NDCG_FOR_LOSS}"]
        avg_refit_at, fraction_refit = BatchSimulation.get_refit_info(refit_moments)
        metrics["avg_refit_at"] = avg_refit_at
        metrics["fraction_refit"] = fraction_refit
        result = {
            "metrics": metrics,
            "loss": metrics["loss"],
        }
        return result

    @classmethod
    def log_result(cls, train_result: Dict[str, Any]) -> None:
        """
        - Log each metric
        - Log 'ndcg' at timestep. The timesteps are k. Hereby we can show a graph in
        the mlflow ui which shows the ndcg_at_k
        - Log the parameters
        Parameters
        ----------
        train_result: Dict[str, Any]
            The result. contains metrics at index 'metrics', and hyper parameters at
            index 'parameters'
        """
        log_metrics(train_result["metrics"])
        log_params(train_result["parameters"])
        for k in BatchSimulation.K_VALUES:
            log_metric("ndcg", train_result["metrics"][f"ndcg_at_{k}"], step=k)

    def log_params(self) -> None:
        super().log_params()
        log_param("k_for_loss", self.K_FOR_NDCG_FOR_LOSS)
