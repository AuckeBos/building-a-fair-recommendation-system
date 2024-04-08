from typing import Any, Dict, List

import numpy as np
from DataLoaders.cross_validation_generator import create as create_generator
from Environments.base_simulation import BaseSimulation
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.HyperparameterSearch.base_hyperparameter_search import (
    BaseHyperparameterSearch,
)
from helpers import running_average, set_ids_to_subsets
from mlflow import log_metric, log_param, log_params, log_text
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit


class HyperparameterSearchTypeB(BaseHyperparameterSearch):
    """
    The HyperparameterSearch of type B optimizes the hyperparameters using ndcg@self.K
    for each timestep. It uses the PrequentialSimulation. For each item in the
    testset, and computes ndcg@self.K for the recommendations that the RS makes at
    that time. This means that for each tr,te split, the scores are of different
    length (due to different testset sizes). Hence we override functionality to:
    - Crop the test sets to the length of the shortest set. This makes each split
    comparable (they learn from the same number of items)
    - Crop the scores of each run to the lenght of the shortest list. Even though the
     testset are of equal length, the score lenghts might differ a little: Since
     ndcg@self.K cannot be computed for the last K items for each user,
     as the TestsetCandidateSelector provides less than K candidates at those points.
     Hence we crop to the shortest length before computing average.

    The final score for a hyperparameter combination is averaged over each split.
    for each split it computes the running average ndcg@K. The window is
    self.WINDOW_FINAL_SCORE, and we take the last value. Hence it uses the
    average ndcg@K  over the last  WINDOW_FINAL_SCORE timesteps. The loss is
    1 - this score.

    This evaluation is used in the thesis:
    - Section 4.4.3.2 discusses the PrequentialSimulation, which is the one
    used in this evaluator
    - Section 4.4.4.2 discusses the result of this parameter searched. The tables are
    generated from the data that is logged to mlflow. The figures that are shown here
    are generated through plot_run of the PrequentialSimulationEvaluator


    Attributes
    ----------
    AVERAGE_WINDOWS: List[int]
        Will compute running average of ndcg@self.K with these window sizes

    Parameters (Will be logged to mlflow)
    ----------
    K: int
        the k for which we compute ndcg@k
    WINDOW_FINAL_SCORE: int
        The final score of a parameter combination is, averaged over all train,
        test splits, the average ndcg@self.K over the last WINDOW_FINAL_SCORE
        recommendation lists. Value should be one of AVERAGE_WINDOWS
    """

    N_COMBINATIONS = 1
    BATCH_SIZE = 1
    RUN_COUNT = 1
    USER_COUNT = 1

    AVERAGE_WINDOWS = [10, 50, 100, 250, 500]
    K = 10
    WINDOW_FINAL_SCORE = 100

    SIMULATOR_CLASS = PrequentialSimulation
    SIMULATOR_KWARGS = {"k": K}

    def split(self) -> None:
        """
        Each train,test split will have a different number of ratings in the testset.
        Hence each run would see another amount of data. This is a) not comparable (some
        splits receive more ratings hence more info), b) the 'scores' of the
        simulator would have different lenghts. This makes computing the average
        'scores' impossible. Since if we average over the last element, we actually
        only take the element of the tr,te split with the longest test size. Hence:

        Cut of each test set at the length of the shortest test set
        """
        splits = list(
            create_generator(
                GroupShuffleSplit,
                test_size=self.USER_COUNT,
                n_splits=self.RUN_COUNT,
                random_state=self.RANDOM_STATE,
            ).split(self.X, self.y, self.X.user, self.X.item.to_numpy())
        )
        testset_lenghts = [len(te_idx) for _, te_idx in splits]
        log_text(str(testset_lenghts), "testset_lengths.txt")
        # Compute len of shortest test set. Cut off each test set (they are shuffled)
        min_test_len = min(testset_lenghts)
        self.splits = [
            set_ids_to_subsets(self.X, self.y, tr_idx, te_idx[:min_test_len])
            for tr_idx, te_idx in splits
        ]

    @classmethod
    def format_train_result(
        cls,
        simulations: List[PrequentialSimulation],
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        Dict[str, Any]
            - At index 'metrics' a dict:
                - ndcgs -> ndcg@self.k for each timestep. indexed on timestep
                - running_avg_ndcgs -> mapping w to timestep to running
                    average ndcg@self.K at that timestep
                - loss -> the loss. Eg
                 1 - last_running_avg_window_{self.WINDOW_FINAL_SCORE}_ndcg@{self.K}
                - avg_refit_at -> The avg moment a user went from cold to warm
                - fraction_refit -> Fraction of users that went from cold to warm
            - At index 'loss' the loss

        """
        scores = cls._get_train_score(simulations)
        refit_moments = [
            m for simulation in simulations for m in simulation.refit_moments.values()
        ]

        # Wil map 'w' to dict. dict maps 'timestep' to avg ndcg at timestep with
        # window w
        running_avg_ndcgs = {}
        for w in cls.AVERAGE_WINDOWS:
            avg = running_average(scores, w)
            running_avg_ndcgs[w] = {i + w: score for i, score in enumerate(avg)}
        avg_refit_at, fraction_refit = BaseSimulation.get_refit_info(refit_moments)
        metrics = {
            "ndcgs": dict(enumerate(scores)),  # map timestep to ndcg@self.k
            "running_avg_ndcgs": running_avg_ndcgs,
            "loss": 1 - list(running_avg_ndcgs[cls.WINDOW_FINAL_SCORE].values())[-1],
            # Loss is 1 - running_avg_window_{self.WINDOW_FINAL_SCORE}_ndcg@{self.K} at
            # final timestep
            "avg_refit_at": avg_refit_at,
            "fraction_refit": fraction_refit,
        }
        return {
            "metrics": metrics,
            "loss": metrics["loss"],
        }

    @classmethod
    def _get_train_score(cls, simulations: List[BaseSimulation]) -> NDArray[float]:
        """
        In this search, not every simulation has the same number of scores. Hence we
        crop all scores to the length of the shortest score list
        """
        scores = [simulation.compute_scores() for simulation in simulations]
        lenghts = list(map(len, scores))
        log_text(str(lenghts), "score_lengths.txt")
        min_len = min(lenghts)
        new_scores = np.array([score[:min_len] for score in scores])
        avg_scores = np.mean(new_scores, axis=0)
        return avg_scores

    @classmethod
    def log_result(cls, train_result: Dict[str, Any]) -> None:
        """
        - Log each metric. These are
            - ndcg_at_cls.K at each timestep
            - ndcg_at_cls.K running average over w for each w for each timestep
            - loss, avg_refit_at, fraction_refit
        - Log the parameters
        Parameters
        ----------
        train_result: Dict[str, Any]
            The result. contains metrics at index 'metrics', and hyper parameters at
            index 'parameters'
        """
        metrics = train_result["metrics"]
        key = f"ndcg_at_{cls.K}"
        for i, score in metrics["ndcgs"].items():
            log_metric(key, score, step=i)

        for w, avg_ndcgs in metrics["running_avg_ndcgs"].items():
            key = f"ndcg_at_{cls.K}_running_{w}_avg"
            for i, score in avg_ndcgs.items():
                log_metric(key, score, step=i)
        for key in ["loss", "avg_refit_at", "fraction_refit"]:
            log_metric(key, metrics[key])
        log_params(train_result["parameters"])

    def log_params(self) -> None:
        super().log_params()
        log_param("k", self.K)
        log_param("window_final_score", self.WINDOW_FINAL_SCORE)
