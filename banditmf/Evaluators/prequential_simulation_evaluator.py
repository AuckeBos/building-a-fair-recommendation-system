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
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, running_average, set_ids_to_subsets
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mlflow import log_metric, log_params, set_experiment, start_run
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit


class PrequentialSimulationEvaluator(BaseEvaluator):
    """
    The PrequentialSimulationEvaluator runs the PrequentialSimulation RUN_COUNT times 
    for USER_COUNT users. The results are averaged and logged to mlflow.
     
    The scores of each run are plotted over time. Eg the result is a line plot 
    (through MLflow) of ndcg@self.K for each timestep. The total number of timesteps is
    the number of ratings in the testset.  
    
    This evaluation is used in the thesis:
    - Section 4.4.3.2 discusses the PrequentialSimulation, which is the one
    used in this evaluator
    - This evaluator can create plots of ndcg@10 over time. Such plots are shown in 
    the thesis: Figures 4.8 and 4.9. 
    
    Attributes
    ----------
    AVERAGE_WINDOWS: List[int]
        Will compute running average of ndcg@self.K with these window sizes
    
    Parameters (Will be logged to mlflow)
    ----------
    K: int
        the k for which we compute ndcg@k
    """
    AVERAGE_WINDOWS = [10, 50, 100, 250, 500]
    K = 10

    RUN_COUNT = 1
    USER_COUNT = 1
    SHUFFLE_STREAM = True

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        """
        Run RUN_COUNT-fold cross-validation with USER_COUNT in each testset
        """
        self.log_params()
        X, y = DataLoader.movielens(random_state=self.RANDOM_STATE).train
        generator = create_generator(
            GroupShuffleSplit,
            n_splits=self.RUN_COUNT,
            test_size=self.USER_COUNT,
            random_state=self.RANDOM_STATE,
        )

        scores = Parallel(n_jobs=-1)(
            delayed(self.run_once)(*set_ids_to_subsets(X, y, train, test))
            for train, test in generator.split(X, y, X.user, X.item.to_numpy())
        )
        # Log each length as custom metric with value as step, to show on horizontal
        # axis in mlflow
        lengths = [len(x) for x in scores]
        for i, size in enumerate(sorted(lengths)):
            log_metric(f"{i+1}_of_{self.RUN_COUNT}_sets_exhausted", 0.9, step=size)
        max_len = max(lengths)
        avg_len = np.mean(lengths)
        log_metric("test_size", avg_len)  # type: ignore
        # pad scores with nan values up to max length
        scores = np.asarray(
            [
                np.pad(
                    s, (0, max_len - len(s)), "constant", constant_values=np.nan
                )  # type:ignore
                for s in scores
            ]
        )
        avg_scores = np.nanmean(scores, axis=0).tolist()  # type:ignore
        self.log_result(avg_scores)  # type: ignore

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> NDArray[float]:
        """
        Run the evaluation for 1 testset. Eg run simulation, log scores to mlflow
        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
        List[float]
            ndcg@self.K for each timestep
        """
        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            simulator = (
                PrequentialSimulation(train)
                .set_candidate_selector(
                    TestsetCandidateSelector().fit(X_y_to_matrix(*test))
                )
                .fit(
                    *test,
                    shuffle=self.SHUFFLE_STREAM,
                    seed=self.RANDOM_STATE,
                    log_to_mlflow=True,
                    k=self.K,
                )
            )
            log_metric("test_size", len(simulator.stream))
            list(simulator.simulate())
            scores = simulator.compute_scores()
            self.log_result(scores)
            return scores

    def log_result(self, scores: NDArray[float]) -> None:
        """
        Log the metrics to mlflow. Eg:
        - Log ndcg@k for each timestep
        Parameters
        ----------
        scores: List[float]
            ndcg@k for each timestep
        """
        key = f"ndcg_at_{self.K}"
        for i, score in enumerate(scores):
            log_metric(key, score, step=i)
        for w in self.AVERAGE_WINDOWS:
            key = f"ndcg_at_{self.K}_running_{w}_avg"
            avg = running_average(scores, w)
            for i, score in enumerate(avg):
                log_metric(key, score, step=i + w)

    @classmethod
    def plot_run(cls, run_id: str) -> None:
        """
        Plot a run of this evaluation. ndct@10 is plotted over time, with avg window 100
        Parameters
        ----------
        run_id: str
            Id of the parent run
        """

        ndcg = {
            metric.step: metric.value
            for metric in MlflowClient().get_metric_history(
                run_id, "ndcg_at_10_running_100_avg"
            )
        }
        plt.figure(figsize=(15, 5))
        plt.title(r"Running average with a window of 100 of $nDCG@10@t$")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$nDCG@10$")
        plt.plot(ndcg.keys(), ndcg.values())
        plt.show()

    def log_params(self) -> None:
        super().log_params()
        log_params(
            {
                "k": self.K,
                "Initial cold threshold": Casino.DEFAULT_INITIAL_CU_THRESHOLD,
                "Certainty window": Casino.DEFAULT_CERTAINTY_WINDOW,
                "Refit MF threshold": MatrixFactorizationUnit.DEFAULT_REFIT_AT,
                "Num clusters": Clusterer.DEFAULT_NUM_CLUSTERS,
            }
        )
