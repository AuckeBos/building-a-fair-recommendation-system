from typing import Tuple

import numpy as np
import pandas as pd
from Baselines.base_baseline import BaseBaseline
from Baselines.popularity_based_recommender import PopularityBasedRecommender
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_frame, average_ndcgs, frame_to_matrix, set_ids_to_subsets
from mlflow import log_metric, log_param, log_text, start_run
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit


class BaselineEvaluator(BaseEvaluator):
    """
    The BaselineEvaluator performs a K-fold evaluation on a baseline evaluator. For
    each run, it computes the avg ndcg@k for each k in K_VALUES. The results are
    logged to mlflow.

    The goal of this evaluator is to compare performance of BanditMF to a baseline

    This evaluation is used in the thesis:
    - Section 4.4.5.2 discusses the comparison to the popularity-based baseline. Its
    results are  retrieved through this evaluator.

    Attributes
    ----------
    K_VALUES: List[int]
        Compute ndcg@k for each k
    BASELINE_CLASS: str
        The class of the baseline to use
    """

    RUN_COUNT = 1
    USER_COUNT = 1
    RANDOM_STATE = 1100
    K_VALUES = list(range(5, 205, 5))
    BASELINE_CLASS: BaseBaseline.__class__ = PopularityBasedRecommender

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        self.log_params()
        X, y = DataLoader.movielens(self.RANDOM_STATE).train
        generator = create_generator(
            GroupShuffleSplit,
            n_splits=self.RUN_COUNT,
            test_size=self.USER_COUNT,
            random_state=self.RANDOM_STATE,
        )
        scores = []
        for train, test in generator.split(X, y, X.user, X.item.to_numpy()):
            scores.append(self.run_once(*set_ids_to_subsets(X, y, train, test)))
        avg_scores = np.mean(np.array(scores), axis=0)
        self.log_result(avg_scores)

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> NDArray[float]:
        """
        Run the evaluation for 1 testset
        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
        NDArray[float]
            Scores: ndcg@k for all k
        """
        with start_run(nested=True):
            recommender = self.BASELINE_CLASS(self.RANDOM_STATE).fit(*train)
            test_frame = X_y_to_frame(*test)
            test_matrix = frame_to_matrix(test_frame)
            cs = TestsetCandidateSelector().fit(test_matrix)
            predictions = []
            # Can compute n recs directly, as no learning occurs
            for u, n in test[0].groupby("user").size().iteritems():
                recommendations = recommender.recommend(n, cs.candidates(u))
                predictions.extend([(u, i, p) for i, p in recommendations.iteritems()])

            predictions = pd.DataFrame(
                predictions, columns=["user", "item", "prediction"]
            )
            actual = test_frame
            scores = average_ndcgs(actual, predictions, self.K_VALUES)

            self.log_result(scores)
            return scores

    def log_result(self, scores: NDArray[float]) -> None:
        """
        Log scores to mlflow
        Parameters
        ----------
        scores: NDArray[float]
            ndcg@k
        """
        for k, ndcg in zip(self.K_VALUES, scores):
            log_metric(key="nDCG", value=ndcg, step=k)
            log_metric(key=f"ndcg_at_{k}", value=ndcg)

    def log_params(self) -> None:
        super().log_params()
        log_param("Baseline", self.BASELINE_CLASS)
        log_text(str(self.K_VALUES), "k_values.txt")
