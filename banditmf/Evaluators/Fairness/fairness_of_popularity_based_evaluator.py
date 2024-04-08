from typing import Any, Tuple

import numpy as np
import pandas as pd
from Baselines.popularity_based_recommender import PopularityBasedRecommender
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from Evaluators.Fairness.base_fairness_evaluator import BaseFairnessEvaluator
from helpers import X_y_to_frame, frame_to_matrix
from mlflow import log_metric, log_text, start_run


class FairnessOfPopularityBasedEvaluator(BaseFairnessEvaluator):
    """
    The FairnessOfPopularityBasedEvaluator evaluates the fairness of the baseline 
    PopularityBased recommender through k-fold cross-validation. For each fold, 
    for each user, it computes the APT over the top self.K items
    
    The results of this evaluation are not used in the thesis, but rather used for 
    manual investigating behavior
    
    Attributes
    ----------
    See parent class
    """
    RUN_COUNT = 1
    USER_COUNT = 1

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        self.load_data()
        apts = [self.run_once(tr, te) for tr, te in self.splits]
        apt = np.mean(np.array(apts))
        log_metric("apt", apt)
        log_text(apts.__repr__(), "apts.txt")

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> float:
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
        float
            apt

        """
        with start_run(nested=True):
            recommender = PopularityBasedRecommender().fit(*train)
            test_frame = X_y_to_frame(*test)
            test_matrix = frame_to_matrix(test_frame)
            cs = TestsetCandidateSelector().fit(test_matrix)
            pts = []
            # For each user, recommend 10 items and compute apt
            for u in test[0]["user"].unique():
                ranking = recommender.recommend(self.K, cs.candidates(u))
                pt = self.pt(ranking)
                pts.append(pt)
            apt = np.mean(np.array(pts))
            log_metric("apt", apt)
            log_text(pts.__repr__(), "pts.txt")
            return apt

    def log_run(self, simulation: PrequentialSimulation) -> Any:
        """
        Logging is done in run_once
        """
        pass
