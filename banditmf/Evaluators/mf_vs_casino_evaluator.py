from typing import List, Tuple

import numpy as np
import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.casino import Casino
from Components.Rankers.base_ranker import BaseRanker
from Components.Rankers.default_ranker import DefaultRanker
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_frame, X_y_to_matrix, set_ids_to_subsets
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mlflow import log_metric, log_params, set_experiment, start_run
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GroupShuffleSplit


class MfVsCasinoEvaluator(BaseEvaluator):
    """
    The MfVsCasinoEvaluator runs two RatingStreamSimulations RUN_COUNT times, 
    where each run has a testset of USER_COUNT users.
    One simulation simulates BanditMF that always uses MF, one simulation simulates 
    BanditMF that always uses the Casino. Note that for each testset, as soon as a 
    user becomes non-cold, all other ratings of that user are skipped. This is 
    because this evaluator is used to measure performance on cold users
    
    The goal of this evaluator is to compare MF with BanditMF. 
    
    This evaluation is used in the thesis:
    - Section 4.4.5.1 discusses the comparison to the MF baseline. Its results are 
    retrieved through this evaluator, the plot (Figure 4.9) is created through 
    plot_result    
    
    Attributes
    ----------
    K: int
        the k for which we compute ndcg@k
    HYPER_PARAMETERS: Dict[str, Any]:
        Hyperparameters to use
    
    ranker: Ranker
        The ranker to use
    """
    RUN_COUNT = 1
    USER_COUNT = 1
    SHUFFLE = False

    K = 10
    HYPER_PARAMETERS = {
        "certainty_window": 3 * 2,
        "refit_at": 50,
        "num_clusters": 2,
    }
    ranker: BaseRanker

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        # Create and save Ranker
        self.ranker = DefaultRanker()
        self.log_params()
        splits = self._create_splits()

        ndcgs = Parallel(n_jobs=-1)(delayed(self.run_once)(tr, te) for tr, te in splits)
        avg_ndcgs = np.mean(np.array(ndcgs), axis=0)
        log_metric("casino_avg_ndcg", avg_ndcgs[0])
        log_metric("mf_avg_ndcg", avg_ndcgs[1])

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> Tuple[float, float]:
        """
        Run the simulations for 1 testset. Eg run both simulations, compute and log
        results
        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
            casino_avg_ndcg, mf_avg_ndcg
        """
        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            mf, casino = self._create_simulators(train, test)
            mf_results = mf.simulate()
            casino_results = casino.simulate()
            # While users in stream
            while casino.stream:
                user, _, _ = casino.stream[0]
                # If user is not cold anymore, skip it
                if casino.refit_moments[user] != -1:
                    del casino.stream[0]
                    del mf.stream[0]
                else:  # Simulate one
                    next(mf_results)
                    next(casino_results)
            return self.log_result(mf, casino)

    def log_result(
        self,
        mf: PrequentialSimulation,
        casino: PrequentialSimulation,
    ) -> Tuple[float, float]:
        """
        Log the run metrics to mlflow:
        For each ranking (eg each tuple in the testset):
        - Calculate APT, and separate APT for cold and non-cold users
        - Calculate the fraction of candidates that are minority
        - Increment count of rankings for that user
        save as dict artifact
        Parameters
        ----------
        mf: PrequentialSimulation
            The simulated MF simulation
        casino: PrequentialSimulation
            The simulated Casino simulation
        Returns
        -------
        Tuple[float, float]
            casino_avg_ndcg, mf_avg_ndcg
        """
        sims = {
            "casino": casino,
            "mf": mf,
        }
        result = []
        for name, simulator in sims.items():
            scores = simulator.compute_scores()
            for i, ndcg in enumerate(scores):
                log_metric(f"{name}_ndcg", ndcg, step=i)
            score = np.mean(scores)
            log_metric(f"{name}_avg_ndcg", score)
            result.append(score)
        return tuple(result)  # type: ignore

    @classmethod
    def plot_run(cls, run_id: str) -> None:
        """
        Plot a run of this evaluation. The run shows the ndcg@10 for both simulators
        Parameters
        ----------
        run_id: str
            Id of the child run
        """
        colors = ["tab:blue", "tab:red"]
        metric_keys = {
            "casino_ndcg": r"Casino",
            "mf_ndcg": r"MFU",
        }
        metrics = {
            name: {
                metric.step: metric.value
                for metric in MlflowClient().get_metric_history(run_id, key)
            }
            for key, name in metric_keys.items()
        }
        plt.figure(figsize=(15, 5))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$nDCG@10$")
        plt.title(r"$nDCG@10$ for BanditMF using the Casino versus using the MFU")
        for c, (key, values) in zip(colors, metrics.items()):
            xs = values.keys()
            ys = values.values()
            plt.plot(xs, ys, label=key, color=c)
        plt.legend()
        plt.show()

    def _create_splits(
        self,
    ) -> List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]:
        """
        Create train,test splits. Use create_generator().split(), but update the
        train sets as follows:

        Because the mf_simulator needs to predict items to users that do not occur in
        the original train set, we need to update the trainset: Each user in the
        testset must occur in the train set. If they wouldn't, the user would not
        occur in the predictions matrix, hence MF cannot predict for that user. To make
        sure this doesn't happen, we select the first rating for each user in the
        testset, and add that rating to the trainset.
        Returns
        -------
        List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
            List of train test splits

        """
        X, y = DataLoader.movielens(random_state=self.RANDOM_STATE).train
        generator = create_generator(
            GroupShuffleSplit,
            n_splits=self.RUN_COUNT,
            test_size=self.USER_COUNT,
            random_state=self.RANDOM_STATE,
        )
        splits = []
        for tr_idx, te_idx in generator.split(X, y, X.user, X.item.to_numpy()):
            (X_tr, y_tr), (X_te, y_te) = set_ids_to_subsets(X, y, tr_idx, te_idx)
            first_rating_per_user = (
                X_y_to_frame(X_te, y_te).groupby("user", as_index=False).first()
            )
            X_tr = X_tr.append(first_rating_per_user[["user", "item"]])
            y_tr = y_tr.append(first_rating_per_user["rating"])
            splits.append(((X_tr, y_tr), (X_te, y_te)))
        return splits

    def _create_simulators(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> Tuple[PrequentialSimulation, PrequentialSimulation]:
        """
        Create the two simulators for a train,test split.
        The simulators are equal, except for the value of enforce_is_cold
        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
        Tuple[PrequentialSimulation, PrequentialSimulation]
            mf_simulator, casino_simulator
        """

        mf_simulator = (
            PrequentialSimulation(train, self.HYPER_PARAMETERS)
            .set_candidate_selector(
                TestsetCandidateSelector().fit(X_y_to_matrix(*test))
            )
            .set_ranker(self.ranker)
            .set_enforce_is_cold(False)
            .fit(
                *test,
                shuffle=self.SHUFFLE,
                seed=self.RANDOM_STATE,
                ensure_score_length=False,
                k=self.K,
            )
        )
        casino_simulator = (
            PrequentialSimulation(train, self.HYPER_PARAMETERS)
            .set_candidate_selector(
                TestsetCandidateSelector().fit(X_y_to_matrix(*test))
            )
            .set_ranker(self.ranker)
            .set_enforce_is_cold(True)
            .fit(
                *test,
                shuffle=self.SHUFFLE,
                seed=self.RANDOM_STATE,
                ensure_score_length=False,
                k=self.K,
            )
        )
        return mf_simulator, casino_simulator

    def log_params(self) -> None:
        super().log_params()
        log_params(
            {
                "K": self.K,
                "Initial cold threshold": Casino.DEFAULT_INITIAL_CU_THRESHOLD,
                "Certainty window": self.HYPER_PARAMETERS["certainty_window"],
                "Refit MF threshold": self.HYPER_PARAMETERS["refit_at"],
                "Num clusters": self.HYPER_PARAMETERS["num_clusters"],
                "Ranker": self.ranker.__class__.__name__,
            }
        )
