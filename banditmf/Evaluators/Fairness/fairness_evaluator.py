import pickle
import typing
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.Rankers.fair_ranker import FairRanker
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from Evaluators.Fairness.base_fairness_evaluator import BaseFairnessEvaluator
from helpers import X_y_to_matrix, object_to_mlflow
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mlflow import get_run, log_metric, search_runs, set_experiment, start_run


class FairnessEvaluator(BaseFairnessEvaluator):
    """
    The FairnessEvaluator runs a PrequentialSimulation RUN_COUNT times, where
    each run has  a testset of USER_COUNT users. After each simulation, it computes
    the fairness at each timestep, and logs it to mlflow.

    The plots of this evaluation are shown in the thesis:
    - Section 4.5.1: Unfairness plot, Figure 4.12 (Not with the FairRanker but with
        the Ranker)
    - Section 4.6.2.1: fairness plot, Figure 4.13 (With the FairRanker(0.8)

    Attributes
    ----------
    See parent class
    """

    RUN_COUNT = 1
    USER_COUNT = 1

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        """
        Run a simulation RUN_COUNT times.
        """
        # Create and fit FairRanker with p=0.8
        self.load_data()
        self.ranker = FairRanker(0.8).fit(self.X, self.y)
        self.log_params()

        apts = Parallel(n_jobs=-1)(
            delayed(self.run_once)(tr, te) for tr, te in self.splits
        )

        avg_apts = np.mean(np.array(apts), axis=0)
        log_metric("apt", avg_apts[0])
        log_metric("apt_cold", avg_apts[1])
        log_metric("apt_non_cold", avg_apts[2])

    def run_once(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> Tuple[float, float, float]:
        """
        Run the evaluation for 1 testset:
        - Run simulation
        - Compute fairness
        - Log to mlflow
        - Return result for parent aggregation

        Parameters
        ----------
        train: pd.DataFrame
            X, y train data
        test: pd.DataFrame
            X, y test data
        Returns
        -------
        float:
            avg of apt, apt_cold, apt_non_cold
        """
        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            simulator = (
                PrequentialSimulation(train, self.HYPER_PARAMETERS)
                .set_candidate_selector(
                    TestsetCandidateSelector().fit(X_y_to_matrix(*test))
                )
                .set_ranker(self.ranker)
                .fit(
                    *test,
                    shuffle=self.SHUFFLE,
                    seed=self.RANDOM_STATE,
                    ensure_score_length=True,
                    k=self.K,
                )
            )
            # Fully run simulation
            list(simulator.simulate())
            return self.log_run(simulator)  # type: ignore

    @typing.no_type_check
    def log_run(self, simulation: PrequentialSimulation) -> Tuple[float, float, float]:
        """
        Log the run metrics to mlflow:
        For each ranking (eg each tuple in the testset):
        - Calculate PT, and separate PT for cold and non-cold users
        - Calculate the fraction of candidates that are minority
        - Increment count of rankings for that user

        Save as dict artifact to mlflow
        Parameters
        ----------
        simulation: The ran simulation

        Returns
        -------
            apt, apt_cold, apt_non_cold

        """
        # Iterate over rankoings, save pts and frac tail candidates
        pts, cold_pts, non_cold_pts, frac_tail_candidates = [], [], [], []
        counts = defaultdict(int)
        scores = simulation.compute_scores()
        for user, candidates, ranking, is_cold in simulation.rankings:
            pt = self.pt(ranking)
            pts.append(pt)
            lst = cold_pts if is_cold else non_cold_pts
            lst.append(pt)
            frac_tail_candidates.append(
                len([c for c in candidates if c in self.long_tail]) / len(candidates)
            )
            counts[user] += 1
        # Save to mlflow
        data = {
            "pts": pts,
            "cold_pts": cold_pts,
            "non_cold_pts": non_cold_pts,
            "frac_tail_candidates": frac_tail_candidates,
            "counts": counts,
            "refit_moments": simulation.refit_moments,
            "scores": scores,
        }
        object_to_mlflow(data, "fairness_data")
        apt = np.mean(pts)
        apt_cold = np.mean(cold_pts)
        apt_non_cold = np.mean(non_cold_pts)
        log_metric("apt", apt)
        log_metric("apt_cold", apt_cold)
        log_metric("apt_non_cold", apt_non_cold)
        return apt, apt_cold, apt_non_cold

    @classmethod
    def plot_run(cls, parent_run_id: str) -> None:
        """
        Create matplotlib plots for all child runs. They show the PT@10 for each
        timestep. Assumes that the data is saved in an artifact in the run,
        as in self.log_run()
        Parameters
        ----------
        parent_run_id: str
            The id of the parent run
        """
        parent_run = get_run(parent_run_id)
        runs = search_runs(
            [parent_run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'",
        )
        # Loop over child runs
        for run in runs.iterrows():
            # Todo: fix this path extraction
            path = f"{run[1]['artifact_uri']}/fairness_data/artifact.pickle"[8:]
            with open(path, "rb") as file:
                data = pickle.load(file)
            plt.figure(figsize=(25, 5), dpi=200)
            pts = np.array(data["pts"])
            cold_pts = np.array(data["cold_pts"])
            non_cold_pts = np.array(data["non_cold_pts"])

            print(f"APT: {np.mean(pts):.4f}")
            print(f"APT for cold users: {np.mean(cold_pts):.4f}")
            print(f"APT for non-cold users: {np.mean(non_cold_pts):.4f}")
            # Save starting points of new users, and refit moments of new users
            u_start, u_refit = [], []
            i = 0
            for user, count in data["counts"].items():
                u_start.append(i)
                refit_at = data["refit_moments"][user]
                if refit_at > -1:
                    u_refit.append(i + refit_at)
                i += count
            # Scatter the u_start points at y=0
            plt.scatter(
                u_start,
                [0] * len(u_start),
                color="green",
                marker="o",
                label=r"Start new $u$",
            )
            # Scatter the u_refit points at y=0
            plt.scatter(
                u_refit,
                [0] * len(u_refit),
                color="red",
                marker="x",
                label=r"Cold $\to$ non-cold switch",
            )
            scores = np.array(data["scores"])
            # Create gaps in the plots where the ndcg@k was 0, these were the points
            # where the ndcg was not computed
            scores[scores == 0.0] = np.nan
            # Plot ndcg, pts
            plt.plot(range(len(scores)), scores, c="blue", label="nDCG", linestyle="-")
            plt.plot(range(len(pts)), pts, c="y", label="APT", linestyle="--")
            # Figure layout
            plt.xlim(left=0, right=2500)
            plt.legend(loc="upper right")
            plt.xlabel(r"$t$")
            plt.ylabel(r"$nDCG@10, PT@10$")
            plt.title(r"$PT@10$ and $nDCG@10$")
            plt.show()
            plt.savefig(f"../plots/figs/{run[1]['run_id']}.png")
