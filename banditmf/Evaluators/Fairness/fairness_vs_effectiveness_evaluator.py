import typing
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.Rankers.fair_ranker import FairRanker
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from Evaluators.Fairness.base_fairness_evaluator import BaseFairnessEvaluator
from helpers import X_y_to_matrix
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from mlflow import log_metric, log_param, set_experiment, start_run
from mlflow.tracking import MlflowClient


class FairnessVsEffectivenessEvaluator(BaseFairnessEvaluator):
    """
    The FairnessVsEffectivenessEvaluator measures the balance between effectiveness 
    and fairness. It uses the FairRanker to create a ranker that is fair to some 
    extent:
    
    Create RUN_COUNT splits. Each testset has USER_COUNT users
    For p in PS:
        create FairRanker with p=p
        start child run
        For each tr,te split:
            run simulation, save apt and ndcg
        compute avg apt
        save
    save all apt and ndcg for each p in mlflow
    
    The plots of this evaluation are shown in the thesis:
    - Section 4.6.2.2: Figure 4.14. nDCG vs APT is discussed
        
    Attributes
    ----------
    PS:
        The values of p to test
    """
    PS = np.round(np.arange(0, 1.1, 0.1), 2)  # type: ignore
    RUN_COUNT = 1
    USER_COUNT = 1

    @BaseEvaluator.register_mlflow
    def evaluate(self) -> None:
        self.load_data()
        # Create fit and save Ranker, p will be set inside loop
        self.ranker = FairRanker(0).fit(self.X, self.y)
        self.log_params()

        results = Parallel(n_jobs=-1)(
            delayed(self.run_for_p)(self.splits, deepcopy(self.ranker).set_p(p))
            for p in self.PS
        )

        for p, result in zip(self.PS, results):
            # Set P to integer, such that can log with step to mlflow
            p = int(p * 10)
            log_metric("avg_ndcg", result[0], step=p)
            log_metric("avg_apt", result[1], step=p)
            log_metric("avg_apt_cold", result[2], step=p)
            log_metric("avg_apt_non_cold", result[3], step=p)

    def run_for_p(
        self,
        splits: List[
            Tuple[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
        ],
        ranker: FairRanker,
    ) -> Tuple[float, float, float, float]:
        """
        Run all simulations (all folds) for a specific value of p
        Parameters
        ----------
        splits: List
            List of train,test splits
        ranker: FairRanker
            The ranker with a certain p value
        Returns
        -------
        float:
            avg ndcg@k, apt, apt cold, apt non cold
        """

        @typing.no_type_check
        def run_once(
            train: Tuple[pd.DataFrame, pd.Series],
            test: Tuple[pd.DataFrame, pd.Series],
        ) -> Tuple[float, float, float, float]:
            """
            Run the evaluation for 1 split. Return avg ndcg@self.K, apt
            """
            simulator = (
                PrequentialSimulation(train, self.HYPER_PARAMETERS)
                .set_candidate_selector(
                    TestsetCandidateSelector().fit(X_y_to_matrix(*test))
                )
                .set_ranker(ranker)
                .fit(
                    *test,
                    shuffle=self.SHUFFLE,
                    seed=self.RANDOM_STATE,
                    ensure_score_length=True,
                    k=self.K,
                )
            )
            list(simulator.simulate())
            return self.get_simulation_result(simulator)

        set_experiment(self.__class__.__name__)
        with start_run(run_name="child", tags={"mlflow.parentRunId": self.run_id}):
            log_param("p", ranker.p)
            results = Parallel(n_jobs=-1)(  # type: ignore
                delayed(run_once)(tr, te) for tr, te in splits  # type: ignore
            )
            avg_results = np.mean(np.array(results), axis=0)
            log_metric("avg_ndcg", avg_results[0])
            log_metric("avg_apt", avg_results[1])
            log_metric("avg_apt_cold", avg_results[2])
            log_metric("avg_apt_non_cold", avg_results[3])
            return tuple(avg_results)  # type: ignore

    def get_simulation_result(
        self, simulation: PrequentialSimulation
    ) -> Tuple[float, float, float, float]:
        """
        Get the result of a simulation
        Parameters
        ----------
        simulation: PrequentialSimulation
            The ran simulation
        Returns
        -------
            Tuple[float, float,float, float]
            avg ndcg@self.K, apt, apt_cold, apt_non_cold

        """
        pts, cold_pts, non_cold_pts = [], [], []  # type: ignore
        for user, candidates, ranking, is_cold in simulation.rankings:
            pt = len(ranking[ranking.index.isin(self.long_tail)]) / len(ranking)
            lst = cold_pts if is_cold else non_cold_pts
            pts.append(pt)
            lst.append(pt)

        ndcgs = simulation.compute_scores()
        apt = np.mean(np.array(pts))
        apt_cold = np.mean(np.array(cold_pts))
        apt_non_cold = np.mean(np.array(non_cold_pts))
        ndcg = np.mean(ndcgs[np.nonzero(ndcgs)])
        return ndcg, apt, apt_cold, apt_non_cold

    def log_params(self) -> None:
        super().log_params()
        log_param("PS", self.PS)

    @classmethod
    def plot_run(cls, parent_run_id: str) -> None:
        """
        Plot a parent run of this evaluation in a plot with 2 subplots
        Parameters
        ----------
        parent_run_id: str
            Id of the parent run
        """
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        fig, (sub1, sub2) = plt.subplots(1, 2, figsize=(15, 5))
        metric_keys = {
            "avg_apt": r"$APT@10$",
            "avg_apt_cold": r"$APT_C@10$",
            "avg_apt_non_cold": r"$APT_M@10$",
            "avg_ndcg": r"$nDCG@10$",
        }
        # Load metrics by MlflowClient
        metrics = {
            name: {
                metric.step / 10: metric.value
                for metric in MlflowClient().get_metric_history(parent_run_id, key)
            }
            for key, name in metric_keys.items()
        }
        # Plot 1: only 1 y-axis, plot all metrics
        sub1.set_xlabel(r"$p$")
        sub1.set_ylabel("Value")
        sub1.set_title("(a)")
        for c, (key, values) in zip(colors, metrics.items()):
            xs = values.keys()
            ys = values.values()
            sub1.plot(xs, ys, label=key, color=c)

        # Plot 2: nDCG on left axis, APT on right axis
        ndcg = metrics[r"$nDCG@10$"]
        sub2.set_xlabel(r"$p$")
        sub2.set_ylabel(r"$nDCG@10$")
        sub2.set_title("(b)")
        sub2.plot(ndcg.keys(), ndcg.values(), label=r"$nDCG@10$", color=colors[-1])
        del metrics[r"$nDCG@10$"]
        sub2_2 = sub2.twinx()
        sub2_2.set_ylabel(r"$APT@10$")
        for c, (key, values) in zip(colors[:-1], metrics.items()):
            xs = values.keys()
            ys = values.values()
            sub2_2.plot(xs, ys, label=key, color=c)
        fig.legend(
            handles=[
                Line2D([0], [0], color=c, label=l)
                for c, l in zip(colors, metric_keys.values())
            ]
        )
        fig.suptitle(r"$APT@10$ and $nDCG@10$ for $p \in \{0.1, 0.2, \dots, 1.0\}$")
        fig.show()
        plt.savefig(f"../plots/figs/{parent_run_id}.png")
