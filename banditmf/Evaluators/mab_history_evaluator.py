from collections import defaultdict
from typing import Iterator, Tuple, ValuesView

import pandas as pd
from beautifultable import BeautifulTable
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from Components.history_aware_mab import HistoryAwareMab
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Environments.batch_simulation import BatchSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, running_average, set_ids_to_subsets
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit


class MabHistoryEvaluator(BaseEvaluator):
    """
    The MabHistoryEvaluator is used to investigate the history of the MABs of the
    casino. It uses the history that is saved in the HistoryAwareMAB class to create
    matplotlib plots.

    To create the plots to evaluate the evaluator does the following:
    - Create a train test split, the testset contains USER_COUNT users
    - While the stream is not empty, ask recommendations of BanditMF, forcing it to
    use the Casino, eg it sets enforce_is_cold to True
    - Compute and update reward, save state
    - After each user has been fully iterated, show plot of mab history, showing UCB,
    means, and other stats over time
    - During the evaluation, print table of recommendation

    The results of this evaluation are not used in the thesis, but rather used for
    manual investigating behavior. They were used to decide upon the mechanism for
    certainty_achieved

    Attributes
    ----------
    HYPER_PARAMETERS: Dict[str, Any]:
        Hyperparameters to use for BanditMF
    """

    HYPER_PARAMETERS = {
        "certainty_window": 3 * 2,
        "refit_at": 50,
        "num_clusters": 2,
    }

    USER_COUNT = 1

    def evaluate(self) -> None:
        X, y = DataLoader.movielens().train
        generator = create_generator(
            GroupShuffleSplit, n_splits=1, test_size=self.USER_COUNT
        )
        train, test = next(generator.split(X, y, X.user, X.item.to_numpy()))
        self._evaluate(*set_ids_to_subsets(X, y, train, test))

    def _evaluate(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> None:
        """
        Run _show_recommendation_history while printing the results in a table
        Parameters
        ----------
        train:  Tuple[pd.DataFrame, pd.Series]
            X,y train
        test:  Tuple[pd.DataFrame, pd.Series],
            X,y test
        """
        table = BeautifulTable()
        table.columns.header = [
            "User",
            "Cluster",
            "Item     ",
            "Prediction  ",
            "Actual",
            "Reward",
            "Recommender        ",
        ]
        for line in table.stream(self._show_recommendation_history(train, test)):
            print(line)

    def _show_recommendation_history(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> Iterator[ValuesView]:
        """
        Run the evaluator Eg request recommendations, save history, splot results
        Parameters
        ----------
        train:  Tuple[pd.DataFrame, pd.Series]
            X,y train
        test:  Tuple[pd.DataFrame, pd.Series],
            X,y test


        """
        test_matrix = X_y_to_matrix(*test)
        simulator = (
            BatchSimulation(train)
            .set_enforce_is_cold(True)
            .set_candidate_selector(TestsetCandidateSelector().fit(test_matrix))
            .fit(*test, shuffle=False, log_to_mlflow=False)
        )
        for _ in simulator.simulate():
            user, item, prediction = simulator._recommendations[-1]
            rating = test_matrix.loc[user, item]
            reward = simulator.rewarder.get_reward(user, item)
            log_data = {
                "user": user,
                "cluster": simulator.recommender._casino._multi_armed_bandits[
                    user
                ].latest_pull,
                "item": item,
                "prediction": prediction,
                "actual": rating,
                "reward": reward,
                "recommender": simulator.recommender._recommender_types[user],
            }
            # If next item is new user, plot history of current user
            if simulator.finished or simulator.stream[0][0] != user:
                self._plot_mab_history(
                    user, simulator.recommender._casino._multi_armed_bandits[user]
                )
            yield log_data.values()

    def _plot_mab_history(self, user: int, mab: HistoryAwareMab) -> None:
        """
        Plot the history of an MAB
        Plot the history of the arms of the MAB of one user
        Parameters
        ----------
        user: int
            The user
        mab HistoryAwareMab:
            The mab that has the history stored
        """
        arms = mab.arms
        history = mab.state_history
        data_to_plot: dict = {}

        # At each timestep, for each arm, compute its #pulls / total pulls
        pulls_div_total_diff = defaultdict(lambda: [0])  # type: ignore

        for i, policy in enumerate(mab.state_history[1:]):
            for arm in arms:
                now = policy.arm_to_count[arm] / policy.total_count
                prev = history[i - 1].arm_to_count[arm] / history[i - 1].total_count

                pulls_div_total_diff[arm].append(abs(now - prev) / now)
        # For each arm, compute running average of pulls_div_total_diff
        N = self.HYPER_PARAMETERS["certainty_window"]
        averages_pulls_div_total_diff = {}
        for arm in arms:
            averages_pulls_div_total_diff[arm] = [0] * (N - 1)  # type: ignore
            averages_pulls_div_total_diff[arm].extend(
                running_average(pulls_div_total_diff[arm], N)  # type: ignore
            )

        # Initialize all the lineplots
        for arm in arms:
            data_to_plot[f"UCB arm {arm}"] = []
            data_to_plot[f"Pulls / total arm {arm}"] = []
            data_to_plot[
                f"Running {N} avg Pulls / total arm {arm}"
            ] = averages_pulls_div_total_diff[arm]
        x = []

        for i, policy in enumerate(history):
            x.append(i)
            for arm in arms:
                data_to_plot[f"UCB arm {arm}"].append(policy.arm_to_expectation[arm])
                data_to_plot[f"Pulls / total arm {arm}"].append(
                    policy.arm_to_count[arm] / policy.total_count
                )

        # Plot all lines
        plt.figure(dpi=250)
        for name, values in data_to_plot.items():
            plt.plot(x, values, label=name)
        plt.legend()
        plt.title(f"MAB history for user {user}")
        plt.xlabel(r"$t$")
        plt.show()
        # Compute and print when the certainty threshold was reached
        for i in range(len(history)):
            if mab.certainty_achieved(i):
                cluster = mab.favorite_arms[i]
                print(f"Certainty achieved at timestep {i}: cluster is {cluster}")
                break
