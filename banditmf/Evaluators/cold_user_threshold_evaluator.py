from typing import Tuple

import numpy as np
import pandas as pd
from Components.CandidateSelectors.testset_candidate_selector import (
    TestsetCandidateSelector,
)
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Environments.batch_simulation import BatchSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, set_ids_to_subsets
from sklearn.model_selection import GroupShuffleSplit


class ColdUserThresholdEvaluator(BaseEvaluator):
    """
    The ColdUserThresholdEvaluator is used to investigate the percentage of users
    that switches from cold to non cold within one testset.

    It uses leave 1 out RUN_COUNT times. Hence each split has a testset that contains
    all ratings for 1 user. The stream is iterated, and the switch moment is saved.
    The results are printed to stdout

    The results of this evaluation are not used in the thesis, but rather used for
    manual investigating behavior. They were used to decide upon the mechanism for
    certainty_achieved
    """

    RUN_COUNT = 1
    USER_COUNT = 1

    def evaluate(self) -> None:
        result = {}
        # Use Leave one Out RUN_COUNT times
        generator = create_generator(
            GroupShuffleSplit, n_splits=self.RUN_COUNT, test_size=self.USER_COUNT
        )
        X, y = DataLoader.movielens().train
        splits = generator.split(X, y, X.user, X.item.to_numpy())
        for train, test in splits:
            user, sub_result = self.evaluate_one_split(
                *set_ids_to_subsets(X, y, train, test)
            )
            if sub_result["certainty_achieved"]:
                print(
                    f"Certain at timestep {sub_result['certain_at']}/"
                    f"{sub_result['num_records']}: User {user} belongs in cluster "
                    f"{sub_result['cluster']}"
                )
            else:
                print(
                    f"No certainty achieved within {sub_result['certain_at']} for user "
                    f"{user}"
                )
            result[user] = sub_result
        num_certain = len([1 for x in result.values() if x["certainty_achieved"]])
        print(f"Certain for {num_certain} of {self.RUN_COUNT} users")
        avg_certain_at = np.average(  # type: ignore
            [x["certain_at"] for x in result.values() if x["certainty_achieved"]]
        )
        avg_len_certains = np.average(  # type: ignore
            [x["num_records"] for x in result.values() if x["certainty_achieved"]]
        )
        print(f"On average, certain at {avg_certain_at} / {avg_len_certains}")

        avg_len_unclustered = np.average(  # type: ignore
            [x["certain_at"] for x in result.values() if not x["certainty_achieved"]]
        )

        print(f"Average len of unclustered users: {avg_len_unclustered}")

    def evaluate_one_split(
        self,
        train: Tuple[pd.DataFrame, pd.Series],
        test: Tuple[pd.DataFrame, pd.Series],
    ) -> Tuple[int, dict]:
        """
        Evaluate one split
        Returns
        -------
        Log data as dict:
        - certainty_achieved: true if certainty achieved
        - certain_at: Moment of certainty achieved, or len(testset)
        - num_records: len(testset)
        """
        result = {
            "certainty_achieved": False,
            "certain_at": len(test[0]),
            "num_records": len(test[0]),
        }

        simulator = (
            BatchSimulation(train)
            .set_candidate_selector(
                TestsetCandidateSelector().fit(X_y_to_matrix(*test))
            )
            .fit(*test, shuffle=False)
        )
        for i, user in enumerate(simulator.simulate()):
            mab = simulator.recommender._casino._multi_armed_bandits[user]
            if mab.certainty_achieved():
                result["certain_at"] = i
                result["cluster"] = mab.latest_favorite_arm()
                result["certainty_achieved"] = True
                break
        user = test[0].iloc[0].user
        return user, result
