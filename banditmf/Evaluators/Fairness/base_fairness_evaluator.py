from abc import abstractmethod
from typing import Any, List, Tuple

import pandas as pd
from Components.casino import Casino
from Components.Rankers.base_ranker import BaseRanker
from DataLoaders.cross_validation_generator import create as create_generator
from DataLoaders.data_loader import DataLoader
from Environments.prequential_simulation import PrequentialSimulation
from Evaluators.base_evaluator import BaseEvaluator
from helpers import X_y_to_matrix, set_ids_to_subsets
from mlflow import log_params
from sklearn.model_selection import GroupShuffleSplit


class BaseFairnessEvaluator(BaseEvaluator):
    """
    Base class for fairness evaluators, provides functionality used in all evaluators
    
    Attributes
    ----------        
    X: pd.DataFrame
        Unsplit train data
    y: pd.Series
        Labels of X
    splits:  List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
        List of train,test splits. Do not contain the indices, but the actual data        
    long_tail: pd.Index
        Items in the long tail, eg the bottom 80% unpopular items. Set using
         DataLoader.get_categories
    
    ranker: Ranker
        The ranker to use. Can be either a DefaultRanker or a FairRanker

    HYPER_PARAMETERS: Dict[str, Any]:
        Hyperparameters to use for BanditMF
    K: int
        the k for which we compute ndcg@k and APT@k at each timestep
    """
    X: pd.DataFrame
    y: pd.Series
    splits: List[Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]]
    long_tail: pd.Index
    ranker: BaseRanker

    HYPER_PARAMETERS = {
        "certainty_window": 3 * 2,
        "refit_at": 50,
        "num_clusters": 2,
    }

    K = 10

    def load_data(
        self,
    ) -> None:
        """
        Load the data used in every fairness evaluator:
        - Set self.X and self.y based on the Dataloader
        - Set self.long_tail to the bottom 80% of the items
        - Set self.splits
        """
        self.X, self.y = DataLoader.movielens(random_state=self.RANDOM_STATE).train
        # Save long tail
        categories = DataLoader.get_categories(
            X_y_to_matrix(self.X, self.y), "popularity"
        )
        self.long_tail = categories[categories.values == 0].index
        generator = create_generator(
            GroupShuffleSplit,
            n_splits=self.RUN_COUNT,
            test_size=self.USER_COUNT,
            random_state=self.RANDOM_STATE,
        )
        self.splits = [
            set_ids_to_subsets(self.X, self.y, tr_idx, te_idx)
            for tr_idx, te_idx in generator.split(
                self.X, self.y, self.X.user, self.X.item.to_numpy()
            )
        ]

    @abstractmethod
    def log_run(self, simulation: PrequentialSimulation) -> Any:
        """
        Log a ran simulation to mlflow. Return the data that must be aggregated to be
        logged to the parent run
        Parameters
        ----------
        simulation: The ran simulation

        Returns
        -------
            Any

        """
        pass

    @classmethod
    def plot_run(cls, run_id: str) -> None:
        """
        Create matplotlib plots for this run.
        Parameters
        ----------
        run_id: str
            The id of the run. Might be parent run id or child run id, depending on
            implementation
        """
        pass

    def pt(self, ranking: pd.Series) -> float:
        """
        Compute the PT for a ranking
        """
        return len(ranking[ranking.index.isin(self.long_tail)]) / len(ranking)

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
