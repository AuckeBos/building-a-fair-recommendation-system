import os
from math import ceil
from typing import Optional, Tuple

import pandas as pd
from lenskit.datasets import ML1M, MovieLens
from sklearn.model_selection import GroupShuffleSplit


class DataLoader:
    """The DataLoader is responsible for loading the data
    For now, can only load the movielens data

    Attributes
    ----------
    _data: pd.DataFrame
        The complete set of data, as read from FS. Should contain one row per rating,
        at least columns user,item,rating

    train: Tuple[pd.DataFrame, pd.Series]
        X,y train data. Initialized on init()

    test: Tuple[pd.DataFrame, pd.Series]
        X,y test data. Initialized on init()

    random_state: int
        Provided through init(), for reproducibility

    TEST_SIZE: float
        Test size in [0, 1.0]. Defines the percentage of *users* to be selected as test
        set. All ratings of these users are selected as test set, the remaining
        ratings define the train set
    """

    _data: pd.DataFrame
    train: Tuple[pd.DataFrame, pd.Series]
    test: Tuple[pd.DataFrame, pd.Series]
    random_state: Optional[int]
    TEST_SIZE = 0.1

    def __init__(self, data: pd.DataFrame, random_state: Optional[int] = None):
        """
        Initialize with the data. Should only be called from within this class,
        from abstract methods that export specific data loadings, for example
        movielens(). Save data and GroupSplit in train, test

        Parameters
        ----------
        data
            The complete dataset, as read from FS
        """
        self._data = data
        self.random_state = random_state
        self._train_test_split()

    @staticmethod
    def movielens(random_state: Optional[int] = None, dataset_type: str = None) -> "DataLoader":
        """
        Load the movielens dataset

        Parameters
        ----------
        random_state: int
            Seed for split
        dataset_type: str
            Type of movielens. Kan be 'latest' or '1M'. Default is 'latest'
        Returns
        -------
        DataLoader
            DataLoader initialized with movielens data

        """
        dataset_type = dataset_type or "latest"
        if dataset_type == "latest":
            path = os.path.join(os.path.dirname(__file__), "../../data/ml-latest-small")
            data = MovieLens(path)
        elif dataset_type == "1M":
            path = os.path.join(os.path.dirname(__file__), "../../data/ml-1m")
            data = ML1M(path)
        else:
            raise ValueError(f"Movielens {dataset_type} not supported")
        ratings = data.ratings
        ratings.drop("timestamp", axis=1, inplace=True)
        return DataLoader(ratings, random_state)

    @classmethod
    def get_categories(cls, matrix: pd.DataFrame, type: str = None) -> pd.Series:
        """
        Get the category for each item in the matrix. The category is used for
        fairness computation
        Parameters
        ----------
        matrix: User x Item rating matrix (includes train and test)
        type: The type of categorization. Supported:
        - 'popularity', uses _get_popularity_categories

        Returns
        -------
            pd.Series:
                Of length n_items, value is category of the item

        """
        if type == "popularity":
            return cls._get_popularity_categories(matrix)
        else:
            raise ValueError(f"Invalid type {type} for categorization")

    @classmethod
    def _get_popularity_categories(cls, matrix: pd.DataFrame) -> pd.Series:
        """
        The popularity categorization assigns 1 of two categories to each item:
        - 1: short_head_items: These are all items that occur in the top 20% when
         the items are ranked based on popularity. Popularity is measured in number
         of ratings in the matrix. These items contain ~80% of all ratings, and are
         advantaged
        - 0: long_tail_items. These are all items that are in the long tail, eg all
            items that are not in the short head. These are the bottom 80% of the
            items; they contain only 20% of the ratings: they are disadvantaged
        Parameters
        ----------
        matrix: pd.DataFrame
            User x Item matrix

        Returns
        -------
            pd.Series
                Of len n_items, each value is 0 or 1
        """
        counts = matrix.count(axis=0).to_frame("count")
        # Sort on count, reset index
        frame = counts.sort_values("count", ascending=False)
        # Top 20% of items sorted on n_ratings: short head
        short_head_items = frame.head(n=ceil(0.2 * len(frame))).index
        # Category is 1 if item is in short_head, else 0
        frame["category"] = frame.index.isin(short_head_items).astype(int)
        return frame["category"]

    @staticmethod
    def _drop_invalid_items(
        train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        After a train/test split, cold items may not occur in the test set. This is
        because these items wouldn't occur in the predictions of the matrix
        factorization, thus we cannot make predictions. Hence if items occur in the
        testset that don't occur at all in the training set, we drop those ratings.
        Parameters
        ----------
        train: pd.DataFrame
            List of train ratings, columns user,item,rating
        test: pd.DataFrame
            List of test ratings, columns user,item,rating
        Returns
        -------
        (train, test) with invalid items drop from test
        """
        valid_items = train["item"].unique()
        test = test[test["item"].isin(valid_items)]
        return train, test

    def _train_test_split(self) -> None:
        """
        Train test split self._data:
        Use GroupShuffleSplit, eg group on user, and select self.TEST_SIZE percent
        of users out of the complete data. Then select all items from those user from
        the data as testset, rest is trainingset.

        Note that invalid items are dropped, see _drop_invalid_items
        """
        splitter = GroupShuffleSplit(
            test_size=self.TEST_SIZE, n_splits=1, random_state=self.random_state
        )
        train_ids, test_ids = next(
            splitter.split(self._data, groups=self._data["user"])
        )
        train = self._data.iloc[train_ids]
        test = self._data.iloc[test_ids]
        train, test = self._drop_invalid_items(train, test)
        # Convert frames to X,y. X is first 2 cols, y is last col
        self.train = train.iloc[:, :2], train.iloc[:, -1]
        self.test = test.iloc[:, :2], test.iloc[:, -1]
