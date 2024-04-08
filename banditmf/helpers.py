import typing
from datetime import datetime
from os.path import exists
from pickle import HIGHEST_PROTOCOL, dump
from typing import Any, Callable, Dict, List, Optional, Tuple, no_type_check

import numpy as np
import pandas as pd
from mlflow import log_artifact
from numpy.typing import NDArray
from sklearn.metrics import ndcg_score
from sqlalchemy import Numeric
from structlog import PrintLogger, get_logger

"""
Helper functionality and variables

Attributes
----------
IS_DEVELOPMENT : bool
    Should be true while developing the algorithm.
    Is used to save and read data from filesystem, to prevent recomputing on every run
"""
IS_DEVELOPMENT = False


def save_to_fs(filename: str, data: pd.DataFrame) -> bool:
    """
    Save a dataframe to the Filesystem, if IS_DEVELOPMENT

    Parameters
    ----------
    filename: str
        The filename to save to
    data: pd.DataFrame
        The frame tto save

    Returns
    -------
    bool
        True if saved, false otherwise

    """
    if IS_DEVELOPMENT:
        data.to_pickle(f"..\\temp\\{filename}.pkl")
    return IS_DEVELOPMENT


def read_from_fs(filename: str, instance: object, variable: str) -> bool:
    """
    Read a pd.DataFrame from filesystem into an attribute of an instance

    Parameters
    ----------
    filename: str
        The filename to read from
    instance: object
        The instance to save to
    variable: str
        The attribute name to save to

    Returns
    -------
    bool
        True if read, false otherwise

    """
    loc = f"..\\temp\\{filename}.pkl"
    if not IS_DEVELOPMENT or not exists(loc):
        return False
    data = pd.read_pickle(loc)
    setattr(instance, variable, data)
    return True


def frame_to_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a melted ratings frame into a matrix.
    A melted frame has one row for each rating, columns user item rating.
    The matrix is a user x item matrix, cell values are ratings
    """
    return frame.pivot(index="user", columns="item", values="rating")


def matrix_to_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Inverse of frame_to_matrix:
    Convert a user x items matrix into a melted frame: one row for each rating, columns
    user,item,rating.
    """
    melted = matrix.melt(ignore_index=False, value_name="rating")
    # The index contains the user ids, convert index to column 'user' and reset index
    melted["user"] = melted.index
    melted.index.name = None
    melted = melted[melted["rating"].notna()]
    return melted


def X_y_to_frame(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Merge X,y into a three column frame: user,item,rating
    - Drop all but the first two columns of X
    - Merge y into X by appending as col
    - Drop any double user,item combinations
    Parameters
    ----------
    X: Features. All but the first two columns will be dropped, shape (n_features,
    n_items)
    y: Values. Should be the ratings, shape (n_features,)

    Returns
    -------
    pd.DataFrame shape (3, n_item): The melted frame, one row per rating

    """
    X = X.iloc[:, :2]  # type: ignore
    frame = X
    frame.insert(2, "rating", y)
    frame = frame.drop_duplicates(subset=["user", "item"])
    return frame


def X_y_to_matrix(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    X_y_to_frame, but frame converted to matrix
    Parameters
    ----------
    X: Features. All but the first two columns will be dropped, shape (n_features,
    n_items)
    y: Values. Should be the ratings, shape (n_features,)

    Returns
    -------
    pd.DataFrame shape (n_user, n_items) values are ratings or NaN
    """
    frame = X_y_to_frame(X, y)
    matrix = frame_to_matrix(frame)
    return matrix


def running_average(x: NDArray[Numeric], n: int) -> NDArray[Numeric]:
    """
    Compute the running average over a list of values
    Parameters
    ----------
    x: NDArray[Numeric]
        The list of values
    n: int
        The windows size
    Returns
    -------
    NDArray[Numeric]
        Running average of x, of length len(x) - n
    """
    return np.convolve(x, np.ones(n) / n, mode="valid")


def keys_of_max_val(d: Dict[Any, Numeric]) -> List[int]:
    """
    Get the keys that have the max value in a dict
    """
    v = np.array(list(d.values()))
    k = np.array(list(d.keys()))
    return k[np.where(v == max(v))]


def object_to_mlflow(obj: Any, name: str) -> None:
    """
    Save an object to an artifact by:
    - Saving the object to a temp pickle file
    - Saving the temp pickle file as artifact in the current mlfow run
    Parameters
    ----------
    obj: Any
        The dict to save
    name: str
        The artefact name
    """
    with open("../temp/files/artifact.pickle", "wb") as handle:
        dump(obj, handle, protocol=HIGHEST_PROTOCOL)
    log_artifact("../temp/files/artifact.pickle", name)


@typing.no_type_check
def interact_with_fs(variables: List) -> Callable:
    """
    Decorator for a function that interacts with fs. Before running the func:
    - Check if we can read each of the variables from filesystem
    - If true, we have set the data on self, and thus we do not run fun() but return
    - If false, we run func(), which must set all the variables, and now we save the
    variables to fs for the next run
    Parameters
    ----------
    variables: List
        List of tuples (filename, variable): The filename to read/write, the variable
        to read/write in the class

    Returns
    -------
        Decorator that consumes the function to decorate. Note that any return value
        of func will not be returned when decorated, since we won't run func in case
        the variables can be read from fs.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for (filename, variable) in variables:
                if not read_from_fs(filename, self, variable):
                    break
            else:  # No break: all variables found, return
                return
            # Break: Some variables not found, run func and save to fs
            func(self, *args, **kwargs)
            for (filename, variable) in variables:
                save_to_fs(filename, getattr(self, variable))

        return wrapper

    return decorator


@no_type_check
def validate_parameters(func):
    """
    Decorator function for Algorithm objects. run self.validate_params() before the
    function is ran
    """

    def wrapper(self, *args, **kwargs):
        self.validate_parameters()
        return func(self, *args, **kwargs)

    return wrapper


def set_ids_to_subsets(
    X: pd.DataFrame, y: pd.Series, tr_ids: list, te_ids: list
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """
    Convert train,test ids to train test sets
    Parameters
    ----------
    X: Features
    y: labels
    tr_ids: list of ids for training
    te_ids: List of ids for testing
    Returns
    -------
        Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
        (X_train, y_train),(X_test, y_test)

    """
    return (X.iloc[tr_ids], y.iloc[tr_ids]), (X.iloc[te_ids], y.iloc[te_ids])


def create_logger(total_steps: Optional[int] = None) -> PrintLogger:
    """
    Create a structlog logger, it saves the total steps and creation time
    Returns
    -------
    PrintLogger
    """
    return get_logger(created_at=datetime.now(), total_steps=total_steps)


def log(logger: PrintLogger, msg: str, step: Optional[int] = None) -> None:
    """
    Log a msg using a logger. Add eta if 'step' is provided and logger was created
    via create_logger
    """
    created_at = logger._initial_values.get("created_at")  # type: ignore
    total_steps = logger._initial_values.get("total_steps")  # type: ignore
    eta = None
    if step is not None and created_at is not None and total_steps is not None:
        time_exceeded = datetime.now() - created_at
        time_per_step = time_exceeded / step
        time_remaining = (total_steps - step) * time_per_step
        eta = (datetime.now() + time_remaining).isoformat()
    logger.msg(msg, eta=eta, step=step)  # type: ignore


def average_ndcgs(
    y: pd.DataFrame, predictions: pd.DataFrame, ks: List[int]
) -> NDArray[float]:
    """
    Compute ndcg scores: for each user compute ndcg@k for each k. Average over all user
    Parameters
    ----------
    y: pd.DataFrame
        Actual ratings. columns user,item,rating
    predictions: pd.DataFrame
        Predictions. columns user,item,rating. Same len as y
    ks: List[int]
        Ks to compute ndcg@k for
    Returns
    -------
    NDarray[float]
        ndcg@k for each k in k

    """
    # Sort both frames. Now they are of equal length and equally ordered
    y.sort_values(["user", "item"], inplace=True)
    predictions.sort_values(["user", "item"], inplace=True)
    # Double list: one row per user, for each user each ndcg@k
    scores = []
    grouped = y.groupby("user").indices
    for u, indices in grouped.items():
        user_scores = []
        y_true = y.iloc[indices].rating
        y_pred = predictions.iloc[indices].prediction
        for k in ks:
            ndcg = ndcg_score([y_true], [y_pred], k=k)
            user_scores.append(ndcg)
        scores.append(user_scores)
    # Avg for each user: shape (len(self.K_VALUES))
    avg_scores = np.mean(scores, axis=0)
    return avg_scores
