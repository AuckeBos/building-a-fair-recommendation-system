import functools
from abc import abstractmethod
from typing import Callable

import mlflow


class BaseEvaluator:
    """
    An evaluator evaluates a certain behaviour of BanditMF.
    The BaseEvaluator provides functionality used in many evaluators.
    
    Note that although strictly not allowed, some evaluators use private properties 
    of some models, because they need access to attributes for logging purposes
    
    Attributes
    ----------
    run_id: str
        Parent mlflow run id. Set in the register_mlflow decorator

    RUN_COUNT: int
        Most evaluators run k-fold cross-validation. This variable defines K
    USER_COUNT: int
        If the evaluator runs k-fold cv, the testset consists of all ratings of 
        USER_COUNT users
    RANDOM_STATE: int
        Seed, for reproducibility
    SHUFFLE: bool
        If the evaluator uses a Simulation (most of them do), the stream of that 
        simulation is either shuffled or sorted
            
    """
    RUN_COUNT: int
    USER_COUNT: int
    RANDOM_STATE = 1100
    SHUFFLE = False
    run_id: str

    @abstractmethod
    def evaluate(self) -> None:
        pass

    # noinspection Mypy
    @staticmethod
    def register_mlflow(func: Callable) -> Callable:
        """
        Decorator function. Sets the current experiment based on the class name.
        Starts a run before running the function, ends it after
        Parameters
        ----------
        func: callable
            The function to run
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):  # type: ignore
            mlflow.set_experiment(self.__class__.__name__)
            with mlflow.start_run(run_name="parent") as run:
                self.run_id = run.info.run_id
                return func(self, *args, **kwargs)

        return wrapper

    def log_params(self) -> None:
        """
        Evaluators log params to mlflow. Log the base params here, implementations
        should call super() and add own parameters
        """
        mlflow.log_params(
            {
                "Num splits": self.RUN_COUNT,
                "Num test users": self.USER_COUNT,
                "seed": self.RANDOM_STATE,
                "shuffle": self.SHUFFLE,
            }
        )
