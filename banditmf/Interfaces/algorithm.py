from sklearn.base import BaseEstimator


class Algorithm(BaseEstimator):
    """
    Base class for Components that consume hyperparameters in their __init__

    Offers a validate_parameters function to set the parameters to their default
    values if they are not provided via __init__, based on the naming convention

    "The default name for param 'param_name' is 'DEFAULT_PARAM_NAME'"
    """

    def validate_parameters(self) -> None:
        """
        Validate the parameters of an Algorithm. For now:
        - For each param name in get_param_names, check if a default value exists. If
        so, set it if the param was not set in init().

        The default name for param 'param_name' is 'DEFAULT_PARAM_NAME'
        """
        param_names = self._get_param_names()
        for param_name in param_names:
            default_name = f"DEFAULT_{param_name.upper()}"
            # If self.param_name is None, set its value to the default value
            setattr(
                self,
                param_name,
                getattr(self, param_name) or getattr(self, default_name, None),
            )
