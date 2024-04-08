import typing
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class _CrossValidationGenerator(BaseCrossValidator):
    """
    The CrossValidationGenerator provides a static create() function to create an
    instance of a ValidationGenerator from one of
    https://scikit-learn.org/stable/modules/cross_validation.html
    The create function consumes the type of the generator, and **kwargs.
    It creates creates a new class, and returns an instance of that class. the class:
    - Has the name Custom{GeneratorClass}
    - Has base classes GeneratorClass and BaseCrossValidator
    - Has the split() function of the GeneratorClass overridden to the custom
    implementation in _CrossValidationGenerator. This function uses the base split
    class, but excludes the items in the test set that do not occur in the train set.

    Note: this class must never be initialized

    See the Sklearn glossary:
    https://scikit-learn.org/stable/glossary.html#term-cross-validation-generator
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "The _CrossValidationGenerator may never be instantiated"
        )

    @typing.no_type_check
    def split(self, X, y, groups, items) -> Any:
        """
        After a train/test split, cold items may not occur in the test set. This is
        because these items wouldn't occur in the predictions of the matrix
        factorization, thus we cannot make predictions. Hence if items occur in the
        testset that don't occur at all in the training set, we drop those ratings.

        Use the base generator to split, then update ids by removing invalid train ids
        Parameters
        ----------
        X: The complete dataset to split, shape (n_features, n_items)
        y: The labels, shape (n_features,)
        groups: The groups, shape (n_items,).
        items: The item id for each row, shape (n_items,). Used to check which item
            ids exist in the train ids, and thus should remain in the test ids
        Returns
        -------
            Yields, like the split() of the generator, (train,test) tuples

        """

        for train, test in self.__class__.__base__.split(self, X, y, groups):
            # Valid testset items are the items that exist in train
            valid_items = np.unique(items[train])
            # Drop all items from test that are not valid
            test = test[np.flatnonzero(np.in1d(items[test], valid_items))]
            yield train, test


@typing.no_type_check
def create(generator_class, **kwargs) -> BaseCrossValidator:
    """
    Function should be used to create a generator that is safe to be used for
    BanditMF. This generator will have its split() function overriden by the one above
    Parameters
    ----------
    generator_class
        The sklearn generator class to user
    kwargs
        The parameters for the generator
    Returns
    -------
        Instance of class Custom{generator_class} with split() overridden

    """
    class_name = f"Custom{generator_class.__name__}"
    cls = type(
        class_name,
        (generator_class, BaseCrossValidator),
        {
            "split": _CrossValidationGenerator.split,
        },
    )
    return cls(**kwargs)
