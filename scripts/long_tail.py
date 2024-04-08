import os
from math import ceil
from typing import Tuple

import pandas as pd
from lenskit.datasets import MovieLens
from matplotlib import pyplot as plt

from helpers import frame_to_matrix

"""
Script to plot the long tail structure of Movielens.

This script is used in the thesis:
- Section 4.5, Figure 4.10.
"""

# Load frame
path = os.path.join(os.path.dirname(__file__), "../data/ml-latest-small")
movielens = MovieLens(path).ratings  # type: pd.DataFrame

# Alternatively: Load 1M frame
# path = os.path.join(os.path.dirname(__file__), "../data/ml-1m")
# movielens = ML1M(path).ratings  # type: pd.DataFrame

matrix = frame_to_matrix(movielens)


def plot_long_tail(matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ "
    Show a matplotlib plot with the long tail separation.
    Parameters
    ----------
    matrix: pd.DataFrame
        User x Item materixz

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        short_head, long_tail
    """
    # One row for each item, add column count and column avg rating
    counts = matrix.count(axis=0).to_frame("count")
    counts["avg_rating"] = matrix.mean(axis=0)
    # Sort on count, reset index
    frame = counts.sort_values("count", ascending=False).reset_index()

    def plot_fraction_line(
        frame: pd.DataFrame, subset: pd.DataFrame, color: str, style: str
    ) -> None:
        """
        Add the vertical fraction line to a matplotlib plot
        Parameters
        ----------
        frame: pd.DataFrame
            All items
        subset: pd.DataFrmae
            Subset
        color: str
            Coloor of the line
        style: str
            Style of the line
        """
        fraction = len(subset) / len(frame)
        n_ratings = int(subset["count"].sum() / frame["count"].sum() * 100)
        plt.axvline(
            x=len(frame) * fraction,
            ymin=0.03,
            ymax=0.97,
            label=f"{int(fraction * 100)}% of items, {n_ratings}% of ratings",
            color=color,
            linestyle=style,
        )

    # Plot long tail line
    frame.plot(y="count", use_index=True)

    shortest_head = frame.head(n=ceil(0.05 * len(frame)))
    plot_fraction_line(frame, shortest_head, "c", "-.")

    short_head = frame.head(n=ceil(0.2 * len(frame)))
    plot_fraction_line(frame, short_head, "red", "--")

    long_tail = frame[~frame["item"].isin(short_head["item"])]
    plt.xlabel("Item")
    plt.ylabel("#Ratings")
    plt.title("The long tail of the Movielens Latest dataset")
    plt.legend()
    plt.show()
    return short_head, long_tail


short_head, long_tail = plot_long_tail(matrix)
test = ""
