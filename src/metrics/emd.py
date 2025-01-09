from typing import Union, Callable, Optional

import numpy as np
import ot
from scipy.spatial.distance import cdist


def construct_ground_distance(
    size: int,
    points: Union[None, np.ndarray] = None,
    metric: Union[str, Callable] = "euclidean"
) -> np.ndarray:
    """
    Constructs a ground distance matrix for Earth Mover's Distance.

    Parameters:
    ----------
    size : int
        The number of bins or points in the distribution.
    points : Union[None, np.ndarray], optional
        Coordinates of points in multidimensional space.
        If None, assumes a 1D histogram where indices represent bins.
        Shape should be (size, dimensions) for multidimensional data.
    metric : Union[str, Callable], optional
        The distance metric to compute ground distances.
        - For 1D: "manhattan", "euclidean", or other supported metrics.
        - For multidimensional: a metric supported by `scipy.spatial.distance.cdist`
          or a callable function for custom metrics.
        Default is "euclidean".

    Returns:
    -------
    np.ndarray
        A ground distance matrix of shape (size, size), where entry (i, j)
        represents the cost of moving mass between bin i and bin j.

    Raises:
    ------
    ValueError
        If `points` is not compatible with the given `size`.

    Example:
    --------
    # Example 1: 1D histogram with absolute difference
    >>> construct_ground_distance(3, metric="manhattan")
    array([[0., 1., 2.],
           [1., 0., 1.],
           [2., 1., 0.]])

    # Example 2: Multidimensional points with Euclidean distance
    >>> points = np.array([[0, 0], [1, 0], [2, 0]])
    >>> construct_ground_distance(size=3, points=points)
    array([[0., 1., 2.],
           [1., 0., 1.],
           [2., 1., 0.]])
    """
    if points is None:
        # Assume 1D histogram: bins are represented by integers 0 to size-1
        indices = np.arange(size).reshape(-1, 1)  # Shape (size, 1)
        distance_matrix = cdist(indices, indices, metric=metric)
    else:
        # Multidimensional points: Ensure `points` matches `size`
        if points.shape[0] != size:
            raise ValueError(
                f"Expected {size} points, but got {points.shape[0]}."
            )
        distance_matrix = cdist(points, points, metric=metric)

    return distance_matrix


def calculate_emd(
    source_distribution: np.ndarray,
    result_distribution: np.ndarray,
    target_index: Optional[int] = None,
    points: Optional[np.ndarray] = None,
    metric: str = "euclidean"
) -> float:
    """
    Calculates the Earth Mover's Distance (EMD) between two distributions
    using the POT library's `emd` method, with optional transfer direction correction.

    Parameters:
    ----------
    source_distribution : np.ndarray
        The first distribution (source), represented as a probability vector.
    result_distribution : np.ndarray
        The second distribution (result), represented as a probability vector.
        Both distributions must have the same size and sum to 1.
    target_index : Optional[int], optional
        The index of the target style in the distributions.
        If provided, used for transfer direction correction.
    points : np.ndarray, optional
        Coordinates of points for constructing the ground distance matrix.
        Shape should be (size, dimensions) if provided. If None, assumes 1D bins.
    metric : str, optional
        The distance metric for constructing the ground distance matrix.
        Default is "Euclidean".

    Returns:
    -------
    float
        The Earth Mover's Distance between the two distributions,
        with optional transfer direction correction applied.

    Raises:
    ------
    ValueError
        If the distributions are not of the same size or do not sum to 1.
        If `target_index` is out of bounds.

    Example:
    --------
    # Example 1: 1D histograms with direction correction
    >>> source_dist = np.array([0.4, 0.6, 0.0])
    >>> result_dist = np.array([0.2, 0.3, 0.5])
    >>> calculate_emd(source_dist, result_dist, target_index=2, metric="manhattan")
    -0.5

    # Example 2: 1D histograms without direction correction
    >>> calculate_emd(source_dist, result_dist, metric="manhattan")
    0.5
    """
    # Validate input distributions
    if len(source_distribution) != len(result_distribution):
        raise ValueError("Distributions must have the same size.")
    if not np.isclose(source_distribution.sum(), 1) or not np.isclose(result_distribution.sum(), 1):
        print(source_distribution.sum())
        print(result_distribution.sum())
        raise ValueError("Distributions must sum to 1.")

    if target_index is not None:
        if target_index < 0 or target_index >= len(source_distribution):
            raise ValueError(f"Target index {target_index} is out of bounds.")

    # Deduce the size of the distributions
    size = len(source_distribution)

    # Construct the ground distance matrix
    ground_distance = construct_ground_distance(size=size, points=points, metric=metric)

    # Compute the EMD
    emd_value = ot.emd2(source_distribution, result_distribution, ground_distance)

    # Apply transfer direction correction if target_index is provided
    if target_index is not None and result_distribution[target_index] < source_distribution[target_index]:
        emd_value = -emd_value

    return emd_value
