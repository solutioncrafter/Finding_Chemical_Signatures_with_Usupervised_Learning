"""
This module provides functions to evaluate and visualize the performance of
Non-negative Matrix Factorization (NMF) models with varying numbers of
components.

It includes the following functionalities:

1. Creating an NMF pipeline with a specified number of components.
2. Calculating the norms of the residuals for different numbers of components.
3. Plotting the evaluation curve for the number of NMF components.
4. Evaluating the NMF model and plotting the evaluation curve.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline


def create_pipeline(n_components: int) -> Pipeline:
    """
    Create an NMF pipeline with the specified number of components.

    Parameters:
    n_components (int): Number of components for NMF.

    Returns:
    sklearn.pipeline.Pipeline: The NMF pipeline.
    """
    return Pipeline([
        ('nmf', NMF(
            n_components=n_components,
            init='nndsvd',
            solver='cd',
            beta_loss='frobenius',
            tol=1e-4,
            random_state=0,
            max_iter=100000,
            shuffle=True))
    ])


def calculate_residuals(data: np.ndarray,
                        n_components_range: range) -> list[float]:
    """
    Calculate the norms of the residuals for different numbers of components.

    Parameters:
    data (numpy.ndarray): The input data.
    n_components_range (range): The range of component numbers to evaluate.

    Returns:
    list: The norms of the residuals for each number of components.
    """
    residuals_norm = []

    for n_components in n_components_range:
        pipeline = create_pipeline(n_components)
        pipeline.fit(data)

        W = pipeline.named_steps['nmf'].transform(data)
        H = pipeline.named_steps['nmf'].components_
        spectra_reconstructed = np.dot(W, H)
        residuals = data - spectra_reconstructed
        norm_residuals = np.linalg.norm(residuals)

        residuals_norm.append(norm_residuals)

    return residuals_norm


def plot_nmf_component_evaluation_curve(n_components_range: range,
                                        residuals_norm: list[float]) -> None:
    """
    Plot the evaluation curve for the number of NMF components.

    Parameters:
    n_components_range (range): The range of component numbers evaluated.
    residuals_norm (list): The norms of the residuals for each
    number of components.
    """
    plt.figure(figsize=(8, 3))
    plt.plot(n_components_range, residuals_norm, marker='o', linestyle='--',
             color='b')
    plt.title('Norm of Residuals vs Number of Components in NMF')
    plt.xlabel('Number of Components')
    plt.ylabel('Norm of Residuals')
    plt.xticks(n_components_range)
    plt.grid(True)
    plt.axvline(x=3, color='red', linestyle='--', linewidth=1)
    plt.show()


def evaluate_and_plot_nmf_component_number(data: np.ndarray,
                                           n_components_range: range) -> None:
    """
    Evaluate the NMF model and plot the evaluation curve.

    Parameters:
    data (numpy.ndarray): The input data.
    n_components_range (range): The range of component numbers to evaluate.
    """
    residuals_norm = calculate_residuals(data, n_components_range)
    plot_nmf_component_evaluation_curve(n_components_range, residuals_norm)
