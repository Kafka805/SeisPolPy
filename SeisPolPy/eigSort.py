# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:55:18 2024

@author: Austin Abreu
"""
import numpy as np
import numpy.typing as npt


def eig(vec1, vec2, vec3) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    # Combine the arrays into a 3xN matrix
    A = np.vstack((vec1, vec2, vec3))

    # Calculate the covariance matrix
    CM = np.cov(A)

    # Calculate the eigenvalues and eigenvectors
    eVals, eVecs = np.linalg.eig(CM)

    return eVals, eVecs


def sort(vals, vecs) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    # Retrieve the sorting indices from the eigenvalue vector
    sorted_indices = np.argsort(vals)[::-1]

    # Use the indices to sort the matrices
    sorted_vals = vals[sorted_indices]
    sorted_vecs = vecs[:, sorted_indices]

    return sorted_vals, sorted_vecs


def eigSort(
    arr1: npt.ArrayLike, arr2: npt.ArrayLike, arr3: npt.ArrayLike
) -> tuple[npt.ArrayLike]:
    """
    Solves for eigenvalues and eigenvectors by using the covariance matrix
    Decomposition method. Then, sorts the eigenvalues in descending order,
    and sorts the eigenvectors to correspond.

    Inputs:
        arr1 (np.Array or list)
        arr2 (np.Array or list)
        arr3 (np.Array or list)

    Outputs:
        sorted_eVals (np.Array): array of eigenvalue scalars sorted in desc-
                                    ending order.
        sorted eVecs (np.Array): 3x3 matrix of eigenvectors, ordered corresp-
                                    onding to indices of ordered values
    """

    # Check if inputs are NumPy arrays or can be converted to NumPy arrays
    try:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        arr3 = np.array(arr3)
    except ValueError:
        arr1_type = type(arr1).__name__
        arr2_type = type(arr2).__name__
        arr3_type = type(arr3).__name__
        raise TypeError(
            f"In the 'eigen_analysis' function: Input arrays must"
            " be NumPy arrays or convertible to NumPy arrays,"
            "but received {arr1_type}, {arr2_type}, and"
            " {arr3_type}"
        )

    # Solve for the eigenspace of the input vectors
    eVals, eVecs = eig(arr1, arr2, arr3)

    # Sort the eigenvalues and eigenvectors
    sorted_eVals, sorted_eVecs = sort(eVals, eVecs)

    return sorted_eVals, sorted_eVecs


def main(**kwargs):
    print(":)")


if __name__ == "__main__":
    main()
