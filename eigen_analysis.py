# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:55:18 2024

@author: Austin Abreu
"""
import numpy as np
import numpy.typing as npt

def eigen_analysis(arr1: npt.ArrayLike,
                   arr2: npt.ArrayLike,
                   arr3: npt.ArrayLike):
    
    # Check if inputs are NumPy arrays or can be converted to NumPy arrays
    try:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        arr3 = np.array(arr3)
    except ValueError:
        arr1_type = type(arr1).__name__
        arr2_type = type(arr2).__name__
        arr3_type = type(arr3).__name__
        raise TypeError(f"In the 'eigen_analysis' function: Input arrays must"
                        " be NumPy arrays or convertible to NumPy arrays,"
                        "but received {arr1_type}, {arr2_type}, and"
                        " {arr3_type}")
    
    # Combine the arrays into a 3xN matrix
    A = np.vstack((arr1, arr2, arr3))
    
    # Calculate the covariance matrix
    CM = np.cov(A)
    
    # Calculate the eigenvalues and eigenvectors
    eVals, eVecs = np.linalg.eig(CM)
    
    # Sort the eigenvalues and eigenvectors
    sorted_indices = np.argsort(eVals)[::-1]
    sorted_eVals = eVals[sorted_indices]
    sorted_eVecs = eVecs[:, sorted_indices]
    
    return sorted_eVals, sorted_eVecs
