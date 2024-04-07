# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:58:32 2024

@author: Austin Abreu
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from .dStruct import dataStruct

# Computes the rectilinear polarization factor
def compute_rectilinearity(eigs) -> float:
    rectParam = 1 - ((eigs[1] + eigs[2]) / (2*eigs[0]))
    
    return rectParam

# Computes the planar polarization function
def compute_planarity(eigs: npt.ArrayLike) -> float:
    planParam = 1 - ((2*eigs[2]) / (eigs[0] + eigs[1]))
    
    return planParam

# Computes the angles of the source signal
def compute_angles(a, b, g) -> float:
    inci = np.arctan2(np.sqrt(a**2 + b**2),g)
    if inci < 0:
        inci = inci+180
    elif  inci > 180:
        inci = inci-180

    azi = np.arctan2((a*np.sign(g)),(b*np.sign(g)))
    if azi < 0:
        azi = azi+360
    elif  azi > 360:
        azi = azi-360
        
    return inci, azi
    
# Computes the Norm of a vector
def norm(x: list[int], *, order:int = 2) -> int:
    #Default is the Euclidean Norm
    if isinstance(order,int) is False:
        raise Exception('Please provide an integer dimension for the norm')
    
    step = [i**order for i in x]
    out = sum(step)**(1/order)
    
    return out

# Computes the direction cosines of an array of input vectors
def dir_cosines(eigen_vectors: npt.ArrayLike) -> float:
    big_norm = norm(eigen_vectors[:,0])
    
    alpha = eigen_vectors[0,0] / big_norm
    beta = eigen_vectors[1,0] / big_norm
    gamma = eigen_vectors[2,0] / big_norm
    
    return alpha, beta, gamma

def polarity(Vecs: npt.ArrayLike, Vals: npt.ArrayLike) -> pd.Series:
    """
    Computes the polarity ratio values and solves for angles of incidence from 
    ordered eigenvalues and eigenvectors. To be used in combination with 
    'eigen_analysis.py'

    Parameters:
    Vecs (numpy.array): 3x3 matrix of vectors, assigned to eigenvalues found 
                        in Vals.
    Vals (numpy.array): Array of eigenvalues.
    
    Outputs:
    dataPacket (pd.Series): Output calculations, with labels, for assimilation
                            with main dataframe in SeisPol.
    """
    
    #Initialize output datastructure and call computation functions
    
    dataPacket = dataStruct(length = 1)
    
    dataPacket.body['Rectilinearity'] = compute_rectilinearity(Vals)
    dataPacket.body['Planarity'] = compute_planarity(Vals)
    dataPacket.body['Normalized Diff'] = (Vals[1] - Vals[2]) / Vals[0]

    # Retrieve the direction cosines
    alpha, beta, gamma = dir_cosines(Vecs)
    
    # Compute the angles from the direction cosines
    dataPacket['Incident'], dataPacket['Azimuth']  = compute_angles(alpha,
                                                                   beta, 
                                                                   gamma)

    return dataPacket