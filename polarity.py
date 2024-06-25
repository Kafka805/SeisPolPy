"""
@author: Austin Abreu
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from .dStruct import dataStruct

# Computes the rectilinear polarization factor
def compute_rectilinearity(eigs: npt.ArrayLike) -> float:
    """
    Parameters
    ----------
    eigs : list or array
        List of eigenvalues for computation.
        
    Outputs
    -------
    rectParm : float
        Rectilinear polariation factor.
    """
    rectParam = 1 - ((eigs[1] + eigs[2]) / (2*eigs[0]))
    
    return rectParam


# Computes the planar polarization function
def compute_planarity(eigs: npt.ArrayLike) -> float:
    """
    Parameters
    ----------
    eigs: npt.ArrayLike
        List of eigenvalues for computation.

    Outputs
    -------
    planParam: float
        Planar polarization factor.

    """
    planParam = 1 - ((2*eigs[2]) / (eigs[0] + eigs[1]))
    
    return planParam


# Computes the angles of the source signal
def compute_angles(a:float, b:float, g:float) -> float:
    """
    Parameters
    ----------
    a : float
        Direction cosine defined from the Z-axis
    b : float
        Direction cosine defined from the Y-axis.
    g : float
        Direction cosine defined from the X-axis.

    Outputs
    -------
    inci : float
        Angle of incidence.
    azi : float
        Back-azimuthal angle.
    """
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


# Computes the direction cosines of an array of input vectors
def dir_cosine(vector: npt.ArrayLike,
                basis: npt.ArrayLike = None) -> float:
    """
    Parameters
    ----------
    eigen_vectors : npt.ArrayLike
        List of eigenvectors representing the analysis space.
        
    basis : npt.ArrayLike
        Set of basis vectors to use in computation of the cosines.

    Outputs
    -------
    alpha : float
        Angle defined from the Z-axis
    beta : float
        Angle defined from the Y-axis.
    gamma : float
        Angle defined from the X-axis.
    """
    if basis is None:
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Computes the Norm of a vector
    def norm(x: list[int], *, order:int = 2) -> int:
        
        #Default is the Euclidean Norm
        if type(order) is not int:
            raise Exception('Please provide an integer dimension for the norm')
        
        step = [i**order for i in x]
        out = sum(step)**(1/order)
        
        return out
    
    vecNorm = norm(vector)
    
    alpha = np.dot(vector, basis[:,0]) / vecNorm
    beta = np.dot(vector, basis[:,1]) / vecNorm
    gamma = np.dot(vector, basis[:,2]) / vecNorm
    
    return alpha, beta, gamma

# Handles data organization and computation calls
def polarity(Vecs: npt.ArrayLike, Vals: npt.ArrayLike) -> pd.Series:
    """
    Computes the polarity ratio values and solves for angles of incidence from 
    ordered eigenvalues and eigenvectors. To be used in combination with 
    'eigSort.py'

    Parameters:
    ----------
    Vecs (numpy.array): 3x3 matrix of vectors, assigned to eigenvalues found 
                        in Vals.
    Vals (numpy.array): Array of eigenvalues.
    
    Outputs:
    --------
    dataPacket (pd.Series): Output calculations, with labels, for assimilation
                            with main dataframe in SeisPol.
    """
    
    #Initialize output datastructure and call computation functions
    
    dataPacket = dataStruct(length = 1)
    
    dataPacket.body['Rectilinearity'] = compute_rectilinearity(Vals)
    dataPacket.body['Planarity'] = compute_planarity(Vals)
    dataPacket.body['Normalized Diff'] = (Vals[1] - Vals[2]) / Vals[0]

    # Retrieve the direction cosines of the largest eigenvalue (P-wave)
    alpha, beta, gamma = dir_cosine(Vecs[:,0])
    
    # Compute the angles from the direction cosines
    dataPacket.body['Incident'], dataPacket.body['Azimuth']  = compute_angles(
                                                                alpha,
                                                                beta,
                                                                gamma)

    return dataPacket