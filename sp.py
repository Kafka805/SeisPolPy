import numpy as np
import pandas as pd
import numpy.typing as npt
from obspy import Stream, Trace, UTCDateTime
from scipy.signal.windows import tukey
from obspy.signal.detrend import simple
from scipy.signal import butter, filtfilt
import os
    
        
def filtermerge(st, corner_l=2, corner_h=8, order=1):
    """
    Pre-processing for seismic data streams and traces. Designed for calls from SeisPol in mind.

    Parameters:
    st (obspy.Stream): Seismic data stream containing traces
    corner_l (float): Low corner frequency for the bandpass filter (default: 2 Hz)
    corner_h (float): High corner frequency for the bandpass filter (default: 8 Hz)
    order (int): Order of the Butterworth filter (default: 1)

    Returns:
    obspy.Stream: Filtered and merged stream
    """

    # Guard Clause to ensure each trace is from the same station, and that each trace has the same sampling rate.
    if isinstance(st, Stream) is False:
        st_type = type(st).__name__
        raise TypeError(f"Error in Filtermerge: Expected stream object as"
                        " input, but received {st_type}.")

    if len({tr.stats.sampling_rate for tr in st}) > 1:
        raise ValueError("Error in Filtermerge: The input traces have "
                         "different sampling rates.")
 
    # Extract sampling rate from traces
    dt = st[0].stats.delta

    # Create a new stream to store the filtered traces
    filtered_st = Stream()

    # Find the earliest start time and latest end time across all traces
    start_times = [tr.stats.starttime for tr in st]
    end_times = [tr.stats.endtime for tr in st]
    earliest_start = min(start_times)
    latest_end = max(end_times)

    for tr in st:
  # Trim or pad the trace to align with the earliest start and latest end times
        start_offset = (tr.stats.starttime - earliest_start) * tr.stats.sampling_rate
        end_offset = (latest_end - tr.stats.endtime) * tr.stats.sampling_rate
        padded_data = np.pad(tr.data, (int(start_offset), int(end_offset)),
                             mode='constant')

        # Detrend and demean the trace data
        padded_data = simple(padded_data) - np.mean(padded_data)

        # Butterworth filter parameters
        nyquist = 0.5 / dt
        low = corner_l / nyquist
        high = corner_h / nyquist
        b, a = butter(order, [low, high], btype='band', analog=False)

        # Filter the trace data
        filtered_data = filtfilt(b, a, padded_data)

  # Create a new trace with the filtered data and add it to the filtered stream
        filtered_tr = Trace(filtered_data)
        filtered_tr.stats.starttime = earliest_start
        filtered_st += filtered_tr

    return filtered_st

def eigen_analysis(arr1: npt.ArrayLike,
                   arr2: npt.ArrayLike,
                   arr3: npt.ArrayLike) -> tuple(npt.ArrayLike, npt.ArrayLike):
    
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


def polarity(Vecs: npt.ArrayLike, Vals: npt.ArrayLike) -> pd.Series:
    """
    Computes the polarity ratio values and solves for angles of incidence from 
    ordered eigenvalues and eigenvectors. To be used in combination with 
    'eigen_analysis.py'

    Parameters:
    Vecs (numpy.array): 3x3 matrix of vectors, assigned to eigenvalues found 
                        in Vals.
    Vals (numpy.array): Array of eigenvalues.

    """
    
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
            inci = np.arctan2(np.sqrt(alpha**2 + beta**2),gamma)
            if inci < 0:
                inci = inci+180
            elif  inci > 180:
                inci = inci-180
        
            azi = np.arctan2((alpha*np.sign(gamma)),(beta*np.sign(gamma)))
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
    
    #Initialize output datastructure and call computation functions
    dataPacket = pd.Series({
        'Rectilinearity': compute_rectilinearity(Vals),
        'Planarity': compute_planarity(Vals),
        'Azimuth': None,
        'Incident': None,
        'Normalized Diff': (Vals[1] - Vals[2]) / Vals[0]
    })

    # Retrieve the direction cosines
    alpha, beta, gamma = dir_cosines(Vecs)
    
    # Compute the angles from the direction cosines
    dataPacket['Incident'], dataPacket['Azimuth']  = compute_angles(alpha,
                                                                   beta, 
                                                                   gamma)

    return dataPacket


def SeisPol(st: Stream, window_size:int, window_overlap:float = 0.5, 
            scope:tuple = None, write:bool = False) -> pd.DataFrame:
    """
    Processes seismic data in Obspy Stream format to determine the polarity 
    parameters described in Jurkevics (1986).

    Parameters:
    st (obspy.Stream): Seismic data containing 3 traces from 1 station.
    period (int?): The target time-frame to subdivide the data by in seconds
    scope (tuple): Tuple describing time-frame to isolate for analysis.
    """
    #Guard Clause
    if len(st) != 3:
        raise ValueError(f"Expected stream object of size 3, received"
                         " {len(st)}.")
    
    # Utilize filtermerge to perform pre-processing
    working_signal = filtermerge(st)

    # Construct or edit the beginning and end times for the trace
    if scope is not None:
        for tr in working_signal:
            tr.trim(scope(0), scope(1))
    
    step = window_size // 2
    numsOut:int = st[0].stats.npts // step

    # Initialize output datastructure
    dataDummy = np.full((numsOut, 5), np.nan)

    dataNames = [
        'Rectilinearity',
        'Planarity',
        'Azimuth',
        'Incident',
        'Normalized Diff'
    ]
    
    dataSet = pd.DataFrame(dataDummy, columns = dataNames)

    # Construct the tukey window envelope
    cTW = tukey(window_size, 0.5)

    # Compute polarity metrics for each window
    working_data = np.vstack([tr.data for tr in working_signal])
    
    dataSet = pd.DataFrame(columns=dataNames, index=range(numsOut))

    for i in range(numsOut):
        try:
            start = i * step
            end = start + window_size
            windows = working_data[:, start:end]
    
            window1 = windows[0, :] * cTW
            window2 = windows[1, :] * cTW
            window3 = windows[2, :] * cTW
    
            eVals, eVecs = eigen_analysis(window1, window2, window3)
            dataPass = polarity(eVecs, eVals)
            dataSet.iloc[i] = dataPass
        except ValueError:
            pass
        # The final portion raises an error because there aren't enough samples.
        # Need to come up with a strategy for handling this.

    # Save the computed data to disk
    if write is True:
        dataSet.to_csv(os.getcwd())

    return dataSet


def main(kwargs: dict) -> None:
    SeisPol(**kwargs)

if __name__ == '__main__':
    main()