import os

import numpy as np
import pandas as pd
from obspy import Stream
from scipy.signal.windows import tukey
from .filtermerge import filtermerge
from .eigen_analysis import eigen_analysis
from .polarity import polarity

def analyze(st: Stream, window_size:int, window_overlap:float = 0.5, 
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


#def main(kwargs: dict) -> None:
#   SeisPol(**kwargs)

#if __name__ == '__main__':
#   main()