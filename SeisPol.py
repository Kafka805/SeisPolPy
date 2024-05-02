import os

import numpy as np
import pandas as pd
import numpy.typing as npt
from obspy import Stream
from scipy.signal.windows import tukey
from .filtermerge import filtermerge
from .eigSort import eigSort
from .polarity import polarity
from .dStruct import dataStruct


def _applyWindow(
    data, window
) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:

    window1 = data[0, :] * window
    window2 = data[1, :] * window
    window3 = data[2, :] * window

    return window1, window2, window3


def _computeParams(sig1, sig2, sig3):

    eVals, eVecs = eigSort(sig1, sig2, sig3)
    dataPass = polarity(eVecs, eVals)

    return dataPass


def seisPol(
    st: Stream, window_size: int, scope: tuple = None, write: bool = False
) -> pd.DataFrame:
    """
    Processes seismic data in Obspy Stream format to determine the polarity
    parameters as described in Jurkevics (1986).

    Parameters:
    st (obspy.Stream): Seismic data containing 3 traces from 1 station.
    window_size (int?): The target time-frame to subdivide the data by in seconds
    scope (tuple): Tuple describing time-frame to isolate for analysis.
    """
    # Guard Clause
    if len(st) != 3:
        raise ValueError(
            f"""Expected stream object of size 3, received
                          {len(st)}."""
        )

    # Utilize filtermerge to perform pre-processing
    working_signal = filtermerge(st)

    # Construct or edit the beginning and end times for the trace
    if scope is not None:
        for tr in working_signal:
            tr.trim(scope(0), scope(1))

    step = window_size // 2
    numsOut: int = st[0].stats.npts // step

    # Initialize output datastructure
    dataSet = dataStruct(length=numsOut)

    # Construct the Tukey window envelope
    cTW = tukey(window_size, 0.5)

    # Compute polarity metrics for each window
    working_data = np.vstack([tr.data for tr in working_signal])

    for i in range(numsOut):
        start = i * step
        end = start + window_size
        windows = working_data[:, start:end]

        try:
            window1, window2, window3 = _applyWindow(windows, cTW)
            dataPass = _computeParams(window1, window2, window3)
            dataSet.body.iloc[i] = dataPass.body

        except ValueError:
            cTW_end = tukey(windows.shape[1], 0.5)
            window1, window2, window3 = _applyWindow(windows, cTW_end)
            dataPass = _computeParams(window1, window2, window3)
            dataSet.body.iloc[i] = dataPass.body

    # Save the computed data to disk
    if write is True:
        dataSet.to_csv(os.getcwd())

    return dataSet.body


# def main(kwargs: dict) -> None:
#   SeisPol(**kwargs)

# if __name__ == '__main__':
#   main()
