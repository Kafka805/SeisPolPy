"""@author: Austin Abreu."""

import numpy as np
import numpy.typing as npt
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.detrend import simple
from scipy.signal import butter, filtfilt


# %%
def butterworth(data, dt, corner_l, corner_h, order, *,
                test=False) -> npt.ArrayLike:
    """
    Signal processing function that performs a butterworth filtering method on
    the input signal.
    Designed for calls from SeisPol, for use in the pre-processing package.

    """
    # Butterworth filter parameters
    nyquist = 0.5 / dt
    low = corner_l / nyquist
    high = corner_h / nyquist
    b, a = butter(order, [low, high], btype="band", analog=False)

    if test is True:
        return b, a, filtfilt(b, a, data)
    else:
        return filtfilt(b, a, data)


def trimPad(tr, start, end) -> npt.ArrayLike:
    """
    Identifies the earliest time and latest times embedded in the stream, then
    trims, or pads the data with zeros, to these times using np.pad.

    """
    start_offset = (tr.stats.starttime - start) * tr.stats.sampling_rate
    endOffset = (end - tr.stats.endtime) * tr.stats.sampling_rate

    editedData = np.pad(
        tr.data,
        (int(start_offset), int(endOffset)),
        mode="constant",
        constant_values=0,
    )

    return editedData


def statsPull(st) -> dict:
    """
    Extract relevant metadata from traces.

    Inputs:
    st : Obspy stream object containing traces. Traces should be from the same
            source.

    Ouputs:
    out (dict): a dictionary containing
                    SR : the sampling rate of the traces
                    Start : the earliest timestamp from the stream
                    End : the latest timestap from the stream

    """
    if len({tr.stats.sampling_rate for tr in st}) > 1:
        raise ValueError(
            "Error in Filtermerge(statsPull): The input traces "
            "have different sampling rates."
        )

    dt: float = st[0].stats.delta

    # Initialize container dictionaries
    startTimes: list[UTCDateTime] = []
    endTimes: list[UTCDateTime] = []

    # Find the start/end times embedded in the traces.
    # This is somewhat redundant, assuming you've ensured your data quality.
    for tr in st:
        startTimes.append(tr.stats.starttime)
        endTimes.append(tr.stats.endtime)

    if (startTimes[0] == startTimes[1] == startTimes[2]) is False:
        raise ValueError(
            "Error in FilterMerge(statsPull):"
            "The given trace start times are different."
        )
    if (endTimes[0] == endTimes[1] == endTimes[2]) is False:
        raise ValueError(
            "Error in FilterMerge(statsPull):"
            "The given trace end times are different."
        )

    earliestStart = min(startTimes)
    latestEnd = max(endTimes)

    return {"sr": dt, "Start": earliestStart, "End": latestEnd}


# %%
def filterMerge(working_st, corner_l=2, corner_h=8, order=1) -> Stream:
    """Handler function for pre-processing for seismic data streams and traces.
    Designed for calls from SeisPol.

    Parameters
    ----------
    st (obspy.Stream): Seismic data stream containing traces
    corner_l (float): Low corner frequency for the bandpass filter
                        (default: 2 Hz). Passed to 'butterworth.'
    corner_h (float): High corner frequency for the bandpass filter
                        (default: 8 Hz. Passed to 'butterworth.'
    order (int): Order of the Butterworth filter (default: 1).
                    Passed to 'butterworth.'

    Returns
    -------
    obspy.Stream: Filtered and merged stream
    """

    # Guard Clause to ensure each trace is from the same station, and that each
    # trace has the same sampling rate.
    if isinstance(working_st, Stream) is False:
        input_type = type(working_st).__name__
        raise TypeError(
            f"Error in Filtermerge: Expected stream object as"
            " input, but received {input_type}."
        )

    # Retrieve relevant metadata
    metaData = statsPull(working_st)

    # Create a new stream to store the filtered traces
    filtered_st = Stream()

    for tr in working_st:
        # Call trimPad to perform datasize editing.
        # This will destroy all metadata.
        paddedData = trimPad(tr, metaData["Start"], metaData["End"])

        # Detrend and demean the trace data
        paddedData = simple(paddedData) - np.mean(paddedData)

        # Call butterworth to perform the filtering we have designed
        filteredData = butterworth(
            paddedData, metaData["sr"], corner_l, corner_h, order
        )

        # Create a new trace with the filtered data and add it to the
        # filtered stream
        filtered_tr = Trace(filteredData)
        filtered_tr.stats.starttime = metaData["Start"]
        filtered_st += filtered_tr

    return filtered_st
