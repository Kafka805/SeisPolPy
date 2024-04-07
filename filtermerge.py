# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:47:48 2024

@author: Austin Abreu
"""
import numpy as np
from obspy import Stream, Trace
from obspy.signal.detrend import simple
from scipy.signal import butter, filtfilt
#%%
def filtermerge(st, corner_l=2, corner_h=8, order=1):
    """
    Pre-processing for seismic data streams and traces. Designed for calls from
    SeisPol in mind.

    Parameters:
    st (obspy.Stream): Seismic data stream containing traces
    corner_l (float): Low corner frequency for the bandpass filter (default: 2 Hz)
    corner_h (float): High corner frequency for the bandpass filter (default: 8 Hz)
    order (int): Order of the Butterworth filter (default: 1)

    Returns:
    obspy.Stream: Filtered and merged stream
    """

    # Guard Clause to ensure each trace is from the same station, and that each
    # trace has the same sampling rate.
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