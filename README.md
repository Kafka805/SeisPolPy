# Description:

## SeisPol.py
### Inputs
- **st {Obspy.Stream}:** An Obspy Stream instance with 3 traces from the same station. Traces should be ordered st[0] == East/West, st[1] == North, st[2] == Vertical. Please ensure the following is true about your trace data for best results:
	- The traces are equal in length
	- The traces have the same start and end times
	- The sampling rate, and consequent delta-time, is available in the metadata
- window_size (integer): An integer number of samples the define the time-axis width of the sampling window.
- scope (tuple[UTCDateTime, UTCDateTime]): Tuple containing two UTCDateTime instances that define the outer boundaries of the analysis. scope[0] defines the first (closest) sample at which the analysis begins; scope[1] defines the last (closest) sample at which the analysis ends. Data outside this time scope will be expunged by the function to save resources. This allows for flexible workflows, easy-of-use, and customizable output size.
- write (boolean): If write is *True*, the function writes the output array to a csv using pandas.to_csv in the current working directory. If write is *False*, nothing happens.

### Outputs
**pd.DataFrame:** a pandas dataframe of the following structure: 

| Rectilinearity   | Planarity       | Azimuth         | Incident        | Normalized Diff |
| ---------------- | --------------- | --------------- | --------------- | --------------- |
| np.ndarray[1, n] | np.ndarray[1,n] | np.ndarray[1,n] | np.ndarray[1,n] | np.ndarray[1,n] |

Where *n* is the number of *windows* contained by the analysis *scope*. These values are the output parameters for the signal.

**If write is *True***: a .csv file in the current working directory containing the dataframe

### Description
SeisPol acts as the "handler" function for the analysis. The public function is seisPol(), which takes the inputs described above. The design flow is as follows:
1. Timeseries data and metadata is extracted from the streams, pre-processed, and filtered using *filtermerge*.
2. The timeseries is pruned to the *scope.*
3. *window_size* is used to calculate the number of outputs and the datastructure is instantiated via *dStruct*. The tukey window is created.
4. Data is processed at each window interval:
	1. The data is segmented then windowed with the Tukey window
	2. Eigenvalues are computed by *eigSort*.
	3. The parameters are computed by *polarity*.
	4. The data is passed back to seisPol and the data structure.
5. If *write* is *True*: the data is written to a .csv file.
6. The function returns the pandas datastructure containing the computed parameters.
