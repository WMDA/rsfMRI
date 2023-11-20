from testing.connectivity_measure import ConnectivityMeasure
from decouple import config
from fNeuro.utils.pickling import load_pickle
import os
import numpy as np


resting_path = config('resting')
time_series = load_pickle(os.path.join(resting_path, 'measures', 'time_series'))
an_time_series = list(time_series['an'].values())
hc_time_series = list(time_series['hc'].values())
group = np.asarray(hc_time_series + an_time_series)
participant_1 = group[0]

conn = ConnectivityMeasure(kind='cyclic', vectorize=True)

print(conn.fit_transform(an_time_series))