from decouple import config
import os
import numpy as np
import smote_variants

from fNeuro.utils.pickling import load_pickle

if __name__ == "__main__":
    resting_path = config('resting')
    time_series = load_pickle(os.path.join(resting_path, 'measures', 'time_series'))
    an_time_series = time_series['an']
    hc_time_series = time_series['hc']
    group = np.asarray(an_time_series + hc_time_series)
    label = np.asarray([0 for sub in range(len(an_time_series))] + [1 for sub in range(len(hc_time_series))])
    connectome = Cyclic_analysis().fit(group)
    oversampler = smote_variants.Assembled_SMOTE()
    X, y = oversampler.sample(connectome, label)