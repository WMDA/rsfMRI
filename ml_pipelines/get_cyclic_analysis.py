from fNeuro.ml.mvpa_functions import load_pickle, save_pickle
from fNeuro.connectivity.connectivity import Cyclic_analysis
from decouple import config
import os
import numpy as np
import smote_variants


def pre_process() -> None:
    
    '''
    Function to run Cylic analysis and
    smote to oversample. Results get 
    saved as pickle files

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    resting_path = config('resting')
    time_series = load_pickle(os.path.join(resting_path, 'measures', 'time_series'))
    an_time_series = time_series['an']
    hc_time_series = time_series['hc']
    group = np.asarray(an_time_series + hc_time_series)
    label = np.asarray([0 for sub in range(len(an_time_series))] + [1 for sub in range(len(hc_time_series))])
    connectome = Cyclic_analysis().fit(group)
    oversampler = smote_variants.Assembled_SMOTE()
    X, y = oversampler.sample(connectome, label)
    save_pickle(os.path.join(resting_path, 'measures', 'X'), X)
    save_pickle(os.path.join(resting_path, 'measures', 'y'), y)
    save_pickle(os.path.join(resting_path, 'measures', 'connectome'), connectome)

if __name__ == "__main__":
    pre_process()