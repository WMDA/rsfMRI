from fNeuro.utils.pickling import load_pickle, save_pickle
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
    an_time_series = list(time_series['an'].values())
    hc_time_series = list(time_series['hc'].values())
    part_hc = list(time_series['HC'].keys())
    part_an = list(time_series['AN'].keys())
    part = part_hc + part_an
    group = np.asarray(an_time_series + hc_time_series)
    label = np.asarray([0 for sub in range(len(an_time_series))] + [1 for sub in range(len(hc_time_series))])
    connectome = Cyclic_analysis().fit(group)
    oversampler = smote_variants.Assembled_SMOTE()
    X, y = oversampler.sample(connectome, label)
    save_pickle(os.path.join(resting_path, 'measures', 'X'), X)
    save_pickle(os.path.join(resting_path, 'measures', 'y'), y)
    save_pickle(os.path.join(resting_path, 'measures', 'connectome'), connectome)
    save_pickle(os.path.join(resting_path, 'measures', 'connectome_participant_order'), part)

if __name__ == "__main__":
    pre_process()