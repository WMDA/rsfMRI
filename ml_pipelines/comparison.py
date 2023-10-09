from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from nilearn.connectome import ConnectivityMeasure
import smote_variants
from fNeuro.utils.pickling import load_pickle, save_pickle
from fNeuro.ml.ml_functions import estimate_model
from decouple import config
import numpy as np
import os


def comparisons():

    '''
    Function to compare AUC and accuaracy of different
    connectivity measures.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    svc_parameters = {'C': [1e0, 1e1, 1e2]}
    svc = LinearSVC(dual=True)
    svc = GridSearchCV(svc, svc_parameters)
    
    resting_path = config('resting')
    time_series = load_pickle(os.path.join(resting_path, 'measures', 'time_series'))
    an_time_series = time_series['an']
    hc_time_series = time_series['hc']
    group = np.asarray(an_time_series + hc_time_series)
    label = np.asarray([0 for sub in range(len(an_time_series))] + [1 for sub in range(len(hc_time_series))])
    kinds = ["correlation", "partial correlation", "tangent"]
    
    scores = {}
    for kind in kinds:
        scores[kind] = []
        connectivity = ConnectivityMeasure(kind=kind, vectorize=True).fit_transform(group)
        oversampler= smote_variants.Assembled_SMOTE()
        cor_X, cor_y = oversampler.sample(connectivity, label)
        scores[kind].append(estimate_model(svc, cor_X, cor_y))
    scores['cyclic'] = load_pickle(os.path.join(resting_path, 'measures', 'svc_model'))['Accura']
    save_pickle(os.path.join(resting_path, 'measures', 'svc_comparison'), scores)

if __name__ == "__main__":
    comparisons()
