from nilearn.maskers import NiftiMapsMasker
from nilearn import datasets
import os
import glob
from decouple import config
from fNeuro.utils.pickling import save_pickle

if __name__ == "__main__":
    print('Setting up enviornment\n')
    resting_path = config('resting')
    fmri_imgs = glob.glob(f'{resting_path}/cleaned/*.nii.gz')
    mdsl = datasets.fetch_atlas_msdl()
    
    masker = NiftiMapsMasker(
        maps_img=mdsl['maps'],
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        verbose=5,
    ).fit()
    
    time_series ={
        'an': {},
        'hc': {}
    }
    
    print('Extracting time series')
    for part in fmri_imgs:
        time_series_data = masker.transform(part)
        if 'sub-B1' in part:
            time_series['hc'][part] = time_series_data
        else:
            time_series['an'][part] = time_series_data
    
    save_pickle(os.path.join(resting_path, 'measures', 'time_series'), time_series)