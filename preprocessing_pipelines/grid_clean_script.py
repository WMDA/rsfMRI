from decouple import config
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn import image 
import os
import re
import argparse

def options() -> dict:
    '''
    Function to accept accept command line flags

    Parameters
    ---------
    None

    Returns
    -------
    dictionary of flags given
    '''
    flags = argparse.ArgumentParser()
    flags.add_argument('-s', '--s', dest='subjects',
                       help='str of image path')
    return vars(flags.parse_args())

def get_subject_name(img: str):
    
    '''
    Function to return

    Parameters
    ----------
    img: str
        str of image path
    
    Returns
    -------
    Subject name: str
        subject name
    '''
    
    return re.findall(r'sub-B....', img)[0] + '_cleaned.nii.gz'

def get_counfounds(img: str):
    
    '''
    Load the confounds dataframe

    Parameters
    ----------
    img: str
        str of image path
    
    Returns
    -------
    confounds: pd.DataFrame

    '''
    confounds = load_confounds_strategy(img, denoise_strategy='compcor', motion='full')[0]
    return confounds.drop(list(confounds.filter(regex='cosine*')), axis=1)


def process_image(img: str):
    
    '''
    Function to clean images

    Parameters
    ----------
    img: str
        str of image path

    Return
    ------
    None
    '''
    name = get_subject_name(img)
    print('Working on: ', name)
    to_save_path = os.path.join(config('resting'), 'cleaned')
    confounds = get_counfounds(img)
    fmri_cleaned_image =  image.clean_img(
                           img,
                           low_pass=0.08,
                           high_pass=0.01,
                           t_r=2,
                           ensure_finite=True,
                           standardize="zscore_sample",
                           confounds=confounds
                           )
    fmri_cleaned_image.to_filename(os.path.join(to_save_path, name))

if __name__ == "__main__":
    img = options()['subjects']
    process_image(img)
    print('done')
