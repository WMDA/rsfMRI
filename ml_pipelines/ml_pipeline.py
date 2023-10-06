from get_cyclic_analysis import pre_process
from ml_building import ml_building

if __name__ == "__main__":
    print('Building predictor and features matrix\n')
    pre_process()
    print('Building and fitting ml models\n')
    ml_building()
    print('Finished')
