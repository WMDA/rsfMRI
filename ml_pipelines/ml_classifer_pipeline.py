from get_cyclic_analysis import pre_process
from ml_building import ml_classifer
from comparison import comparisons

if __name__ == "__main__":
    print('Building predictor and features matrix\n')
    pre_process()
    print('Building and fitting ml models\n')
    ml_classifer()
    print('Building Comparison models')
    comparisons()
    print('Finished')

