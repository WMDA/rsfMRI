# ml imports
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import r_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Generic
from decouple import config
import os
import numpy as np
import pandas as pd
import re

# fNeuro
from fNeuro.utils.pickling import load_pickle, save_pickle
from fNeuro.ml.ml_functions import estimate_regression_model

def get_features(measure: str, resting_path: str) -> dict:

    '''
    Function to get X, Y

    Parameters
    ----------
    measure: str
        str of measure, either aq10
        or ADOS total

    resting_path: str
        str of absolute path to store results
    
    Returns
    -------
    dict: dictionary object
        dict of X and y

    '''
    autistic_traits = pd.read_csv(os.path.join(resting_path, 'autistic_traits_neuroimaging.csv'))
    autistic_traits['B_Number'] = autistic_traits['B_Number'].str.rstrip()
    autistic_traits = autistic_traits.drop(autistic_traits[autistic_traits['B_Number'].str.contains('B2010')].index).reset_index(drop=True)
    autistic_traits['B_Number'].loc[autistic_traits[autistic_traits['B_Number'] == 'B2024b'].index[0]] = 'B2024'
    connectome = load_pickle(os.path.join(resting_path, 'measures', 'connectome'))
    participant_order = load_pickle(os.path.join(resting_path, 'measures', 'connectome_participant_order'))
    order = [re.findall('B\d\d\d\d', participant)[0] for participant in participant_order]
    connectome_dict = dict(zip(order, connectome))
    df = autistic_traits[['B_Number', measure]].dropna().reset_index(drop=True)
    return {
        'X': np.array([connectome_dict.get(key) for key in df['B_Number'].values]),
        'y': df[measure].values
    }

def feature_selection(X: np.array, y: np.array) -> np.array:
    '''
    Function to filter features by pearson R.

    Parameters
    ----------
    X: np.array
        array of X values
    y: np.array
        arry of y values

    Returns
    -------
    np.array: array
        array of filtered X values
    '''
    filtering_features = r_regression(X, y)
    index = np.where((filtering_features > 0.1) | (filtering_features < -0.1))[0]
    return  X[0:, index]

def regression_models_cross_val(X: np.array, 
                                y: np.array, 
                                resting_path: str, 
                                pickle_suffix: str):
    '''
    Function to build regression models and
    score with cross validation

    Parameters
    ----------
    X: np.array
        array of X values
    y: np.array
        arry of y values
    resting_path: str
        str of absolute path to store results
    pickle_suffix: str
        str of suffix on pickle file
    
    Returns
    -------
    None
    '''
    
    # LassoCV model
    print('\nBuilding lasso model')
    lasso = linear_model.LassoCV(cv=10, n_jobs=3)
    scores_lasso = estimate_regression_model(lasso, X, y)
    scores_lasso['model'] = lasso.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'lasso_{pickle_suffix}'), scores_lasso)

    # Ridge
    print('\nBuilding Ridge model')
    ridge = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    scores_ridge = estimate_regression_model(ridge, X, y)
    scores_ridge['model'] = ridge.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'ridge_{pickle_suffix}'), scores_ridge)

    # SVR
    print('\nBuilding SVR model')
    svr_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2]}
    svr_regression = GridSearchCV(svm.SVR(kernel='linear', gamma="auto"), svr_parameters)
    scores_svr = estimate_regression_model(svr_regression, X, y)
    scores_svr['model'] = svr_regression.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'svr_{pickle_suffix}'), scores_svr)

    # Random forest regressor
    print('\nBuilding Random forest model')
    tree_parm = {'max_depth':[3, 5, 7]}
    ran_forest = GridSearchCV(RandomForestRegressor(random_state=0), tree_parm)
    score_ran = estimate_regression_model(ran_forest, X, y)
    score_ran['model'] = ran_forest.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'random_forest_{pickle_suffix}'), score_ran)

    # stacked
    print('\nBuilding stacked model')
    estimators = [('ridge',linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])), ('svr', svm.SVR(kernel='linear', gamma="auto"))]
    stacked_estimator = StackingRegressor(estimators)
    score_stacked_estimator = estimate_regression_model(stacked_estimator, X, y)
    score_stacked_estimator['model'] = stacked_estimator.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'stacked_model_{pickle_suffix}'), score_stacked_estimator)

def regression_models_no_cross_val(X: np.array, 
                                   y: np.array, 
                                   resting_path: str, 
                                   pickle_suffix: str):
    '''
    Function to build regression models and
    score with test train split

    Parameters
    ----------
    X: np.array
        array of X values
    y: np.array
        arry of y values
    resting_path: str
        str of absolute path to store results
    pickle_suffix: str
        str of suffix on pickle file
    
    Returns
    -------
    None
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,  
                                                        random_state=0, 
                                                        test_size=0.3)
    
    # SVC model 
    print('\nBuilding SVR model')
    svc_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2]}
    svr_ados = GridSearchCV(svm.SVR(kernel='linear', gamma="auto"), svc_parameters)
    svr_ados.fit(X_train, y_train)
    svr_score_ados = svr_ados.score(X_test, y_test)
    svr_mae_ados = mean_absolute_error(y_test, svr_ados.predict(X_test))
    svr_model = {
        'model': svr_ados.fit(X, y),
        'r2': svr_score_ados,
        'MAE': svr_mae_ados
    }
    save_pickle(os.path.join(resting_path, 'measures', f'svr_{pickle_suffix}'), svr_model)

    # LassoCV
    print('\nBuilding lasso model')
    lasso_ados = linear_model.LassoCV(cv=10, n_jobs=3)
    lasso_ados.fit(X_train, y_train)
    lasso_ados_score = lasso_ados.score(X_test, y_test)
    lasso_mae_ados = mean_absolute_error(y_test, lasso_ados.predict(X_test))
    lasso_model = {
        'model': lasso_ados,
        'r2': lasso_ados_score,
        'MAE': lasso_mae_ados
    }
    save_pickle(os.path.join(resting_path, 'measures', f'lasso_{pickle_suffix}'), lasso_model)

    # Ridge
    print('\nBuilding Ridge model')
    ridge_ados = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    ridge_ados.fit(X_train, y_train)
    ridge_ados_score = ridge_ados.score(X_test, y_test)
    ridge_mae_ados = mean_absolute_error(y_test, ridge_ados.predict(X_test))
    ridge_model = {
        'model': ridge_ados,
        'r2': ridge_ados_score,
        'MAE': ridge_mae_ados
    }
    save_pickle(os.path.join(resting_path, 'measures', f'ridge_{pickle_suffix}'), ridge_model)

    # RandomForest model
    print('\nBuilding Random forest model')
    tree_parm = {'max_depth':[3, 5, 7]}
    ran_forest_ados = GridSearchCV(RandomForestRegressor(random_state=0), tree_parm)
    ran_forest_ados.fit(X_train, y_train)
    ran_forest_ados_score = ran_forest_ados.score(X_test, y_test)
    ran_forest_mae_ados = mean_absolute_error(y_test, ran_forest_ados.predict(X_test))
    ran_forest_model = {
        'model': ran_forest_ados,
        'r2': ran_forest_ados_score,
        'MAE': ran_forest_mae_ados
    }
    save_pickle(os.path.join(resting_path, 'measures', f'ran_{pickle_suffix}'), ran_forest_model)

if __name__ == '__main__':
    print('Building regression models')
    resting_path = config('resting')
    aq10_unfiltered_features = get_features('aq10', resting_path)
    ados_unfiltered_features = get_features('ADOS total', resting_path)
    aq10_filtered_features = feature_selection(aq10_unfiltered_features['X'], aq10_unfiltered_features['y'])
    ados_filtered_features = feature_selection(ados_unfiltered_features['X'], ados_unfiltered_features['y'])
    print('\nBuilding AQ10 models with all features')
    regression_models_cross_val(aq10_unfiltered_features['X'], 
                                aq10_unfiltered_features['y'], 
                                resting_path,
                                'aq10_unfiltered')
    print('\nBuilding AQ10 models with feature selection')
    regression_models_cross_val(aq10_filtered_features, 
                                aq10_unfiltered_features['y'], 
                                resting_path,
                                'aq10_filtered')
    print('\nBuilding ADOS models with all features')
    regression_models_no_cross_val(ados_unfiltered_features['X'], 
                                   ados_unfiltered_features['y'],
                                   resting_path,
                                   'ados_unfitered'
                                   )
    print('\nBuilding ADOS models with feature selection')
    regression_models_no_cross_val(ados_filtered_features, 
                                   ados_unfiltered_features['y'],
                                   resting_path,
                                   'ados_fitered'
                                   )
