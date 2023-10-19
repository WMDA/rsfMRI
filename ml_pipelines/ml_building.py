# ml imports
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

# Generic
from decouple import config
import os
import numpy as np
import pandas as pd
import re

# fNeuro
from fNeuro.utils.pickling import load_pickle, save_pickle
from fNeuro.ml.ml_functions import estimate_model, estimate_regression_model


def ml_classifer():
    
    '''
    Function to build, svc, decision tree,
    Random forest and stacked ml models.
    Saves everything as pickle objects

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    resting_path = config('resting')
    X = load_pickle(os.path.join(resting_path, 'measures', 'X'))
    y = load_pickle(os.path.join(resting_path, 'measures', 'y'))

    # SVC model
    print('Building SVC model\n')
    svc_parameters = {'C': [1e0, 1e1, 1e2]}
    svc = LinearSVC(dual=True)
    svc = GridSearchCV(svc, svc_parameters)
    svc_model = estimate_model(svc, X, y)
    svc_model['model'] = svc.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', 'svc_model'), svc_model)
    
    # Logistic model
    print('Building Logistic model\n')
    log_parameters = {'C': [1e0, 1e1, 1e2]}
    logistic = LogisticRegression() 
    logistic = GridSearchCV(logistic, log_parameters)
    logistic_model = estimate_model(logistic, X, y)
    logistic_model['model'] = logistic.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', 'logistic_model'), logistic_model)

    # Tree model 
    print('Building Decesion tree model\n')
    tree_parameters ={'criterion':["gini", "entropy", "log_loss"]}
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf = GridSearchCV(tree_clf, tree_parameters)
    tree_model = estimate_model(tree_clf, X, y)
    tree_model['model'] = tree_clf.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', 'tree_model'), tree_model)

    # Random forest model
    print('Building Random Forest model\n')
    ranforest = RandomForestClassifier()
    ranforest = GridSearchCV(ranforest, tree_parameters)
    random_forest_model = estimate_model(ranforest, X, y)
    random_forest_model['model'] = ranforest.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', 'random_forest_model'), random_forest_model)

    # Stacked model
    print('Building Stacked model\n')
    estimators = [('rf', RandomForestClassifier()), ('log', LogisticRegression())]
    stacked_estimator = StackingClassifier(estimators, final_estimator=LogisticRegression())
    stacked_estimator_model = estimate_model(stacked_estimator, X, y)
    stacked_estimator_model['model'] = stacked_estimator.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', 'stacked_model'), stacked_estimator_model)

def aq10_no_feature_selection():
    resting_path = config('resting')
    autistic_traits = pd.read_csv(os.path.join(resting_path, 'autistic_traits_neuroimaging.csv'))
    autistic_traits['B_Number'] = autistic_traits['B_Number'].str.rstrip()
    autistic_traits = autistic_traits.drop(autistic_traits[autistic_traits['B_Number'].str.contains('B2010')].index).reset_index(drop=True)
    autistic_traits['B_Number'].loc[autistic_traits[autistic_traits['B_Number'] == 'B2024b'].index[0]] = 'B2024'
    connectome = load_pickle(os.path.join(resting_path, 'measures', 'connectome'))
    participant_order = load_pickle(os.path.join(resting_path, 'measures', 'connectome_participant_order'))
    order = [re.findall('B\d\d\d\d', participant)[0] for participant in participant_order]
    connectome_dict = dict(zip(order, connectome))
    
    aq_df = autistic_traits[['B_Number', 'aq10']].dropna().reset_index(drop=True)
    y_aq = aq_df['aq10'].values
    x_aq = np.array([connectome_dict.get(key) for key in aq_df['B_Number'].values])

def regression_models_cross_val(X, y, resting_path, pickle_suffix):
    # LassoCV model
    lasso = linear_model.LassoCV(cv=10, n_jobs=3)
    scores_lasso = estimate_regression_model(lasso, X, y)
    scores_lasso['model'] = lasso.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'lasso_{pickle_suffix}'), scores_lasso)

    # Ridge
    ridge = linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    scores_ridge = estimate_regression_model(ridge, X, y)
    scores_ridge['model'] = ridge.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'ridge_{pickle_suffix}'), scores_ridge)

    # SVR
    svr_parameters = {'C': [1e-2, 1e-1, 1e0, 1e1, 1e2]}
    svr_regression = GridSearchCV(svm.SVR(kernel='linear', gamma="auto"), svr_parameters)
    scores_svr = estimate_regression_model(svr_regression, X, y)
    scores_svr['model'] = svr_regression.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'svr_{pickle_suffix}'), scores_svr)

    # Random forest regressor
    tree_parm = {'max_depth':[3, 5, 7]}
    ran_forest = GridSearchCV(RandomForestRegressor(random_state=0), tree_parm)
    score_ran = estimate_regression_model(ran_forest, X, y)
    score_ran['model'] = ran_forest.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'random_forest_{pickle_suffix}'), score_ran)

    # stacked
    estimators = [('ridge',linear_model.RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])), ('svr', svm.SVR(kernel='linear', gamma="auto"))]
    stacked_estimator = StackingRegressor(estimators)
    score_stacked_estimator = estimate_regression_model(stacked_estimator, X, y)
    score_stacked_estimator['model'] = stacked_estimator.fit(X, y)
    save_pickle(os.path.join(resting_path, 'measures', f'stacked_model_{pickle_suffix}'), score_stacked_estimator)

if __name__ == "__main__":
    ml_classifer()

