from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from decouple import config
import os

from fNeuro.utils.pickling import load_pickle, save_pickle
from fNeuro.ml.ml_functions import estimate_model


def ml_building():
    
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

if __name__ == "__main__":
    ml_building()

