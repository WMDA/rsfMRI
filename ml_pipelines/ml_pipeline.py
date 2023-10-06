from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from decouple import config
import os
import smote_variants

from fNeuro.MVPA.mvpa_functions import load_pickle