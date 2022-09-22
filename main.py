import flask
from flask import request, jsonify
from joblib import dump, load

import os
import tarfile
import urllib.request

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score, cross_val_predict

import logging

DOWNLOAD_FILE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSL7SdxKVAe864US4WNcKQiwFNHR8qf4SG5fWy0R4wpFNX5ECGgOlrqMZ1yvGtKPG9k2lHI5caxmt_B/pub?gid=273652227&single=true&output=csv"
DATA_DIR  = "dataset_frasal"
DATA_FILE = "frasal_data.csv"
DATA_PATH = os.path.join(DATA_DIR,DATA_FILE)

app = flask.Flask(__name__)

# url per l'interrogazione di test
# http://localhost:5000/?x1=7.5&x2=12.5&x3=16.5&x4=26.5&x5=26.5&x6=4.2&x7=105&x8=115&x9=3.5
#
# url per l'addestramento
# http://localhost:5000/learn

@app.route('/', methods=['GET'])
def home():
    x1 = request.args.get('x1')
    x2 = request.args.get('x2')
    x3 = request.args.get('x3')
    x4 = request.args.get('x4')
    x5 = request.args.get('x5')
    x6 = request.args.get('x6')
    x7 = request.args.get('x7')
    x8 = request.args.get('x8')
    x9 = request.args.get('x9')

    predict_clf_D1 = load('frasal_model_D1.joblib')
    predict_clf_D2 = load('frasal_model_D2.joblib')

    st = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

    val = predict_clf_D1.predict([st]).flatten()[0]

    if (val == 'P'):
        val = predict_clf_D2.predict([st]).flatten()[0]

    return jsonify(val)

def fetch_frasal_data(file_url=DOWNLOAD_FILE,data_path=DATA_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)
   # urllib.request.urlretrieve(file_url,data_path)

def load_frasal_data(data_path=DATA_PATH):
  return pd.read_csv(data_path)

@app.route('/learn', methods=['GET'])
def learnModel():
    fetch_frasal_data()
    data = load_frasal_data()
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    training_data_01 = train_set.drop("Classe", axis=1)
    training_data_01 = training_data_01.drop("Classe N/P", axis=1)
    training_data_01 = training_data_01.values
    training_label_01 = train_set["Classe N/P"].copy()

    test_data_01 = test_set.drop("Classe", axis=1)
    test_data_01 = test_data_01.drop("Classe N/P", axis=1)
    test_data_01 = test_data_01.values
    test_label_01 = test_set["Classe N/P"].copy()

    tree_clf_01 = tree.DecisionTreeClassifier()
    tree_clf_01 = tree_clf_01.fit(training_data_01, training_label_01)

    NN_clf_01 = MLPClassifier(solver='sgd',
                              hidden_layer_sizes=(5, 2),
                              random_state=1,
                              learning_rate_init=0.001,
                              max_iter=99999)
    NN_clf_01 = NN_clf_01.fit(training_data_01, training_label_01)

    svm_clf_01 = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42, max_iter=990000))
    ])

    svm_clf_01 = svm_clf_01.fit(training_data_01, training_label_01)

    voting_clf_01 = VotingClassifier(
        estimators=[('lr', tree_clf_01), ('rf', NN_clf_01), ('svc', svm_clf_01)],
        voting='hard')

    voting_clf_01 = voting_clf_01.fit(training_data_01, training_label_01)

    data_01 = train_set.drop(train_set[train_set.Classe == 'N'].index)
    training_data = data_01.drop("Classe", axis=1)
    training_data = training_data.drop("Classe N/P", axis=1)
    training_data = training_data.values
    training_label = data_01["Classe"].copy()

    tdata_01 = test_set.drop(test_set[test_set.Classe == 'N'].index)
    test_data = tdata_01.drop("Classe", axis=1)
    test_data = test_data.drop("Classe N/P", axis=1)
    test_data = test_data.values
    test_label = tdata_01["Classe"].copy()

    tree_clf = tree.DecisionTreeClassifier()
    tree_clf = tree_clf.fit(training_data, training_label)

    parameters = {'solver': ['lbfgs'],
                  'max_iter': [5000, 10000, 40000, 100000, 900000],
                  'alpha': 10.0 ** -np.arange(1, 5),
                  'hidden_layer_sizes': np.arange(5, 10),
                  'random_state': [0, 1]}

    NN_clf_grid = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

    NN_clf_grid.fit(training_data, training_label)
    NN_clf = NN_clf_grid.best_estimator_
    NN_clf = NN_clf.fit(training_data, training_label)

    svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42, max_iter=90000))
    ])

    svm_clf = svm_clf.fit(training_data, training_label)

    voting_clf = VotingClassifier(
        estimators=[('lr', tree_clf), ('nn', NN_clf), ('svc', svm_clf)],
        voting='hard')

    voting_clf = voting_clf.fit(training_data, training_label)

    dump(voting_clf_01, 'frasal_model_D1.joblib')
    dump(voting_clf, 'frasal_model_D2.joblib')

    out_1 = (cross_val_score(voting_clf_01, test_data_01, test_label_01, cv=3, scoring="accuracy"))
    out_2 = (cross_val_score(voting_clf, test_data, test_label, cv=3, scoring="accuracy"))

    str_1 = np.array2string(out_1, precision=2, separator=',',suppress_small=True)
    str_2 = np.array2string(out_2, precision=2, separator=',', suppress_small=True)

    return "LEARNED: "+str_1+" - "+str_2;

app.run()
