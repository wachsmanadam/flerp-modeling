import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import OrderedDict
from scipy.stats import binomtest

SUBJECTS = ['pp01', 'pp02', 'pp03', 'pp04', 'pp05', 'pp06', 'pp07', 'pp08', 'pp10', 'pp11', 'pp12', 'pp13',
            'pp14', 'pp15', 'pp16', 'pp17', 'pp18', 'pp19', 'pp21']
MODELS = ['LinearSVC', 'GaussianNB', 'LogisticRegression', 'RandomForestClassifier']

with open('../subject_model_results', 'rb') as f:
    training_results = pickle.load(f)

# Create data structures to enumerate results
h2l_svm_result, h2l_nb_result, h2l_logreg_result, h2l_randomforest_result = {}, {}, {}, {}
l2h_svm_result, l2h_nb_result, l2h_logreg_result, l2h_randomforest_result = {}, {}, {}, {}

for item in [h2l_svm_result, h2l_nb_result, h2l_logreg_result, h2l_randomforest_result, l2h_svm_result, l2h_nb_result, l2h_logreg_result, l2h_randomforest_result]:
    item['accuracy'] = []
    item['f1_score'] = []
    item['confusion_matrix'] = []
    item['params'] = []

for subject_id in SUBJECTS:
    subject_level = training_results[subject_id]
    linear, naivebayes, logreg, randomforest = subject_level['LinearSVC'], subject_level['GaussianNB'], \
                                               subject_level['LogisticRegression'], subject_level['RandomForestClassifier']