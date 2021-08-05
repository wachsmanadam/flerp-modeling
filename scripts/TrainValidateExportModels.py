from modeltesters import SVMTester, NaiveBayesTester
from integrationclasses import IntegratedBiosignalClass

MODELS = [SVMTester, NaiveBayesTester]
PARAM_GRIDS = {'LinearSVC': {'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge'], 'C': [0.1, 1.0, 10.0, 100.0]},
               'GaussianNB': {}}

import os
import pickle

import pandas as pd
import numpy as np

os.chdir('..')
subject_folders = []
for dir in os.listdir('dataset_pickles'):
    if os.path.isdir(os.path.abspath('dataset_pickles/'+dir)):
        subject_folders.append(os.path.abspath('dataset_pickles/'+dir))

subject_results = {}

for subject_path in subject_folders:
    subject_name = os.path.split(subject_path)[1]

    for pickle_path in os.listdir(subject_path):
        if "InputEEG" in pickle_path:
            with open(os.path.join(subject_path, pickle_path), 'rb') as f:
                eeg_input = pickle.load(f)
        elif pickle_path.endswith(".pickle"):
            with open(os.path.join(subject_path, pickle_path), 'rb') as f:
                eye_input = pickle.load(f)

    integrated = IntegratedBiosignalClass(eye_input, eeg_input)

    model_info = {}
    for model_class in MODELS:
        model_instance = model_class(integrated.GetModelInput_a(downsample_distractors=True))
        print('Subject data integrated into model')

        modelname = model_instance.name

        gridsearch_result = model_instance.ModelParameterSearch(PARAM_GRIDS[modelname], scoring = 'accuracy')
        best_param_set = gridsearch_result.best_params_
        model_instance.SetModelParamsAndFit(best_param_set)

        metrics = model_instance.TestTopParameters(best_param_set)

        result = {'test_metric': metrics, 'params': best_param_set}
        model_info[modelname] = result


    subject_results[subject_name] = model_info

    print(f'Finished {subject_name} modeling')

with open('subject_model_results', 'wb') as f:
    pickle.dump(subject_results, f)