from modeltesters import SVMTester, NaiveBayesTester, LogRegTester, RandomForestTester
from integrationclasses import IntegratedBiosignalClass
import os
import pickle

MODELS = [SVMTester, NaiveBayesTester, LogRegTester, RandomForestTester]
METRICS = ['accuracy', 'f1_score', 'confusion_matrix']

PARAMS = {'LinearSVC': {'penalty': 'l2', 'loss': 'hinge', 'C': 0.25, 'max_iter': 500},
               'GaussianNB': {},
               'LogisticRegression': {'C': 0.1, 'penalty': 'elasticnet', 'l1_ratio': 0.25, 'solver': 'saga', 'max_iter': 200},
               'RandomForestClassifier': {'n_estimators': 1000, 'max_depth': 20, 'max_features': 0.25,
                   'criterion': 'gini', 'n_jobs': 2}}

os.chdir('..')
subject_folders = []
for dir in os.listdir('dataset_pickles'):
    if os.path.isdir(os.path.abspath('dataset_pickles/'+dir)):
        subject_folders.append(os.path.abspath('dataset_pickles/'+dir))

subject_results = {}

for subject_path in subject_folders:
    subject_name = os.path.split(subject_path)[1]

    # Get pickles
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
        highload_input = integrated.GetModelInput_b(condition='highload', downsample_distractors=True)
        lowload_input = integrated.GetModelInput_b(condition='lowload', downsample_distractors=True)
        print(f'High load input frame shape: {highload_input.shape}')
        print(f'Low load input frame shape: {lowload_input.shape}')

        h2l_model_instance = model_class(highload_input, testframe=lowload_input)
        l2h_model_instance = model_class(lowload_input, testframe=highload_input)

        print('Subject data integrated into model')

        modelname = h2l_model_instance.name

        param = PARAMS[modelname]
        h2l_metrics = h2l_model_instance.TestTopParameters(param)
        l2h_metrics = l2h_model_instance.TestTopParameters(param)

        result = {'params': param, 'h2l_result': h2l_metrics, 'l2h_result': l2h_metrics}
        model_info[modelname] = result
        print(f'{modelname} assessed')

    subject_results[subject_name] = model_info

    print(f'Finished {subject_name} modeling')


with open('condition_crossover_results', 'wb') as f:
    pickle.dump(subject_results, f)

