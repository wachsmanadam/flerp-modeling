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

svm_result, nb_result, logreg_result, randomforest_result = {}, {}, {}, {}

for item in [svm_result, nb_result, logreg_result, randomforest_result]:
    item['accuracy'] = []
    item['f1_score'] = []
    item['confusion_matrix'] = []
    item['params'] = []


for subject_id in SUBJECTS:
    subject_level = training_results[subject_id]
    linear, naivebayes, logreg, randomforest = subject_level['LinearSVC'], subject_level['GaussianNB'], \
                                               subject_level['LogisticRegression'], subject_level['RandomForestClassifier']

    svm_result['accuracy'].append(linear['test_metric'][0])
    svm_result['f1_score'].append(linear['test_metric'][1])
    svm_result['confusion_matrix'].append(linear['test_metric'][2])
    svm_result['params'].append(linear['params'])

    nb_result['accuracy'].append(naivebayes['test_metric'][0])
    nb_result['f1_score'].append(naivebayes['test_metric'][1])
    nb_result['confusion_matrix'].append(naivebayes['test_metric'][2])
    nb_result['params'].append(naivebayes['params'])

    logreg_result['accuracy'].append(logreg['test_metric'][0])
    logreg_result['f1_score'].append(logreg['test_metric'][1])
    logreg_result['confusion_matrix'].append(logreg['test_metric'][2])
    logreg_result['params'].append(logreg['params'])

    randomforest_result['accuracy'].append(randomforest['test_metric'][0])
    randomforest_result['f1_score'].append(randomforest['test_metric'][1])
    randomforest_result['confusion_matrix'].append(randomforest['test_metric'][2])
    randomforest_result['params'].append(randomforest['params'])

###########

x = np.arange(1,20)

fig, ax = plt.subplots()

ax.set_ylim(0.0, 1.0)
ax.axhline(0.5, c='gray', ls = '--')
ax.set_xticks(x)
ax.set_xticklabels(SUBJECTS, fontdict = {'fontsize': 10, 'rotation': 45})
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0.05, 1.0, 0.1), minor = True)
ax.minorticks_on()

ax.plot(x, svm_result['accuracy'], c='red', marker='^', mfc='black', label = 'LinearSVM')
ax.plot(x, nb_result['accuracy'], c='orange', marker='d', mfc='black', label = 'GaussianNB')
ax.plot(x, logreg_result['accuracy'], c='purple', marker='o', mfc='black', label = 'LogisticReg')
ax.plot(x, randomforest_result['accuracy'], c='green', marker='P', mfc='black', label = 'RandomForest')

ax.legend(loc = 'lower right')
ax.set_title('Accuracy Scores of tuned models')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Subject')

plt.show()

###############

print(np.mean(svm_result['accuracy']), np.std(svm_result['accuracy']))
print(np.mean(nb_result['accuracy']), np.std(nb_result['accuracy']))
print(np.mean(logreg_result['accuracy']), np.std(logreg_result['accuracy']))
print(np.mean(randomforest_result['accuracy']), np.std(randomforest_result['accuracy']))

################

svm_above_chance = []
nb_above_chance = []
logreg_above_chance = []
randomforest_above_chance = []

for i in range(0, len(SUBJECTS)):
    svm_successes = svm_result['confusion_matrix'][i][0, 0]+ svm_result['confusion_matrix'][i][1, 1]
    svm_trials = np.sum(svm_result['confusion_matrix'][i])
    nb_successes = nb_result['confusion_matrix'][i][0, 0] + nb_result['confusion_matrix'][i][1, 1]
    nb_trials = np.sum(nb_result['confusion_matrix'][i])
    logreg_successes = logreg_result['confusion_matrix'][i][0, 0] + logreg_result['confusion_matrix'][i][1, 1]
    logreg_trials = np.sum(logreg_result['confusion_matrix'][i])
    randomforest_successes = randomforest_result['confusion_matrix'][i][0, 0] + randomforest_result['confusion_matrix'][i][1, 1]
    randomforest_trials = np.sum(randomforest_result['confusion_matrix'][i])

    svm_binom = binomtest(svm_successes, svm_trials , p=0.5, alternative='greater')
    nb_binom = binomtest(nb_successes, nb_trials, p=0.5, alternative='greater')
    logreg_binom = binomtest(logreg_successes, logreg_trials, p=0.5, alternative='greater')
    randomforest_binom = binomtest(randomforest_successes, randomforest_trials, p=0.5, alternative='greater')

    if svm_binom.pvalue <= 0.05:
        svm_above_chance.append((SUBJECTS[i], svm_binom.pvalue))
    if nb_binom.pvalue <= 0.05:
        nb_above_chance.append((SUBJECTS[i], nb_binom.pvalue))
    if logreg_binom.pvalue <= 0.05:
        logreg_above_chance.append((SUBJECTS[i], logreg_binom.pvalue))
    if randomforest_binom.pvalue <= 0.05:
        randomforest_above_chance.append((SUBJECTS[i], randomforest_binom.pvalue))

print(len(svm_above_chance), svm_above_chance)
print(len(nb_above_chance), nb_above_chance)
print(len(logreg_above_chance), logreg_above_chance)
print(len(randomforest_above_chance), randomforest_above_chance)
