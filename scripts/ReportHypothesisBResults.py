import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import OrderedDict
from scipy.stats import binomtest, wilcoxon

SUBJECTS = ['pp01', 'pp02', 'pp03', 'pp04', 'pp05', 'pp06', 'pp07', 'pp08', 'pp10', 'pp11', 'pp12', 'pp13',
            'pp14', 'pp15', 'pp16', 'pp17', 'pp18', 'pp19', 'pp21']
MODELS = ['LinearSVC', 'GaussianNB', 'LogisticRegression', 'RandomForestClassifier']

with open('../condition_crossover_results', 'rb') as f:
    crossover_results = pickle.load(f)

# Create data structures to enumerate results
h2l_svm_result, h2l_nb_result, h2l_logreg_result, h2l_randomforest_result = {}, {}, {}, {}
l2h_svm_result, l2h_nb_result, l2h_logreg_result, l2h_randomforest_result = {}, {}, {}, {}

for item in [h2l_svm_result, h2l_nb_result, h2l_logreg_result, h2l_randomforest_result, l2h_svm_result, l2h_nb_result, l2h_logreg_result, l2h_randomforest_result]:
    item['accuracy'] = []
    item['f1_score'] = []
    item['confusion_matrix'] = []
    item['params'] = []

for subject_id in SUBJECTS:
    subject_level = crossover_results[subject_id]
    linear, naivebayes, logreg, randomforest = subject_level['LinearSVC'], subject_level['GaussianNB'], \
                                               subject_level['LogisticRegression'], subject_level['RandomForestClassifier']

    h2l_linear, l2h_linear = linear['h2l_result'], linear['l2h_result']
    h2l_svm_result['accuracy'].append(h2l_linear[0])
    h2l_svm_result['f1_score'].append(h2l_linear[1])
    h2l_svm_result['confusion_matrix'].append(h2l_linear[2])
    l2h_svm_result['accuracy'].append(l2h_linear[0])
    l2h_svm_result['f1_score'].append(l2h_linear[1])
    l2h_svm_result['confusion_matrix'].append(l2h_linear[2])

    h2l_naivebayes, l2h_naivebayes = naivebayes['h2l_result'], naivebayes['l2h_result']
    h2l_nb_result['accuracy'].append(h2l_naivebayes[0])
    h2l_nb_result['f1_score'].append(h2l_naivebayes[1])
    h2l_nb_result['confusion_matrix'].append(h2l_naivebayes[2])
    l2h_nb_result['accuracy'].append(l2h_naivebayes[0])
    l2h_nb_result['f1_score'].append(l2h_naivebayes[1])
    l2h_nb_result['confusion_matrix'].append(l2h_naivebayes[2])

    h2l_logreg, l2h_logreg = logreg['h2l_result'], logreg['l2h_result']
    h2l_logreg_result['accuracy'].append(h2l_logreg[0])
    h2l_logreg_result['f1_score'].append(h2l_logreg[1])
    h2l_logreg_result['confusion_matrix'].append(h2l_logreg[2])
    l2h_logreg_result['accuracy'].append(l2h_logreg[0])
    l2h_logreg_result['f1_score'].append(l2h_logreg[1])
    l2h_logreg_result['confusion_matrix'].append(l2h_logreg[2])

    h2l_randomforest, l2h_randomforest = randomforest['h2l_result'], randomforest['l2h_result']
    h2l_randomforest_result['accuracy'].append(h2l_randomforest[0])
    h2l_randomforest_result['f1_score'].append(h2l_randomforest[1])
    h2l_randomforest_result['confusion_matrix'].append(h2l_randomforest[2])
    l2h_randomforest_result['accuracy'].append(l2h_randomforest[0])
    l2h_randomforest_result['f1_score'].append(l2h_randomforest[1])
    l2h_randomforest_result['confusion_matrix'].append(l2h_randomforest[2])

# ###############
#
data = [(h2l_svm_result, l2h_svm_result), (h2l_nb_result, l2h_nb_result), (h2l_logreg_result, l2h_logreg_result),
        (h2l_randomforest_result, l2h_randomforest_result)]
# titles = ['Linear SVM Accuracy', 'Naive Bayes Accuracy', 'Logistic Regression Accuracy', 'Random Forest Accuracy']
# fig, axes = plt.subplots(2, 2, squeeze=True)
# axes = np.ravel(axes)
# fig.set_figwidth(13)
# fig.set_figheight(10)
#
# x = np.arange(1,20)
#
# i = 0
# for ax in axes:
#     plot_title = titles[i]
#     h2l_data, l2h_data = data[i]
#
#     ax.set_ylim(0.0, 1.0)
#     ax.axhline(0.5, c='gray', ls='--')
#     ax.set_xticks(x)
#     ax.set_xticklabels(SUBJECTS, fontdict={'fontsize': 10, 'rotation': 45})
#     ax.set_yticks(np.arange(0, 1.1, 0.1))
#     ax.set_yticks(np.arange(0.05, 1.0, 0.1), minor=True)
#     ax.minorticks_on()
#     ax.tick_params(axis='x', which='minor', bottom=False)
#
#     ax.plot(x, h2l_data['accuracy'], c='red', marker='v', mfc='black', label = 'High-load to Low-load')
#     ax.plot(x, l2h_data['accuracy'], c='green', marker='^', mfc='black', label='Low-load to High-load')
#
#     ax.legend(loc='lower right')
#     ax.set_title(plot_title)
#     ax.set_ylabel('Accuracy')
#
#     i+=1
#
# plt.show()
#
# ##############
#
# titles = ['Linear SVM F1 Score', 'Naive Bayes F1 Score', 'Logistic Regression F1 Score', 'Random Forest F1 Score']
# fig, axes = plt.subplots(2, 2, squeeze=True)
# axes = np.ravel(axes)
# fig.set_figwidth(13)
# fig.set_figheight(10)
#
# x = np.arange(1,20)
#
# i = 0
# for ax in axes:
#     plot_title = titles[i]
#     h2l_data, l2h_data = data[i]
#
#     ax.set_ylim(0.0, 1.0)
#     ax.axhline(0.5, c='gray', ls='--')
#     ax.set_xticks(x)
#     ax.set_xticklabels(SUBJECTS, fontdict={'fontsize': 10, 'rotation': 45})
#     ax.set_yticks(np.arange(0, 1.1, 0.1))
#     ax.set_yticks(np.arange(0.05, 1.0, 0.1), minor=True)
#     ax.minorticks_on()
#     ax.tick_params(axis='x', which='minor', bottom=False)
#
#     ax.plot(x, h2l_data['f1_score'], c='red', marker='v', mfc='black', label = 'High-load to Low-load')
#     ax.plot(x, l2h_data['f1_score'], c='green', marker='^', mfc='black', label='Low-load to High-load')
#
#     ax.legend(loc='lower right')
#     ax.set_title(plot_title)
#     ax.set_ylabel('F1 Score')
#
#     i+=1
#
# plt.show()

#############

i = 0
for h2l_data, l2h_data in data:
    model_name = MODELS[i]
    mean_h2l_accuracy, mean_l2h_accuracy = np.mean(h2l_data['accuracy']), np.mean(l2h_data['accuracy'])
    std_h2l_accuracy, std_l2h_accuracy = np.std(h2l_data['accuracy']), np.std(l2h_data['accuracy'])
    mean_h2l_f1, mean_l2h_f1 = np.mean(h2l_data['f1_score']), np.mean(l2h_data['f1_score'])
    std_h2l_f1, std_l2h_f1 =np.std(h2l_data['f1_score']), np.std(l2h_data['f1_score'])

    # Calculate overall differences as another way to show direction of effect
    diff_accuracy = np.array(h2l_data['accuracy']) - np.array(l2h_data['accuracy'])
    summed_diff_accuracy = np.sum(diff_accuracy)
    diff_f1 = np.array(h2l_data['f1_score']) - np.array(l2h_data['f1_score'])
    summed_diff_f1 = np.sum(diff_f1)

    h2l_above_chance, l2h_above_chance = [], []
    h2l_confusion_matrices, l2h_confusion_matrices = h2l_data['confusion_matrix'], l2h_data['confusion_matrix']
    # Binomial testing of accuracy
    for j in range(len(SUBJECTS)):
        h2l_matrix, l2h_matrix = h2l_confusion_matrices[j], l2h_confusion_matrices[j]
        h2l_successes, h2l_trials = h2l_matrix[0, 0] + h2l_matrix[1, 1], np.sum(h2l_matrix)
        l2h_successes, l2h_trials = l2h_matrix[0, 0] + l2h_matrix[1, 1], np.sum(l2h_matrix)

        binom_h2l = binomtest(h2l_successes, h2l_trials, p=0.5, alternative='greater')
        binom_l2h = binomtest(l2h_successes, l2h_trials, p=0.5, alternative='greater')

        if binom_h2l.pvalue < 0.05:
            h2l_above_chance.append((SUBJECTS[j], binom_h2l))
        if binom_l2h.pvalue < 0.05:
            l2h_above_chance.append((SUBJECTS[j], binom_l2h))

    print('###########')
    print(model_name)
    print(f'Mean and std of accuracy, high-to-low cognitive load: {mean_h2l_accuracy}, {std_h2l_accuracy}')
    print(f'Mean and std of accuracy, low-to-high cognitive load: {mean_l2h_accuracy}, {std_l2h_accuracy}')
    print(f'Mean and std of f1, high-to-low cognitive load: {mean_h2l_f1}, {std_h2l_f1}')
    print(f'Mean and std of f1, low-to-high cognitive load: {mean_l2h_f1}, {std_l2h_f1}')
    print(f'Summed subject-level accuracy differences (h2l-l2h): {summed_diff_accuracy}')
    print(f'Summed subject-level f1 differences (h2l-l2h): {summed_diff_f1}')
    print(f'Number of subjects above chance (h2l), (l2h): {len(h2l_above_chance)}, {len(l2h_above_chance)}')
    print(f'h2l above chance: {h2l_above_chance}')
    print(f'l2h above chance: {l2h_above_chance}')

    wilcoxon_stat, p_value = wilcoxon(diff_accuracy, alternative='greater')
    if p_value < 0.05:
        is_accuracy_greater = True
    else:
        is_accuracy_greater = False
    print(f'Wilcoxon for accuracy difference: {wilcoxon_stat}, p = {p_value}')
    print(f'Accuracy greater?: {is_accuracy_greater}')

    wilcoxon_stat, p_value = wilcoxon(diff_f1, alternative='greater')
    if p_value < 0.05:
        is_f1_greater = True
    else:
        is_f1_greater = False
    print(f'Wilcoxon for f1 difference: {wilcoxon_stat}, p = {p_value}')
    print(f'f1 score greater?: {is_f1_greater}')
    print('###########')

    i+=1