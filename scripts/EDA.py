import os
import pickle

import pandas as pd
import numpy as np

os.chdir('..')

subject_folders = [os.path.abspath('dataset_pickles/'+dir) for dir in os.listdir('dataset_pickles')]


report_indices = []
report_columns = [f"n_trials", "n_high_load", "n_low_load", "n_targets", "n_distractors", "acc_low_load", "acc_low_load",
              f"overall_chisquare", "overall_pvalue", "overall_significant?", "target_chisquare", "target_pvalue",
                "target_significant?"]
report_data = []

eye_report_columns = ['fixation_duration_wilc', "fixation_duration_pvalue", "median_pupilsize_wilc", "median_pupilsize_pvalue",
                      'max_pupilsize_wilc', 'max_pupilsize_pvalue', 'deltat_wilc', "deltat_pvalue"]
eye_report_data = []

subject_EDA = {}
cross_subject_significant_electrodes = []
for subject_path in subject_folders:
    subject_name = os.path.split(subject_path)[1]
    report_indices.append(subject_name)
    subject_EDA_results = {}
    for pickle_path in os.listdir(subject_path):
        if "InputEEG" in pickle_path:
            with open(os.path.join(subject_path, pickle_path), 'rb') as f:
                eeg_input = pickle.load(f)
        elif pickle_path.endswith(".pickle"):
            with open(os.path.join(subject_path, pickle_path), 'rb') as f:
                eye_input = pickle.load(f)

    # Behavioral

    overall_chi, overall_p = eeg_input.ChiSquareAccuracyBetweenConditions()
    sig = ''
    if overall_p <= 0.05:
        sig = '*'
    subject_EDA_results['all_stimulus_chisquare'] = [overall_chi, overall_p, sig]

    target_chi, target_p = eeg_input.ChiSquareAccuracyBetweenConditions(targets_only=True)
    sig = ''
    if overall_p <= 0.05:
        sig = '*'
    subject_EDA_results['target_stimulus_chisquare'] = [target_chi, target_p, sig]

    # Electrode Correlations

    correlation_labels, electrode_wilcoxon, electrode_p = eeg_input.ElectrodeCorrelationsTargetVsDistractor(n_samples = 1200)
    significant = electrode_p <= 0.05
    significant = np.array(significant)
    correlation_labels = np.array(correlation_labels)
    sig_correlation_labels, sig_electrode_wilcoxon = correlation_labels[significant], electrode_wilcoxon[significant]

    subject_EDA_results['inter_electrode_correlation'] = [list(sig_correlation_labels), list(sig_electrode_wilcoxon), list(electrode_p[significant])]
    # Retrieve the individual electrodes from significant pairs
    sig_electrodes = set()
    for label in sig_correlation_labels:
        electrode_a, electrode_b = label.split('-')
        sig_electrodes.add(electrode_a)
        sig_electrodes.add(electrode_b)

    subject_EDA_results['significant_electrodes'] = sig_electrodes
    cross_subject_significant_electrodes.append(sig_electrodes) # Use to see the electrodes consistently significant across all subjects

    # Eye

    alt_hypotheses, stat_results = eye_input.FixationFeatsTargetVsDistractor(n_valid_pupil = 5, n_samples = 1200)

    subject_EDA_results['target_vs_distractor_fixation_feats'] = (alt_hypotheses, stat_results)

    subject_EDA[subject_name] = subject_EDA_results

    report = [eye_input.sample_metadata.shape[0],
              eye_input.GetHighLoadIndices().shape[0],
              eye_input.GetLowLoadIndices().shape[0],
              eye_input.GetTargetStimulusIndices().shape[0],
              eye_input.GetDistractorStimulusIndices().shape[0],
              eye_input.GetLowLoadByAccuracy()[0].shape[0] / eye_input.GetLowLoadIndices().shape[0],
              eye_input.GetHighLoadByAccuracy()[0].shape[0] / eye_input.GetHighLoadIndices().shape[0],
              subject_EDA_results['all_stimulus_chisquare'][0],
              subject_EDA_results['all_stimulus_chisquare'][1],
              subject_EDA_results['all_stimulus_chisquare'][2],
              subject_EDA_results['target_stimulus_chisquare'][0],
              subject_EDA_results['target_stimulus_chisquare'][1],
              subject_EDA_results['target_stimulus_chisquare'][2]]

    report_data.append(report)

    eye_report = [subject_EDA_results['target_vs_distractor_fixation_feats'][1]['fixation_duration'][0],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['fixation_duration'][1],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['median_pupilsize_ontarget'][0],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['median_pupilsize_ontarget'][1],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['max_pupilsize_ontarget'][0],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['max_pupilsize_ontarget'][1],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['deltat'][0],
                  subject_EDA_results['target_vs_distractor_fixation_feats'][1]['deltat'][1]]

    eye_report_data.append(eye_report)

    print(f'Finished subject {subject_name}')


behavior_frame = pd.DataFrame(report_data, index = report_indices, columns = report_columns)
behavior_frame.to_csv("dataset_pickles/behavior.csv")

eye_frame = pd.DataFrame(eye_report_data, index = report_indices, columns = eye_report_columns)
eye_frame.to_csv("dataset_pickles/eye_tests.csv")

electrodes = cross_subject_significant_electrodes[0]
for electrodeset in cross_subject_significant_electrodes[1::]:
    electrodes = electrodes.intersection(electrodeset)
with open("dataset_pickles/electrodes_of_note.txt", 'w') as f:
    f.write(str(electrodes))
