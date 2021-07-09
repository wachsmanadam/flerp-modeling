from ioclasses import EEGInput, EyeInput
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EEGFILES = ["STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp01_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp02_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp03_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp04_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp05_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp06_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp07_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp08_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp09_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp10_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp11_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp12_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp13_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp14_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp15_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp16_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp17_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp18_raw_fix_demean.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp19_raw_fix_demean.mat",
            # "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp20_raw_fix_demean.mat", # Only one session
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp21_raw_fix_demean.mat"]

EYEFILES = [("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp01_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp01_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp02_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp02_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp03_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp03_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp05_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp05_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp06_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp06_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp07_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp07_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp08_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp08_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp09_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp09_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp10_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp10_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp11_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp11_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp12_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp12_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp13_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp13_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp14_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp14_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp15_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp15_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp16_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp16_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp17_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp17_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp18_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp18_s2V2.mat"),
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp19_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp19_s2V2.mat"),
            #("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp20_s1V2.mat",), # Only one session
            ("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp21_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp21_s2V2.mat")
            ]

def TaskPerformanceReport(einput:EEGInput):
    meta = einput.sample_metadata
    session1_meta, session2_meta = meta[meta['session'] == 1], meta[meta['session'] == 2]
    total_trial_blocks = np.max(session1_meta['trial']) + np.max(session2_meta['trial'])

    all_math, no_math = meta[meta['mathtask?'] == 1], meta[meta['mathtask?'] == 0]
    all_math_target, all_math_distractor, no_math_target, no_math_distractor = all_math[all_math['target?'] == 1], \
                                                                               all_math[all_math['target?'] == 0], \
                                                                               no_math[no_math['target?'] == 1], \
                                                                               no_math[no_math['target?'] == 0]
    correct_math, incorrect_math = np.sum(all_math_target['indicated?'] == 1) + np.sum(
        all_math_distractor['indicated?'] == 0), np.sum(all_math_target['indicated?'] == 0) + np.sum(
        all_math_distractor['indicated?'] == 1)
    correct_no_math, incorrect_no_math = np.sum(no_math_target['indicated?'] == 1) + np.sum(
        no_math_distractor['indicated?'] == 0), np.sum(no_math_target['indicated?'] == 0) + np.sum(
        no_math_distractor['indicated?'] == 1)

    hl_accuracy, ll_accuracy = correct_math / (correct_math + incorrect_math), correct_no_math / (correct_no_math + incorrect_no_math)

    print('################################')
    print(f'Total number of trials: {total_trial_blocks}')
    print(f'Accuracy by condition')
    print(f'           high_load   low_load')
    print(f'correct    {correct_math}       {correct_no_math}')
    print(f'incorrect  {incorrect_math}        {incorrect_no_math}')
    print(f'percents   {hl_accuracy:.2f}       {ll_accuracy:.2f}')

    return total_trial_blocks, correct_math, incorrect_math, correct_no_math, incorrect_no_math, hl_accuracy, ll_accuracy


def AggregatedElectrodeCorrelations(*eeginputs:EEGInput):
    all_corr_btwn_subjects = []
    high_corr_btwn_subjects = []
    low_corr_btwn_subjects = []
    for eeginput in eeginputs:
        all_corr = [np.corrcoef(sample) for sample in eeginput.samples]
        mean_all_corr = np.mean(all_corr, axis=0)

        high_indices, low_indices = eeginput.GetHighLoadIndices(), eeginput.GetLowLoadIndices()
        high_corr, low_corr = [np.corrcoef(sample) for sample in eeginput.samples[high_indices]], [np.corrcoef(sample)
                                                                                                   for sample in
                                                                                                   eeginput.samples[
                                                                                                       low_indices]]
        mean_high_corr, mean_low_corr = np.mean(high_corr, axis=0), np.mean(low_corr, axis=0)

        all_corr_btwn_subjects.append(mean_all_corr)
        high_corr_btwn_subjects.append(mean_high_corr)
        low_corr_btwn_subjects.append(mean_low_corr)

    return all_corr_btwn_subjects, high_corr_btwn_subjects, low_corr_btwn_subjects

eye_list = []
for session1, session2 in EYEFILES:
    eye_list.append(EyeInput(session1), EyeInput(session2))

for eye1, eye2 in eye_list:
    pass

# eeginputs = []
# for eegpath in EEGFILES:
#     eeginputs.append(EEGInput(eegpath))

# t_blocks, hl_acc, ll_acc = [], [], []
# for einput in eeginputs:
#
#     total_trial_blocks, correct_math, incorrect_math, correct_no_math, incorrect_no_math, hl_accuracy, ll_accuracy = TaskPerformanceReport(einput)
#     t_blocks.append(total_trial_blocks)
#     hl_acc.append(hl_accuracy)
#     ll_acc.append(ll_accuracy)
#
# print(f'Mean blocks: {np.mean(t_blocks)}')
# print(f'Mean hl acc: {np.mean(hl_acc)}')
# print(f'Mean ll acc: {np.mean(ll_acc)}')

# agg_all_corr, agg_high_corr, agg_low_corr = AggregatedElectrodeCorrelations(*eeginputs)
# electrode_labels = eeginputs[0].channel_labels
#
# for title, aggregation in zip(['Mean electrode correlation total', 'Mean electrode correlation: High Load', 'Mean electrode correlation: Low Load'], [agg_all_corr, agg_high_corr, agg_low_corr]):
#     mean_agg = np.mean(aggregation, axis = 0)
#
#     plt.matshow(mean_agg, cmap = 'coolwarm')
#     x_pos = np.arange(len(electrode_labels))
#     plt.xticks(x_pos, electrode_labels, rotation = 40.0)
#     plt.yticks(x_pos, electrode_labels)
#     plt.title(title)
#
#     plt.show()
#
#
#     input('Press enter to continue')

# total_alphas = []
# total_thetas = []
# highload_alphas = []
# highload_thetas = []
# lowload_alphas = []
# lowload_thetas = []
# for einput in eeginputs:
#     high_index, low_index = einput.GetHighLoadIndices(), einput.GetLowLoadIndices()
#
#     mean_alphas, mean_thetas = np.mean(einput.alpha_power, axis = 0), np.mean(einput.theta_power, axis = 0)
#     total_alphas.append(mean_alphas)
#     total_thetas.append(mean_thetas)
#
#     high_mean_alphas, high_mean_thetas = np.mean(einput.alpha_power[high_index], axis = 0), \
#                                          np.mean(einput.theta_power[high_index], axis = 0)
#     highload_alphas.append(high_mean_alphas)
#     highload_thetas.append(high_mean_thetas)
#
#     low_mean_alphas, low_mean_thetas = np.mean(einput.alpha_power[low_index], axis=0), \
#                                          np.mean(einput.theta_power[low_index], axis=0)
#     lowload_alphas.append(low_mean_alphas)
#     lowload_thetas.append(low_mean_thetas)
#
# print(np.mean(total_alphas))
# print(np.mean(highload_alphas))
# print(np.mean(lowload_alphas))
# print('=======')
# print(np.mean(total_thetas))
# print(np.mean(highload_thetas))
# print(np.mean(lowload_thetas))



