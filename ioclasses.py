import scipy.io
import scipy.stats as stats
import pandas as pd
import numpy as np
from copy import copy
from functools import reduce

path = r'C:\Users\stonefly\PycharmProjects\flerp-modeling\STD_TNO_FLREP_v1.1.0\Data\InputForClass\SmartEyeFeats_pp01_s1V2.mat'

class ABiosignalInputClass(object):

    def __init__(self, path:str):
        self.srcmat = scipy.io.loadmat(path)

        self.header = str(self.srcmat['__header__'])
        self.comment = self.srcmat['comment'][0]
        self.sample_metadata = pd.DataFrame()

    def _terminate_srcmat(self):
        try:
            del self.srcmat
        except NameError:
            print('No srcmat attribute to delete')

    def GetTargetStimulusIndices(self):
        return self.sample_metadata.index[self.sample_metadata['target?'] == 1]

    def GetDistractorStimulusIndices(self):
        return self.sample_metadata.index[self.sample_metadata['target?'] == 0]

    def GetHighLoadIndices(self):
        return self.sample_metadata.index[self.sample_metadata['mathtask?'] == 1]

    def GetLowLoadIndices(self):
        return self.sample_metadata.index[self.sample_metadata['mathtask?'] == 0]

    def GetTargetIndicationIndices(self):
        return self.sample_metadata.index[self.sample_metadata['indicated?'] == 1]

    def GetDistractorIndicationIndices(self):
        return self.sample_metadata.index[self.sample_metadata['indicated?'] == 0]

    def GetHighLoadByAccuracy(self):
        highload = self.sample_metadata.iloc[self.GetHighLoadIndices()]
        correct_indices = highload.index[highload['correct'] == True]

        incorrect_indices = highload.index[highload['correct'] == False]
        return correct_indices, incorrect_indices

    def GetLowLoadByAccuracy(self):
        lowload = self.sample_metadata.iloc[self.GetLowLoadIndices()]
        correct_indices = lowload.index[lowload['correct'] == True]

        incorrect_indices = lowload.index[lowload['correct'] == False]
        return correct_indices, incorrect_indices

    def ChiSquareAccuracyBetweenConditions(self, targets_only = False):
        """
        Runs a Chi Square test on proportion of correct to incorrect answers between the low and high load conditions,
        to determine if the high load condition proportion of correct and incorrect answers differs significantly from
        the low load proportion. If the targets_only flag is set, this comparison is narrowed down to only presentation
        of visual targets, excluding distractors.
        :param targets_only:
        :return:
        """
        ll_correct_index, ll_incorrect_index = self.GetLowLoadByAccuracy()
        if not targets_only:
            # Need proportion correct under low load
            correct_proportion = ll_correct_index.shape[0] / (ll_correct_index.shape[0] + ll_incorrect_index.shape[0])

            # Need total high_load to derive expected proportions
            n_high_load = self.GetHighLoadIndices().shape[0]

            expected_correct = round(correct_proportion*n_high_load)
            expected_incorrect = n_high_load - expected_correct

            # Get number of observations of correct/incorrect and calculate chi^2
            hl_correct_index, hl_incorrect_index = self.GetHighLoadByAccuracy()
            hl_n_correct, hl_n_incorrect = hl_correct_index.shape[0], hl_incorrect_index.shape[0]
            chi_stat, p_value = stats.chisquare([hl_n_correct, hl_n_incorrect], f_exp=[expected_correct, expected_incorrect])
            return chi_stat, p_value
        if targets_only:
            # Eliminate distractors trials using set intersection
            ll_correct_target, ll_incorrect_target = np.intersect1d(ll_correct_index, self.GetTargetStimulusIndices()), \
                                                     np.intersect1d(ll_incorrect_index, self.GetTargetStimulusIndices())
            correct_proportion = ll_correct_target.shape[0] / (ll_correct_target.shape[0] + ll_incorrect_target.shape[0])

            high_load_target = np.intersect1d(self.GetHighLoadIndices(), self.GetTargetStimulusIndices())
            n_high_load_target = high_load_target.shape[0]

            expected_correct = round(correct_proportion*n_high_load_target)
            expected_incorrect = n_high_load_target-expected_correct

            hl_correct_index, hl_incorrect_index = self.GetHighLoadByAccuracy()
            hl_correct_target, hl_incorrect_target = np.intersect1d(hl_correct_index, self.GetTargetStimulusIndices()), \
                                                     np.intersect1d(hl_incorrect_index, self.GetTargetStimulusIndices())
            n_hl_correct_target, n_hl_incorrect_target = hl_correct_target.shape[0], hl_incorrect_target.shape[0]
            chi_stat, p_value = stats.chisquare((n_hl_correct_target, n_hl_incorrect_target), f_exp=(expected_correct, expected_incorrect))
            return chi_stat, p_value

## Data Description
#
# Each row represents block of 256 EEG samples starting from the point of highest eye saccade speed following stimulus
# presentation
# - begin: index of first EEG sample
# - end: index of last EEG sample. Original sample rate was 512Hz, downsampled to 256Hz. Therefore end - begin always equals 511
# - trial: trial block number
# - saccadenr: order of saccades sampled in a given block
# - stimulusnr: stimulus location revealed during the presentation
# - target?: indicates whether the revealed location was a target (1) or distractor (0)
# - begintrial: timestamp of trial block's start, in ms
# - endtrial: timestamp of trial block's end, in ms
# - indicated?: whether the given stimulus was indicated as a recalled target (1) by the participant at the end of the block
# - session: session number (participants had a break halfway through a run)
# - mathtask?: whether the math/high load condition was present during a presentation

class EEGInput(ABiosignalInputClass):

    def __init__(self, path:str):
        super().__init__(path)

        sample_meta_data_headers = np.concatenate(self.srcmat['infonames'].squeeze(), dtype = 'str')
        self.sample_metadata = pd.DataFrame(self.srcmat['sampleinfo'], columns = sample_meta_data_headers)
        # Add a column explicitly listing correctly-indicated stimuli
        self.sample_metadata['correct'] = self.sample_metadata['indicated?'] == self.sample_metadata['target?']
        self.data = self.srcmat['alldat']

        # Separate power arrays from "raw" samples
        self.alpha_power = np.vstack([self.data[i, :, -1] for i in np.arange(self.data.shape[0])])
        self.theta_power = np.vstack([self.data[i, :, -2] for i in np.arange(self.data.shape[0])])
        self.samples = [pd.DataFrame(self.data[i, :, 0:-2]) for i in np.arange(self.data.shape[0])]
        self.channel_labels = np.concatenate(self.srcmat['channel'].squeeze(), dtype= 'str')

        self._terminate_srcmat()

    # def _ReintroduceMissingRows(self):
    #     TRIAL_BLOCK = set(range(1,16))
    #
    #     session_trial_grouping = self.sample_metadata.groupby(['session', 'trial'])
    #     missing_data_trials = session_trial_grouping[session_trial_grouping['saccadenr'] < 15]
    #     missing_data_indices = missing_data_trials.index
    #
    #     for session_num, trial_num in missing_data_indices:
    #         session_filter = self.sample_metadata['session'] == session_num
    #         trial_filter = self.sample_metadata['trial'] == trial_num
    #         missing_trial = self.sample_metadata[session_filter & trial_filter]
    #
    #         missing_trial_numbers = TRIAL_BLOCK.difference(set(missing_trial['saccadenr']))
    #
    #         for missingno in missing_trial_numbers:
    #             missing_row =


    def InterElectrodeCorrelations(self, trial_indices):
        """

        :param trial_indices:
        :return: (1x496, 1x496 ndarray)
        """
        corr_arrays = []
        for trial_idx in trial_indices:
            trial_electrode_corr = self.samples[trial_idx].T.corr()
            corr_array = []
            for i in range(1,32):
                corrs = trial_electrode_corr.loc[i-1, i:31]
                corr_array.append(corrs)
            corr_array = np.concatenate(corr_array)
            corr_arrays.append(corr_array)

        cross_labels = []
        for i in range(0, 32):
            row_label = self.channel_labels[i]
            column_labels = self.channel_labels[(i+1):32]

            for column_label in column_labels:
                cross_label = '-'.join([row_label, column_label])
                cross_labels.append(cross_label)

        return cross_labels, corr_arrays

    def ElectrodeCorrelationsTargetVsDistractor(self, n_samples = 1000, random_seed = 123):
        rng = np.random.default_rng(random_seed)

        target_trials, distractor_trials = self.GetTargetStimulusIndices(), self.GetDistractorStimulusIndices()

        # Get correlations between target presentations and distractor presentations
        correlation_labels, target_correlation_series = self.InterElectrodeCorrelations(target_trials)
        correlation_labels, distractor_correlation_series = self.InterElectrodeCorrelations(distractor_trials)

        # Stack so that axis 0 is the trial observation
        target_correlation_series, distractor_correlation_series = np.vstack(target_correlation_series), np.vstack(distractor_correlation_series)

        # Approximate all possible differences between all possible pairs of trials using random sampling
        sampled_target = rng.choice(target_correlation_series, (n_samples,), replace = True, axis = 0)
        sampled_distractor = rng.choice(distractor_correlation_series, (n_samples,), replace = True, axis = 0)

        wilcoxon_stats, p_values = [], []
        # Wilcoxon Signed Rank test per pair of channels. If there are stimulus-dependent changes in pairwise electrode
        # correlation, they should be present in the form of p < 0.05 at the index positions for those pairs
        # Null hypothesis: mean pairwise correlation between target and distractor trials is identical
        # Alternative hypothesis: mean of pairwise correlation between target and distractor is significantly greater or less than each other
        for i in range(sampled_target.shape[1]):
            t_channel_samples, d_channel_samples = sampled_target[:, i], sampled_distractor[:, i]

            statistic, p_value = stats.wilcoxon(t_channel_samples, d_channel_samples)
            wilcoxon_stats.append(statistic)
            p_values.append(p_value)

        wilcoxon_stats, p_values = np.array(wilcoxon_stats), np.array(p_values)
        return correlation_labels, wilcoxon_stats, p_values



## Data Description
#
# Each row is a single stimulus presentation recorded for 2 seconds from stimulus onset
# sample_metadata:
# - trial: trial block number
# - fixnr: order of the given stimulus presentation in a block
# - stimulusnr: indicates which location was being revealed during the trial
# - math?: whether math condition was active during the presentation
# - target?: whether the presented stimulus was a target (1) or distractor (0)
# - indicated?: whether the given stimulus was indicated as a recalled target (1) by the participant at the end of the block
# - no_samples_in_timewindow: the number of eye-tracking samples recorded during the interval
# - no_samples_ontarget: number of gaze samples which were within radius to be considered on-target
# - no_samples_ontarget_AND_pupvalid: subset of above which had valid pupilometry data
#
# Each row is aggregated for a single stimulus presentation
# fixations_feats:
# - fixation_duration: total amount of time fixated on presented stimulus (seconds)
# - median_pupilsize_ontarget: self-documented
# - max_pupilsize_ontarget: self-documented
# - deltat: time between presentation and first target fixation
#
# Raw eye-tracking data at 60Hz, where each row is a gaze recording
# pupil_size:
# - reltime: time relative to stimulus onset (seconds)
# - pupilsize: measured pupil size (meters)
# - pupilsizevalid: whether the given recording is valid (1) or not (0)

class EyeInput(ABiosignalInputClass):

    def __init__(self, path:str, extra_paths = None):
        super().__init__(path)

        if extra_paths is not None:
            self._multi_init([path]+extra_paths)
        else:
            # Does not load properly with default params
            self.srcmat = scipy.io.loadmat(path, simplify_cells = True)

            self.xym = self.srcmat['xym'] # Currently unknown
            self.indx = self.srcmat['indx'] # Unknown
            self.xyq = self.srcmat['xyq'] # Unknown

            dat = self.srcmat['dat']
            self.fixation_feats = pd.DataFrame(dat['feats'], columns = dat['featlabel'])
            pupil = dat['pupilsize']
            if len(pupil.shape) < 2 or pupil.shape[1] != 3:
                pupil = np.vstack(pupil)
            self.pupillometry = pd.DataFrame(pupil, columns = dat['pupilsizelabel'])
            self.sample_metadata = pd.DataFrame(dat['info'], columns = dat['infolabel'])
            self.sample_metadata['correct'] = self.sample_metadata['indicated?'] == self.sample_metadata['target?']

            self.trans_times = dat['transtimes'] # Unknown
            self.stim_time = dat['stimtime'] # Unknown, relates to trans_times
            self.delay = dat['delay']
            self.shift_index = dat['shiftindex'] # Unknown
            self.is_multi = False

        self._terminate_srcmat()

    def _multi_init(self, paths):
        header = []
        comment =[]

        xym = []
        indx = []
        xyq = []
        fixation_feats = []
        pupillometry = []
        sample_metadata = []
        trans_times = []
        stim_time = []
        delay = []
        shift_index = []
        for i, path in enumerate(paths):
            session_number = i+1

            srcmat = scipy.io.loadmat(path, simplify_cells = True)

            header.append(str(srcmat.get('__header__')))
            comment.append(srcmat.get('comment'))

            xym.append(srcmat['xym'])
            indx.append(srcmat['indx'])
            xyq.append(srcmat['xyq'])

            dat = srcmat['dat']
            fixation_feats.append(dat['feats'])

            pupil = dat['pupilsize']
            if len(pupil.shape) < 2 or pupil.shape[1] != 3:
                pupil = np.vstack(pupil)
            pupillometry.append(pupil)

            meta = dat['info']
            meta = np.insert(meta, 0, np.full(meta.shape[0], session_number), axis = 1) # Add column for session number
            sample_metadata.append(meta)
            trans_times.append(dat['transtimes'])
            stim_time.append(dat['stimtime'])
            delay.append(dat['delay'])
            shift_index.append(dat['shiftindex'])

            metadata_labels = dat['infolabel']
            feat_labels = dat['featlabel']
            pupil_labels = dat['pupilsizelabel']

        #TODO: Variables of unclear utility/relation to EEG data remain unmerged for now
        self.header = header
        self.comment = comment
        self.xym = xym
        self.index = indx
        self.xyq = xyq
        self.trans_times = trans_times
        self.stim_time = stim_time
        self.delay = delay
        self.shift_index = shift_index

        # Prepend to account for added session number values
        pupil_df = pd.DataFrame(np.vstack(pupillometry), columns=pupil_labels)

        self.pupillometry = pupil_df

        metadata_labels = ['session']+list(metadata_labels)
        meta_df = pd.DataFrame(np.vstack(sample_metadata), columns=metadata_labels)
        meta_df['correct'] = meta_df['target?'] == meta_df['indicated?']  # Add response accuracy
        self.sample_metadata = meta_df

        fixation_df = pd.DataFrame(np.vstack(fixation_feats), columns = feat_labels)
        self.fixation_feats = fixation_df

        self.is_multi = True

    def FilterByNumberOfValidGazes(self, cutoff:int = 60, ontarget_cutoff:int = None, ontarget_pupil_cutoff:int = None):
        if ontarget_pupil_cutoff is not None:
            output_indices = self.sample_metadata.index[self.sample_metadata['no_samples_ontargetANDpupvalid'] >= ontarget_pupil_cutoff]
        elif ontarget_pupil_cutoff is not None:
            output_indices = self.sample_metadata.index[self.sample_metadata['no_samples_ontarget'] >= ontarget_cutoff]
        else:
            output_indices = self.sample_metadata.index[self.sample_metadata['no_samples_in_timewindow'] >= cutoff]

        return output_indices

    def FixationFeatsTargetVsDistractor(self, n_valid_pupil = 5, n_samples = 1000, random_seed = 123):
        rng = np.random.default_rng(random_seed)

        # Filter out trials without pupil data
        valid_indices = self.FilterByNumberOfValidGazes(ontarget_pupil_cutoff=n_valid_pupil)


        target_indices = np.intersect1d(valid_indices, self.GetTargetStimulusIndices())
        distractor_indices = np.intersect1d(valid_indices, self.GetDistractorStimulusIndices())

        target_observations = self.fixation_feats.loc[target_indices, :]
        distractor_observations = self.fixation_feats.loc[distractor_indices, :]

        # Randomly sample rows of the fixation features
        sampled_targets = rng.choice(target_observations, (n_samples,), replace = True, axis = 0)
        sampled_distractors = rng.choice(distractor_observations, (n_samples,), replace = True, axis = 0)

        col_names = target_observations.columns
        # Select alt hypotheses for each column. I suspect that fixation will be longer, median pupil size will be larger,
        # max pupil size will be larger (or not different), and deltat will be shorter for targets vs. distractors.
        alt_hypotheses = {'fixation_duration': 'greater', 'median_pupilsize_ontarget': 'greater',
                          'max_pupilsize_ontarget': 'greater', 'deltat': 'less'}
        stats_results = {col: None for col in col_names}
        # Perform Wilcoxon Signed-Rank on each column of observations to test if the differences between target feature
        # measures and distractor feature measures differ by a statistically significant amount
        for i, feature_name in enumerate(col_names):
            target_feature, distractor_feature = sampled_targets[:,i], sampled_distractors[:, i]
            result_tup = stats.wilcoxon(target_feature, distractor_feature, alternative=alt_hypotheses[feature_name])

            stats_results[col_names[i]] = result_tup

        return alt_hypotheses, stats_results