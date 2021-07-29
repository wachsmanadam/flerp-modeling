from ioclasses import EyeInput, EEGInput
import numpy as np
import pandas as pd
from warnings import warn

class IntegratedBiosignalClass:
    def __init__(self, eyeinput:EyeInput, eeginput:EEGInput):
        self.channel_labels = eeginput.channel_labels

        self._join_and_validate(eyeinput, eeginput)


    def _join_and_validate(self, eyeinput:EyeInput, eeginput:EEGInput):
        invalid_eye_filter = eyeinput.FilterByNumberOfValidGazes(ontarget_pupil_cutoff=5)
        eye_meta = eyeinput.sample_metadata
        eeg_meta = eeginput.sample_metadata
        eye_meta = eye_meta.loc[invalid_eye_filter]

        # Reindex both frames to compatible indices
        eye_meta.set_index(['session', 'trial', 'fixnr'], drop = True, inplace = True)
        eye_meta.index.set_names(['session', 'trialblock', 'trial'], inplace = True)
        eeg_meta.set_index(['session', 'trial', 'saccadenr'], drop = True, inplace = True)
        eeg_meta.index.set_names(['session', 'trialblock', 'trial'], inplace = True)

        # Align features of use to the multiindex
        fixation_feats = eyeinput.fixation_feats.loc[invalid_eye_filter]
        fixation_feats.set_index(eye_meta.index, inplace=True)

        alpha_power = pd.DataFrame(eeginput.alpha_power, index = eeg_meta.index)
        theta_power = pd.DataFrame(eeginput.theta_power, index = eeg_meta.index)
        eeg_samples = zip(eeg_meta.index.values, eeginput.samples)
        eeg_samples = dict(eeg_samples)

        merged = eye_meta.join(eeg_meta, how = 'inner', lsuffix= '_eye', rsuffix= '_eeg')


        # Choose a mutual feature that is unlikely to have accidental misalignment and validate against
        align_stim = merged['stimulusnr_eye'] != merged['stimulusnr_eeg']
        misalignments = align_stim.sum()
        if misalignments > 0:
            warn(f"{misalignments} subject answer rows disagree, dropping rows")
            merged.drop(index = align_stim[align_stim == True].index)

        # Clean up redundant metadata
        merged.drop(columns = ['target?_eeg', 'indicated?_eeg', 'mathtask?', 'correct_eeg', 'stimulusnr_eeg'], inplace = True)
        merged.rename(columns={'target?_eye': 'target?', 'indicated?_eye': 'indicated?', 'math': 'mathtask?',
                               'correct_eye': 'correct', 'stimulusnr_eye': 'stimulusnr'}, inplace= True)

        self.eeg_samples = {}
        for index in merged.index.values:
            self.eeg_samples[index] = eeg_samples[index]

        self.merged_metadata = merged
        self.alpha_power = alpha_power.loc[merged.index]
        self.theta_power = theta_power.loc[merged.index]
        self.fixation_features = fixation_feats.loc[merged.index]

    def GetTargetIndices(self):
        return self.merged_metadata.index[self.merged_metadata['target?'] == 1]

    def GetDistractorIndices(self):
        return self.merged_metadata.index[self.merged_metadata['target?'] == 0]

    def GetModelInput_a(self, downsample_distractors:bool, random_seed = 1234):
        TargetIndices, DistractorIndices = self.GetTargetIndices(), self.GetDistractorIndices()

        if downsample_distractors:
            rng = np.random.default_rng(random_seed)
            DistractorIndices = rng.choice(DistractorIndices, replace = False, size=TargetIndices.shape)

        Targets = []
        Distractors = []
        for index in TargetIndices:
            Input = self._create_row_a(index, 1)
            Targets.append(Input)
        for index in DistractorIndices:
            Input = self._create_row_a(index, 0)
            Distractors.append(Input)

        # Generate labels
        Labels = []
        Labels.extend([chan+'_sample' for chan in self.channel_labels])
        Labels.extend([chan+'_alpha' for chan in self.channel_labels])
        Labels.extend([chan+'_theta' for chan in self.channel_labels])
        Labels.extend([col for col in self.fixation_features.columns])
        Labels.append('target')

        Targets.extend(Distractors)

        inputframe = pd.DataFrame(Targets, columns= Labels)
        return inputframe

    def _create_row_a(self, index:int, is_target:int):
        """
        Generate len 101, mixed-type list of features at a given index
        :param index: MultiIndex of row to extract features from
        :param is_target: 1 or 0, designates whether row is a target
        :return:
        """
        Input = []
        sample = self.eeg_samples[index].T.values
        for i in range(32):
            Input.append(sample[:, i])  # 32 x len 256 arrays

        alpha = self.alpha_power.loc[index].values
        for channel in alpha:
            Input.append(channel)  # 32
        theta = self.theta_power.loc[index].values
        for channel in theta:
            Input.append(channel)  # 32

        fix = self.fixation_features.loc[index].values
        for feat in fix:
            Input.append(feat)  # 4

        Input.append(is_target)

        return Input


if __name__ == "__main__":
    eeg = EEGInput("STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp04_raw_fix_demean.mat")
    eye = EyeInput("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s1V2.mat",
                   ["STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s2V2.mat"])

    integ = IntegratedBiosignalClass(eye, eeg)