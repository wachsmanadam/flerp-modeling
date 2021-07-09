import scipy.io
import pandas as pd
import numpy as np
from functools import reduce

path = r'C:\Users\stonefly\PycharmProjects\flerp-modeling\STD_TNO_FLREP_v1.1.0\Data\InputForClass\SmartEyeFeats_pp01_s1V2.mat'


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

class EEGInput:

    def __init__(self, path:str):
        mat = scipy.io.loadmat(path)

        self.header = str(mat['__header__'])
        self.comment = mat['comment'][0]

        sample_meta_data_headers = np.concatenate(mat['infonames'].squeeze(), dtype = 'str')
        self.sample_metadata = pd.DataFrame(mat['sampleinfo'], columns = sample_meta_data_headers)
        self.data = mat['alldat']
        # Separate power arrays from "raw" samples
        self.alpha_power = np.vstack([self.data[i, :, -1] for i in np.arange(self.data.shape[0])])
        self.theta_power = np.vstack([self.data[i, :, -2] for i in np.arange(self.data.shape[0])])
        self.samples = np.stack([self.data[i, :, 0:-2] for i in np.arange(self.data.shape[0])], axis = 0)
        self.channel_labels = np.concatenate(mat['channel'].squeeze(), dtype= 'str')

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
        highload = self.sample_metadata[self.GetHighLoadIndices()]
        correct_indices = highload.index[highload['indicated?'] == highload['target?']]

        incorrect_indices = highload.index[highload['indicated?'] != highload['target?']]
        return correct_indices, incorrect_indices

    def GetLowLoadByAccuracy(self):
        lowload = self.sample_metadata[self.GetLowLoadIndices()]
        correct_indices = lowload.index[lowload['indicated?'] == lowload['target?']]

        incorrect_indices = lowload.index[lowload['indicated?'] != lowload['target?']]
        return correct_indices, incorrect_indices

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

class EyeInput:

    def __init__(self, path:str):
        mat = scipy.io.loadmat(path, chars_as_strings = True, simplify_cells = True)

        self.header = str(mat['__header__'])
        self.comment = mat['comment'][0]

        self.xym = mat['xym'] # Currently unknown
        self.indx = mat['indx'] # Unknown
        self.xyq = mat['xyq'] # Unknown

        dat = mat['dat']
        self.fixation_feats = pd.DataFrame(dat['feats'], columns = dat['featlabel'])
        pupil = dat['pupilsize']
        if len(pupil.shape) < 2 or pupil.shape[1] != 3:
            pupil = np.vstack(pupil)
        self.pupil_size = pd.DataFrame(pupil, columns = dat['pupilsizelabel'])
        self.sample_metadata = pd.DataFrame(dat['info'], columns = dat['infolabel'])
        self.trans_times = dat['transtimes'] # Unknown
        self.stim_time = dat['stimtime'] # Unknown, relates to trans_times
        self.delay = dat['delay']
        self.shift_index = dat['shiftindex'] # Unknown

    def GetTargetStimulusIndices(self):
        return self.sample_metadata.index[self.sample_metadata['target?'] == 1]

    def GetDistractorStimulusIndices(self):
        return self.sample_metadata.index[self.sample_metadata['target?'] == 0]

    def GetHighLoadIndices(self):
        return self.sample_metadata.index[self.sample_metadata['math?'] == 1]

    def GetLowLoadIndices(self):
        return self.sample_metadata.index[self.sample_metadata['math?'] == 0]

    def GetTargetIndicationIndices(self):
        return self.sample_metadata.index[self.sample_metadata['indicated?'] == 1]

    def GetDistractorIndicationIndices(self):
        return self.sample_metadata.index[self.sample_metadata['indicated?'] == 0]

    def GetHighLoadByAccuracy(self):
        highload = self.sample_metadata[self.GetHighLoadIndices()]
        correct_indices = highload.index[highload['indicated?'] == highload['target?']]

        incorrect_indices = highload.index[highload['indicated?'] != highload['target?']]
        return correct_indices, incorrect_indices

    def GetLowLoadByAccuracy(self):
        lowload = self.sample_metadata[self.GetLowLoadIndices()]
        correct_indices = lowload.index[lowload['indicated?'] == lowload['target?']]

        incorrect_indices = lowload.index[lowload['indicated?'] != lowload['target?']]
        return correct_indices, incorrect_indices


# Time deltas from metadata