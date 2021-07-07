from typing import Dict

import scipy.io
import pandas as pd
import numpy as np
from dataclasses import dataclass

path = r'C:\Users\stonefly\PycharmProjects\flerp-modeling\STD_TNO_FLREP_v1.1.0\Data\InputForClass\SmartEyeFeats_pp01_s1V2.mat'


## Data Description
#
# Each row represents block of 256 EEG samples starting from the point of highest eye saccade speed following stimulus
# presentation
# - begin: timestamp in ms
# - end: timestamp in ms #TODO: why do they seem to all be 511ms in length? 256 at 256 Hz is 1 second and stimuli were presented for 1027ms
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
        self.channel_labels = np.concatenate(mat['channel'].squeeze(), dtype= 'str')

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


# Time deltas from metadata
# Create mask by labels




test_class = EyeInput(path)

metadf = test_class.sample_metadata

print(metadf.head())