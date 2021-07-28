from ioclasses import EyeInput, EEGInput
import pandas as pd
from warnings import warn

class IntegratedBiosignalClass
    def __init__(self, eyeinput:EyeInput, eeginput:EEGInput):
        self.channel_labels = eeginput.channel_labels


    def _join_and_validate(self, eye_meta:pd.DataFrame, eeg_meta:pd.DataFrame):
        # Reindex both frames to compatible indices
        eye_meta.set_index(['session', 'trial', 'fixnr'], drop = True, inplace = True)
        eye_meta.index.set_names(['session', 'trialblock', 'trial'], inplace = True)
        eeg_meta.set_index(['session', 'trial', 'saccadenr'], drop = True, inplace = True)
        eeg_meta.index.set_names(['session', 'trialblock', 'trial'], inplace = True)

        merged = eye_meta.join(eeg_meta, how = 'inner', lsuffix= '_eye', rsuffix= '_eye')

        align_answers = merged['correct_eye'] != merged['correct_eeg']
        misalignments = align_answers.sum()
        if misalignments > 0:
            warn(f"{misalignments} subject answer rows disagree, dropping rows")
            merged.drop(index = align_answers[align_answers == True])

        merged