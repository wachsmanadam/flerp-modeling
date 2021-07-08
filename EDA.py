from ioclasses import EEGInput, EyeInput
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EEGFILES = ["STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp01_raw_fix_demean.mat"]
EYEFILES = [("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp01_s1V2.mat",
            "STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp01_s2V2.mat")]

for eegpath, (eyepath1, eyepath2) in EEGFILES:
    eeg = EEGInput(eegpath)