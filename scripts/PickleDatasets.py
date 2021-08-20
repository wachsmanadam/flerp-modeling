from ioclasses import EEGInput, EyeInput
import os

EEGFILES = ["dataset/InputForClass/InputEEG_pp01_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp02_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp03_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp04_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp05_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp06_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp07_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp08_raw_fix_demean.mat",
            # "dataset/InputForClass/InputEEG_pp09_raw_fix_demean.mat", # Dropped due to more EEGs than eyes
            "dataset/InputForClass/InputEEG_pp10_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp11_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp12_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp13_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp14_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp15_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp16_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp17_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp18_raw_fix_demean.mat",
            "dataset/InputForClass/InputEEG_pp19_raw_fix_demean.mat",
            # "dataset/InputForClass/InputEEG_pp20_raw_fix_demean.mat", # Only one session
            "dataset/InputForClass/InputEEG_pp21_raw_fix_demean.mat"]

EYEFILES = [("dataset/InputForClass/SmartEyeFeats_pp01_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp01_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp02_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp02_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp03_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp03_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp04_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp04_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp05_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp05_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp06_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp06_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp07_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp07_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp08_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp08_s2V2.mat"),
            # ("dataset/InputForClass/SmartEyeFeats_pp09_s1V2.mat",
            # "dataset/InputForClass/SmartEyeFeats_pp09_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp10_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp10_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp11_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp11_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp12_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp12_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp13_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp13_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp14_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp14_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp15_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp15_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp16_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp16_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp17_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp17_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp18_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp18_s2V2.mat"),
            ("dataset/InputForClass/SmartEyeFeats_pp19_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp19_s2V2.mat"),
            #("dataset/InputForClass/SmartEyeFeats_pp20_s1V2.mat",), # Only one session
            ("dataset/InputForClass/SmartEyeFeats_pp21_s1V2.mat",
            "dataset/InputForClass/SmartEyeFeats_pp21_s2V2.mat")]

os.chdir('..')
for eeg_path, (eye_path_a, eye_path_b) in zip(EEGFILES,EYEFILES):

    number_index = eeg_path.find('_pp')
    subject_number = eeg_path[number_index+1:number_index+5]

    subject_folder = 'dataset_pickles\\'+subject_number
    os.mkdir(subject_folder)

    einput, eyeinput = EEGInput(eeg_path), EyeInput(eye_path_a, [eye_path_b])

    einput.WriteAsPickle(subject_folder)
    eyeinput.WriteAsPickle(subject_folder)