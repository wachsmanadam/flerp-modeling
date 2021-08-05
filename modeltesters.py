import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

pd.set_option('display.max_columns', 32)

class AbstractModelTester:

    def __init__(self, inputframe):
        X, y = inputframe.iloc[:, 0:100], inputframe.iloc[:, 100]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        # Use a Standard Scaler trained on the scalar part of the dataset to scale test set
        self.scalar_std_scaler, self.X_train = self._build_standardized_inputframe(X_train)
        scaler, self.X_test = self._build_standardized_inputframe(X_test, self.scalar_std_scaler)
        self.y_train, self.y_test = y_train, y_test

        self.model = None

    def _build_standardized_inputframe(self, X, scalar_std_scaler=None):
        Xlist = []

        scalars = X.iloc[:, 32::]
        if scalar_std_scaler is None:
            scalar_std_scaler = StandardScaler()
        scalar_std_scaler.fit(scalars)
        standard_scalars = scalar_std_scaler.transform(scalars)  # ndarray
        for i in range(X.shape[0]):
            Xarray = []

            samples = X.iloc[i, 0:32]
            row_scalars = standard_scalars[i, :]
            for array in samples:
                array = array.reshape((1, array.shape[0]))
                standardized_array = StandardScaler().fit_transform(
                    array)  # Standardized relative to the overall signal
                Xarray.append(standardized_array.squeeze())

            Xarray = np.concatenate(Xarray, axis=0)
            Xarray = np.concatenate([Xarray, row_scalars], axis=0)
            Xlist.append(Xarray)  # 8260 length array

        Xlist = np.array(Xlist)
        return scalar_std_scaler, Xlist

    def SetModelParamsAndFit(self, param_dict):
        self.model = self.model(**param_dict)

        self.model.fit(self.X_train, self.y_train)

    def ModelParameterSearch(self, gridsearch_dict, scoring):
        cv = GridSearchCV(self.model, gridsearch_dict, scoring = scoring)
        cv.fit(self.X_train, self.y_train)

        return cv

    def TestTopParameters(self, param_dict):
        self.SetModelParamsAndFit(param_dict)

        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)

        return accuracy, f1, confusion

class SVMTester(AbstractModelTester):

    def __init__(self, inputframe:pd.DataFrame):
        super().__init__(inputframe)

        self.model = LinearSVC()

class NaiveBayesTester(AbstractModelTester):

    def __init__(self, inputframe):
        super().__init__(inputframe)

        self.model = GaussianNB()


if __name__ == "__main__":
    from ioclasses import EyeInput, EEGInput
    from integrationclasses import IntegratedBiosignalClass

    eeg = EEGInput("STD_TNO_FLREP_v1.1.0/Data/InputForClass/InputEEG_pp04_raw_fix_demean.mat")
    eye = EyeInput("STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s1V2.mat",
                   ["STD_TNO_FLREP_v1.1.0/Data/InputForClass/SmartEyeFeats_pp04_s2V2.mat"])

    integ = IntegratedBiosignalClass(eye, eeg)

    sv = NaiveBayesTester(integ.GetModelInput_a(True))

    parameters = {}
    cross_validation_search = sv.ModelParameterSearch(parameters, scoring='accuracy')

    print(cross_validation_search.best_params_)
    print(cross_validation_search.best_score_)