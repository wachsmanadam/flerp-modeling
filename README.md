# flerp-modeling
Repository for comparative analysis of machine learning classifiers using fixation-locked event related potentials

## Installation
This is meant to be run in a virtual environment. The core dependencies are all packages that can be found in premade environments like Anaconda, but the code is untested for those.

1. Clone the repository
2. cd into the repository from terminal and run ```py -m env venv```
3. Run ```venv/Scripts/activate```. Your terminal prompt should now be prefaced by ```(venv)```
4. Run ```pip install -r requirements.txt```
  
The environment should now be ready to run the scripts.

## Dataset Requirements  
This code uses the TNO FLERP dataset, available via a request process through [The Cognition and Neuroergonomics Collaborative Technology Alliance's website.](https://dev.cancta.net/C3DS/db_login.php)  
  
When acquired, the specific files used are located under ```Additional Data/InputForClass```. It should contain a number of .mat files with names similar to ```InputEEG_pp01_raw_fix_demean.mat``` and ```SmartEyeFeats_pp12_s2V2.mat```. Copy those files into the ```dataset``` folder.

## Running
The main execution scripts are all contained in the ```scripts``` directory. Current order is ```PickleDatasets.py```, ```EDA.py```, and ```TrainValidateExportModels.py```.

## Additional description
```ioclasses.py``` contains classes that convert and integrate the data from the source .mat files.
```integrationclasses.py``` contains a class that integrates the types from above into a single dataset and contains methods to convert that data into scikit-learn classifier compatible arrays
```modeltesters.py``` contain container classes for running hyperparameter search, training, validation, and testing pipeline using scikit-learn classifiers using the integrationclasses objects.
