# diagnosing-tremor-smarthphones-machinelearning
Code for project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features'

## Introduction:
These programs are used to analyze accelerometer recordings from tremor patients (**part 1**), and then uses unsupervised (**part 2**) and supervised machine learning techniques (part 3) to either cluster the data or train supervised machine learning classifiers using this data, respectively.

## Requirements:
The following must be installed to run all required programs:
- Python 3.7 or above (https://www.python.org/downloads/release/python-377/)
- scikit-learn (https://scikit-learn.org/stable/install.html)
- matplotlib (https://matplotlib.org/users/installing.html)
- UMAP (https://umap-learn.readthedocs.io/en/latest/)

## Part 1:
**Note:** To go directly to training machine learning classifiers using features (both time-series and frequency power-spectra metrics) obtained through analyzing tremor acceletometer recordings, skip to **part 2** below.

### Overview: 
This program takes the raw accelerometer data for of tremor in patients who had recordings conducted (both time series and frequency-power spectrum data) and analyzes them to produce metrics that can be compared across tremor groups (i.e. PD, ET, DT and controls). This program uses a metadata file containing the list of all patients studied (including non-identifying info e.g. initials, clinician-assigned diagnosis, age, etc) and stores this information. Using the stored basic ID info, the program then opens corresponding tremor recording files (4 per patient, each corresponding to one of the 4 recording positions) and conducts both time-series and frequency-power spectrum analysis. 

These analyses are conducted using the functions 'timeseries_analysis' and 'freq_analysis respectively', which are called in the main body of code and take as input a given tremor recording .csv file. The analysis metrics obtained from processing all the recording data are then stored in a single .csv file containing the list of all patients (i.e. initials) and all of the analysis metrics obtained for each patient in corresponding columns. Statistical analysis can then be done using this final output file to compare the values of metrics across each tremor group as a whole (conducted in SPSS for this project).

The final output file containing all analysis features for each patient can also be used as input features for machine learning algorithms (see README file for mor details).

### Configuration:

