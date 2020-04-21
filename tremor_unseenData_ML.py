'''
File: tremor_unseenData_ML.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.

- This program is used to train various machine learning classifiers (using Sklearn) using accelerometer data obtained from tremor patients seen in one center, and tested on accelerometer data recorded on patients in another center.
- Multiple types of machine learning methods (e.g. random forest classifiers, logistic regression) can be selected as the classifier of choice by the user.
- Specifically, this code trains on a training data set and tests on a separate set of unclassified testing data (true labels of this data were revealed after this program was run and predictions were obtained from it), and hence does not use LOOCV
- This program uses as features accelerometer data alone (see tremor_accelerometer_ML.py for code for classifiers trained with accelerometer data, but trained and tested on the same data set using LOOCV).
- The performance of classification is computed and displayed in text for the user. Of note, this was not available until the true class labels of the testing data were released by the other center
- Note: The features used to train and test the classifiers were computed separately using another code (see README) and stored in .csv text files. These files are imported in this program, and various combinations of features to be used can be selected by the user.
'''

#Import all requires packages
from __future__ import division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import os
import subprocess
import math
import statistics
import numpy as np
import io
import pickle
import pprint
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

#PATH refers to the folder path where both this file and the requires input .csv files (containing tremor recording data used as features) are stored
#Note: user must set PATH according to where this file and required .csv files are stored on their computer
PATH = "/Users/rambalachandar/Desktop/University of Toronto/Med School/Fasano Lab/Cincinatti Data"
os.chdir(PATH)

num_tremor_types = 3 #i.e. number of tremor categories is 3 for PD, ET and DT respectively

#one-hot encoding: convert data representation to one-hot encoding, which is required for inputs to train machine learning classifiers in Scikit-learn
def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label-1] = 1.
    return results

'''The following array 'covars' contains the list of variables to be used as features for training the machine learning classifiers
- using ORIGINAL variables and KIN + REST variables (i.e. kinetic and rest variables alone) yielded the best classification results, shown below
- other possible combinations of training features (i.e. covars) are commented below, and can be uncommented and used to re-train classifiers
'''

#covars = ['RE_bat','RE_out','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest','RPC_X_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

#covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest','RE_bat','RE_out','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest']

#ORIGINAL VARIABLES:
covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

#KIN + REST - works better than just KIN
#covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

#covars = ['Mean Inst. Freq_bat','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_amplitude to MeanPeakAmplitude ratio_bat','StddevPeakAmplitude to MeanPeakAmplitude ratio_bat','Peak Freq_X_bat','Peak Freq_Y_bat','Peak Freq_Z_bat','Peak Freq_U_bat','Peak Power_X_bat','Peak Power_Y_bat','Peak Power_Z_bat','Peak Power_U_bat','RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','Mean Inst. Freq_kin','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_amplitude to MeanPeakAmplitude ratio_kin','StddevPeakAmplitude to MeanPeakAmplitude ratio_kin','Peak Freq_X_kin','Peak Freq_Y_kin','Peak Freq_Z_kin','Peak Freq_U_kin','Peak Power_X_kin','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','Mean Inst. Freq_out','TSI_out','TSI_amplitude_out','Mean Peak Amplitude_out','Stddev Peak Amplitude_+out','TSI_amplitude to MeanPeakAmplitude ratio_out','StddevPeakAmplitude to MeanPeakAmplitude ratio_out','Peak Freq_X_out','Peak Freq_Y_out','Peak Freq_Z_out','Peak Freq_U_out','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','RPC_X_out','RPC_Y_out','RPC_Z_out','RPC_U_out','Mean Inst. Freq_rest','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest','TSI_amplitude to MeanPeakAmplitude ratio_rest','StddevPeakAmplitude to MeanPeakAmplitude ratio_rest','Peak Freq_X_rest','Peak Freq_Y_rest','Peak Freq_Z_rest','Peak Freq_U_rest','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','RE_bat','RE_out']

#Data sets:
test =  '' #If using our own data but split to make a test subset, 'fromSplit', or using other center's data then '', or if unknown data set then 'unknown'. if 'fromMissingClinical', train on the subset of one center's patients who had clinical data also obtained and test on those from the same center that did not have clinical data obtained
controls = False #If controls is True, trains classifiers including control data (only can be True if test = '', i.e. testing on external data set). If False, controls not included in analysis

'''
binaryClass:
-binaryClass is a String to set which classes the machine learning classifiers will be trained to classify.
-depending on the value of binaryClass, different .csv files each containing different training data and training labels are read 
-The possible values for binaryClass that the user can choose are as follows:
    - '' (i.e. empty string/no value) -> perform trinary classification, i.e. train classifier to classify PD vs. ET vs. DT. If control=True (see below), controls are included as a fourth group and a quarternary classifier is trained
    - PD -> if want to train classifier to classify PD patients vs non-PD patients
    - ET -> classify ET vs. non-ET
    - DT -> classify DT vs. non-DT

controls:
- controls is a boolean used to set if control data is included as an extra class or not (only can be True if test = '', i.e. testing on external data set)
'''

binaryClass = ''
if binaryClass=='PD':
    if test ==  'fromSplit': #i.e. if use data from one centre, but split into training and testing subsets
        df_trainData = pd.read_csv('AllAnalysisData_fromSplit_training_PDvsnon-PD.csv', index_col=False)
    else: #i.e. if train on one center's data set, test on external center's data-set
        if controls == True:
            #below file: Controls and other non-PD added into analysis as 'non-PD'
            df_trainData = pd.read_csv('AllAnalysisData_training_PDvsnon-PD_+controlsOnly.csv', index_col=False)
        else:
            df_trainData = pd.read_csv('AllAnalysisData_training_PDvsnon-PD.csv', index_col=False)
elif binaryClass=='ET':					
    df_trainData = pd.read_csv('AllAnalysisData_training_ETvsnon-ET.csv', index_col=False)
elif binaryClass=='DT':	
    df_trainData = pd.read_csv('AllAnalysisData_training_DTvsnon-DT.csv', index_col=False)
else:
    if test == 'fromMissingClinical': 
        df_trainData = pd.read_csv('AllAnalysisData_training_sameAsClinical.csv', index_col=False)
    else:
        df_trainData = pd.read_csv('AllAnalysisData_training1.csv', index_col=False)

'''
- The columns from the input text file corresponding to the features selected to be used in 'covars' are obtained below
- All features are then normalized on a scale between -1 and 1 below
'''
df_trainData.reindex(np.random.permutation(df_trainData.index))
df_x_train = df_trainData[covars]
df_y_train = df_trainData[['Diagnosis_num']]
scaler = StandardScaler()
#scaler = RobustScaler() #scale data to normalize
df_x_train = scaler.fit_transform(df_x_train)
df_x_train = pd.DataFrame(df_x_train, columns=covars)
x_train = df_x_train.values
y_train_labels_pre = df_y_train.values
y_train_labels_list = []

#The training data is converted one-hot-encoded form, as required by the initiation functions for the machine learning classifiers 
for y in y_train_labels_pre:
	y_train_labels_list.append(y[0])
y_train = to_one_hot(df_y_train.values,num_tremor_types)
y_train_labels = np.array(y_train_labels_list)#convert to nparray


'''
Test data set is loaded based on specifications chosen by setting 'test' variable as needed (see above) 
'''
if test=='fromSplit':
    df_testData = pd.read_csv('AllAnalysisData_fromSplit_testing_PDvsnon-PD.csv', index_col=False)
elif test=='fromMissingClinical': #trainn on 48 sameAsClinical, test on remaining 30
    df_testData = pd.read_csv('AllAnalysisData_testing_missingClinical.csv', index_col=False)
elif test=='unknown':
    df_testData = pd.read_csv('AllAnalysisData_testing_unknownData.csv', index_col=False)
elif test=='PD':
    df_testData = pd.read_csv('AllAnalysisData_testing1__PDvsnon-PD.csv', index_col=False)
else:
    df_testData = pd.read_csv('AllAnalysisData_testing1.csv', index_col=False)
    
'''
- The columns from the input text file corresponding to the features selected to be used in 'covars' are obtained below
- All features are then normalized on a scale between -1 and 1 below
'''
df_x_test = df_testData[covars]
df_y_test = df_testData[['Diagnosis_num']]
names = np.array(df_testData[['Name']])
scaler = StandardScaler()
#scaler = RobustScaler() #scale data to normalize
df_x_test = scaler.fit_transform(df_x_test)
df_x_test = pd.DataFrame(df_x_test, columns=covars)
x_test = df_x_test.values
y_pred = df_y_test.values
y_pred_labels_pre = df_y_test.values
y_pred_labels_list = []

#The test data is converted one-hot-encoded form, as required by the initiation functions for the machine learning classifiers 
for y in y_pred_labels_pre:
	y_pred_labels_list.append(y[0])
y_pred_onehot = to_one_hot(df_y_test.values,num_tremor_types)
y_pred_labels = np.array(y_pred_labels_list) #convert to nparray

'''
Machine learning classifier of choice can be selected or inputted below, by default random forest classifier is chosen
'''
if binaryClass=='':
    #model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    #model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    #model = GaussianNB()
    model = GaussianProcessClassifier(1.0 * RBF(1.0))
    #model = MLPClassifier(alpha=1, max_iter=1000)
    #model = DecisionTreeClassifier(random_state=0)
    #model = KNeighborsClassifier(2)
    #model = QuadraticDiscriminantAnalysis()
    #model = SVC(gamma='auto')
else:
    #model = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
    #model = AdaBoostClassifier(n_estimators=100, random_state=0)
    #model = KernelRidge(alpha=1.0)
    #model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    #model = DecisionTreeClassifier(random_state=0)
    #model = SVC(gamma='auto')
    #model = SVC(gamma=2, C=1)
    #model = SVC(kernel="linear", C=0.025)
    #model = MLPClassifier(alpha=1, max_iter=1000)
    #model = GaussianProcessClassifier(1.0 * RBF(1.0))
    model = GaussianNB()
    #model = QuadraticDiscriminantAnalysis()
    #model = KNeighborsClassifier(2)
    
clf = model.fit(x_train,y_train_labels)
														
#Test on unknown data:
y_test = clf.predict(x_test)
y_test_prob = clf.predict_proba(x_test)
y_test_label = []
counts = [0,0,0,0]

print(y_test)

'''
- Classification performance (including precision, recall and F1-score) and overall accuracy are outputted to the user in text form using the code below
- Confusion matrix showing the specific results is also shown
'''
print(classification_report(y_pred,y_test))
print('Confusion Matrix')
con_matrix = confusion_matrix(y_pred,y_test)
print(con_matrix)

#ROC - computed for instances of binary classification, e.g. PD vs. non-PD
if binaryClass!='':
    AUC = roc_auc_score(y_test, y_pred)
    print('\nAUC: '+str(AUC)+'\n')

#Show the inputs and predicted outputs for each patient data point
for i in range(len(x_test)):
    UK = ''
    print(max(y_test_prob[i]))
    if max(y_test_prob[i]) <= 0.7:
        UK = ' (low prob)'
    if binaryClass=='':
        if y_test[i] == 1:
            y_test_label.append('PD'+UK)
            counts[0] += 1
        elif y_test[i] == 2:
            y_test_label.append('ET'+UK)
        		#y_test_label.append('ET'+UK)
            counts[1] += 1
        elif y_test[i] == 3:
            y_test_label.append('DT'+UK)
        		#y_test_label.append('DT'+UK)
            counts[2] += 1
    elif binaryClass=='PD':
        if y_test[i] == 0:
            y_test_label.append('PD'+UK)
            counts[0] += 1
        else:
        		y_test_label.append('non-PD'+UK)
        		counts[1] += 1
    #print("Name=%s, Predicted=%s, Prob=%s" % (names[i], y_test_label[i],y_test_prob[i]))
    #print("Predicted=%s, Prob=%s" % (y_test_label[i],y_test_prob[i]))

#Print total number of patients in each class, as predicted by the classifier
print("Predicted:")
#for i in range(len(x_train)):
for i in range(len(x_test)):
	print(y_test_label[i])
if binaryClass=='':
    print("Counts: PD: %s\nET: %s\nDT: %s\n" % (counts[0],counts[1],counts[2]))
elif binaryClass=='PD':
    print("Counts: PD: %s\nnon-PD: %s\n" % (counts[0],counts[1]))

#Save the trained model in a separate file to be loaded for future use if needed
model_file = open('tremor_classifier_model.joblib','wb')
pickle.dump(clf, model_file)
#clf2 = pickle.load(open('tremor_classifier_model.joblib','rb')) 


#Code for loading model (for later use)...
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''
