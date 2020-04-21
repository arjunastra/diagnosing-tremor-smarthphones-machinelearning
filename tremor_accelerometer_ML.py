'''
File: tremor_accelerometer_ML.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.

- This program is used to train various machine learning classifiers (using Sklearn) using data recorded from tremor patients.
- Multiple types of machine learning methods such as random forest classifiers and logistic regression can be selected as the classifier of choice by the user.
- Specifically, this code trains and tests machine learning data on the same data set using leave-one-out cross-validation (LOOCV; although any form of K-fold Cross Validation (KFCV) can be selected by the user).
- This program uses as features either accelerometer data alone, or a combination of both accelerometer and clinical features.
- The performance of classification after LOOCV (or whatever form of KFCV) is computed and displayed in text for the user.
- Note: The features used to train the classifiers were computed separately using another code (see README) and stored in .csv text files. These files are imported in this program, and various combinations of training features can be selected by the user.
'''

#Import all required packages
from __future__ import division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
from sklearn import feature_selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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
import sklearn.linear_model as lm
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

num_tremor_types = 4 #i.e. number of tremor categories is 4 for PD, ET, DT and controls respectively

#fix random seed for reproducibility of random number generator
seed = 7
np.random.seed(seed)

#ROC AUC - function used to generate area under the curve of ROC, using inputs of predicted vs. correct category
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#one-hot encoding: convert data representation to one-hot encoding, which is required for inputs to train machine learning classifiers in Scikit-learn
def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label-1] = 1.
    return results


'''The following array 'covars' contains the list of variables to be used as features for training the machine learning classifiers
- using kinetic and rest variables alone yielded the best classification results (i.e. 'KIN + REST' covars below)
- other possible combinations of training features (i.e. covars) are commented below, and can be uncommented and used to re-train classifiers
'''

#KIN + REST - works better than just KIN
covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

#NEW SIG. VARS:
#covars = ['Peak Freq_Y_rest','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','RPC_X_bat','RPC_Z_bat','RPC_U_bat','Peak Freq_Y_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','StddevPeakAmplitude to MeanPeakAmplitude ratio_kin','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

#ONLY KIN VARIABLES, WORKS DECENTLY:
#covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin']

#BAT + KIN
#covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin']

#less variables
#covars = ['RPC_Z_kin','RPC_U_kin','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
 
#even less variables:
#covars = ['RPC_U_kin','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','TSI_rest','TSI_amplitude_rest']

#List of all features computed on accelerometer data, including those not used as training features
all_covars = ['Mean Inst. Freq_bat','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_amplitude to MeanPeakAmplitude ratio_bat','StddevPeakAmplitude to MeanPeakAmplitude ratio_bat','Peak Freq_X_bat','Peak Freq_Y_bat','Peak Freq_Z_bat','Peak Freq_U_bat','Peak Power_X_bat','Peak Power_Y_bat','Peak Power_Z_bat','Peak Power_U_bat','RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','Mean Inst. Freq_kin','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_amplitude to MeanPeakAmplitude ratio_kin','StddevPeakAmplitude to MeanPeakAmplitude ratio_kin','Peak Freq_X_kin','Peak Freq_Y_kin','Peak Freq_Z_kin','Peak Freq_U_kin','Peak Power_X_kin','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','Mean Inst. Freq_out','TSI_out','TSI_amplitude_out','Mean Peak Amplitude_out','Stddev Peak Amplitude_+out','TSI_amplitude to MeanPeakAmplitude ratio_out','StddevPeakAmplitude to MeanPeakAmplitude ratio_out','Peak Freq_X_out','Peak Freq_Y_out','Peak Freq_Z_out','Peak Freq_U_out','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','RPC_X_out','RPC_Y_out','RPC_Z_out','RPC_U_out','Mean Inst. Freq_rest','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest','TSI_amplitude to MeanPeakAmplitude ratio_rest','StddevPeakAmplitude to MeanPeakAmplitude ratio_rest','Peak Freq_X_rest','Peak Freq_Y_rest','Peak Freq_Z_rest','Peak Freq_U_rest','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','RE_bat','RE_out']
			
#Data sets:
controls = False #If controls is True, trains classifiers including control data. If False, controls not included in analysis

featureSelection = False #Use methods of feature selection or not

'''
binaryClass:
-binaryClass is a String to set which classes the machine learning classifiers will be trained to classify.
-depending on the value of binaryClass, different .csv files each containing different training data and training labels are read 
-The possible values for binaryClass that the user can choose are as follows:
    - '' (i.e. empty string/no value) -> perform trinary classification, i.e. train classifier to classify PD vs. ET vs. DT. If control=True (see below), controls are included as a fourth group and a quarternary classifier is trained
    - PD -> if want to train classifier to classify PD patients vs non-PD patients
    - ET -> classify ET vs. non-ET
    - DT -> classify DT vs. non-DT
    - PDET -> classify combined group of PD & ET cases vs. all other cases
    - sameAsClinical+clinicalVars -> if want to train trinary classifier using both accelerometer data and clinical data as features (the specific clinical data to be used as features can be selected below). Note: only a subset of the total patients analyzed also had clinical data obtained, hence this training set is smaller
    - sameAsClinical -> train classifier on accelerometer data alone, but only on those patients that had clinical data obtained as well. Only a subset of the total patients analyzed also had clinical data obtained, hence this training set is smaller
    - sameAsClinical_PD -> train using same data set and features as sameAsClinical, but trained to classify PD vs non-PD
    - sameAsClinical_ET -> train using same data set and features as sameAsClinical, but trained to classify ET vs non-ET

controls:
- controls is a boolean used to set if control data is included as an extra class or not
'''
binaryClass = 'PD'
if binaryClass=='PD':
    if controls == True:
        #below file: Controls and other non-PD added into analysis as 'non-PD'
        df_totalData = pd.read_csv('AllAnalysisData_training_PDvsnon-PD_+controlsOnly.csv', index_col=False)
    else:
        df_totalData = pd.read_csv('AllAnalysisData_training_PDvsnon-PD.csv', index_col=False)
elif binaryClass=='ET':					
	df_totalData = pd.read_csv('AllAnalysisData_training_ETvsnon-ET.csv', index_col=False)
elif binaryClass=='DT':	
        df_totalData = pd.read_csv('AllAnalysisData_training_DTvsnon-DT.csv', index_col=False)
elif binaryClass=='PDET':
    df_totalData = pd.read_csv('AllAnalysisData_training_PDvsET.csv', index_col=False)
elif binaryClass=='sameAsClinical':
    df_totalData = pd.read_csv('AllAnalysisData_training_sameAsClinical.csv', index_col=False)
elif binaryClass=='sameAsClinical_PD':
    df_totalData = pd.read_csv('AllAnalysisData_training_sameAsClinical_PDvsnonPD.csv', index_col=False)
elif binaryClass=='sameAsClinical_ET':
    df_totalData = pd.read_csv('AllAnalysisData_training_sameAsClinical_ETvsnonET.csv', index_col=False)
elif binaryClass=='sameAsClinical+clinicalVars':
    df_totalData = pd.read_csv('AllAnalysisData_training_sameAsClinical_+clinicalVars.csv', index_col=False)
    
    '''
    - Various combinations of training features (i.e. covars) can be selected below by uncommenting
    - By default, KIN+REST accelerometer variables with the size siginficant clinical features is selected below
    '''
    
    #KIN + REST ONLY
    covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    #covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    
    #KIN + REST (minus BFMDRS):
    #covars = ['Upper body','Bradykinesia (R+L)','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    
    #ONLY BFMDRS ADDED:
    #covars = ['BFMDRS','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    #Less variables (best)
    #covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_U_kin','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','TSI_rest','TSI_amplitude_rest']

    #Less variables (NO BFMDRS added):
    #covars = ['Upper body','Bradykinesia (R+L)','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_U_kin','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','TSI_rest','TSI_amplitude_rest']

    #Less vars (ONLY BFMDRS added):
    #covars = ['BFMDRS','RPC_U_kin','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','TSI_rest','TSI_amplitude_rest']

else:
    if controls == False:
        #Trinary classification:
        df_totalData = pd.read_csv('AllAnalysisData_training1.csv', index_col=False)
    else:
        #Quarternary classification (i.e. including controls):
        df_totalData = pd.read_csv('AllAnalysisData_training1_+controlsOnly.csv', index_col=False)

'''
- The columns from the input text file corresponding to the features selected to be used in 'covars' are obtained below
- All features are then normalized on a scale between -1 and 1 below
'''
df_totalData.reindex(np.random.permutation(df_totalData.index))
df_x_train = df_totalData[covars]
df_x_train_allCovars = df_totalData[all_covars] #full dataset w/ all features, used later for finding best features
df_y_train = df_totalData[['Diagnosis_num']]
names = np.array(df_totalData[['Name']])
scaler = RobustScaler() #scale data to normalize
df_x_train = scaler.fit_transform(df_x_train)
df_x_train = pd.DataFrame(df_x_train, columns=covars)
x_all = df_x_train.values

#Full dataset w/ all features, used later for finding best features
df_x_train_allCovars = scaler.fit_transform(df_x_train_allCovars)
df_x_train_allCovars = pd.DataFrame(df_x_train_allCovars, columns=all_covars)
x_allCovars = df_x_train_allCovars.values

#The training data is converted one-hot-encoded form, as required by the initiation functions for the machine learning classifiers 
y_all_labels_pre = df_y_train.values
y_all_labels_list = []
for y in y_all_labels_pre:
	y_all_labels_list.append(y[0])
y_all = to_one_hot(df_y_train.values,num_tremor_types)
y_all_labels = np.array(y_all_labels_list)#convert to nparray

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.3, random_state=0)

#Define model:
num_inputs = len(covars) #number of features used
DT_num = 0
DT_num_pred = 0
DT_num_pred_cor = 0

#Initiate K-fold Cross-Validation
num_split = y_all_labels.size #15
num_epochs = 150
cvscores = []
tot_cvscores = []
tot_val_acc = np.zeros(num_epochs)
if binaryClass:
	tot_con_matrix = [[0,0],[0,0]]
else:
	tot_con_matrix = [[0,0,0],[0,0,0],[0,0,0]]
Y_test_tot = []
y_pred_tot = []
y_pred = []

#---------------------------------------
#NO k-FOLD CROSS VALIDATION
#---------------------------------------
'''
- NOTE: The following code can be uncommented if the user does not wish to use KFCV, but by default KFCV is used (see next section below)
'''

'''
kfold = StratifiedKFold(n_splits=num_split, shuffle=True)# random_state=seed)
for train, test in kfold.split(np.zeros(y_all_labels.size), y_all_labels):
	model = models.Sequential()
	model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(num_inputs,)))
	#model.add(Dropout(0.5))
	model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	#model.add(Dropout(0.5))
	model.add(layers.Dense(3, activation='softmax'))
	#sgd = optimizers.SGD(lr=0.4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
	history = model.fit(x_all[train],y_all[train],epochs=num_epochs,batch_size=5,validation_data=(x_all[test],y_all[test]))
	kfold = StratifiedKFold(n_splits=num_split, shuffle=True)# random_state=seed)
	break

#Recall/Precision/F-score               
print("Precision/Recall/F-score - micro")
print(precision_recall_fscore_support(Y_test_tot, y_pred_tot, average='micro'))
print("Precision/Recall/F-score - macro")
print(precision_recall_fscore_support(Y_test_tot, y_pred_tot, average='macro'))
print("Precision/Recall/F-score - weighted")
print(precision_recall_fscore_support(Y_test_tot, y_pred_tot, average='weighted'))
print(classification_report(Y_test_tot, y_pred_tot))
'''

#---------------------------------------
#k-FOLD CROSS VALIDATION
#---------------------------------------

'''
- Code for KFCV is below, and by default is done specifically as LOOCV (i.e. n_splits equals the number of training points)
- The user can choose which machine learning classifier they wish to use amongst the selection below, uncommenting the line of code corresponding to the required classifier and inputting the required initialization variables
- By default, random forest classifier is chosen
'''

num_kcross_rand = 1
for i in range(num_kcross_rand):
    kfold = KFold(n_splits=num_split, shuffle=False)
    #LOOCV selected by default, next line can be uncommented to use other structures of KFCV
    #kfold = StratifiedKFold(n_splits=num_split, shuffle=True)# random_state=seed)
    cvscores.clear()
    '''
    Machine learning classifier of choice can be selected or inputted below, by default random forest classifier is chosen
    '''
    #if binaryClass=='':
    #model = RandomForestClassifier(n_estimators=200, max_depth=10,random_state=0)
    #model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    #else:
    #model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    #model = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
    model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
#        model = DecisionTreeClassifier(random_state=0)
    #model = SVC(gamma='auto')
        #model = AdaBoostClassifier(n_estimators=100, random_state=0)
    #model = KernelRidge(alpha=1.0)
    #model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    #model = DecisionTreeClassifier(random_state=0)
    #model = SVC(gamma=2, C=1)
    #model = SVC(kernel="linear", C=0.025)
    #model = MLPClassifier(alpha=1, max_iter=1000)
    #model = GaussianProcessClassifier(1.0 * RBF(1.0))
    #model = GaussianNB()
    #model = QuadraticDiscriminantAnalysis()
    #model = KNeighborsClassifier(5)
            
    for train, test in kfold.split(np.zeros(y_all_labels.size), y_all_labels):
        model.fit(x_all[train],y_all_labels[train])
        clf = model.fit(x_all[train],y_all_labels[train])
        
        #The code commented below can be used to save the trained classifier in a file
#        model_file = open('tremor_classifier_model.joblib','wb')
#        pickle.dump(clf, model_file)
#        clf2 = pickle.load(open('tremor_classifier_model.joblib','rb')) 
        pred = clf.predict(x_all[test])
        scores = clf.score(x_all[test], y_all_labels[test])
        cvscores.append(scores * 100)
        tot_cvscores.append(scores * 100)
        y_pred.append(pred[0])
        print(names[test][0][0]+" - Actual: "+str(y_all_labels[test][0])+" - Pred: "+str(pred[0]))
    
    '''
    - Classification performance (including precision, recall and F1-score) and overall accuracy are outputted to the user in text form using the code below
    - Confusion matrix showing the specific results is also shown
    '''
    print(classification_report(y_all_labels, y_pred))
    print('Confusion Matrix')
    con_matrix = confusion_matrix(y_all_labels, y_pred)
    print(con_matrix)
    print('Accuracy (Trinary): '+str(np.mean(cvscores)))


    #ROC - computed for instances of binary classification, e.g. PD vs. non-PD
    if binaryClass != '' and binaryClass != 'sameAsClinical' and binaryClass != 'sameAsClinical+clinicalVars':
        AUC = roc_auc_score(y_all_labels, y_pred)
        print('AUC: '+str(AUC))
    
    '''Convert Trinary/Quarternary to Binary classification:
    - for each class X (e.g. X = PD, ET or DT), if trinary/quartery classification was conducted, these results are converted into X vs non-X
    - this is done by taking all predicted classes that are non-X and grouping them and comparing them to the group of X
    - the classification performance for each binary classification is then displayed for the user (see below)
    '''
    if binaryClass=='' or binaryClass=='sameAsClinical' or binaryClass=='sameAsClinical+clinicalVars':
        #Convert to PD-nonPD
        y_all_labels_PD = []
        y_pred_PD = []
        for i in range(len(y_pred)):
            if y_all_labels[i] == 1:
                y_all_labels_PD.append(0)
            else:
                y_all_labels_PD.append(1)
            if y_pred[i] == 1:
                y_pred_PD.append(0)
            else:
                y_pred_PD.append(1)
        y_all_labels_PD = np.array(y_all_labels_PD)
        y_pred_PD = np.array(y_pred_PD)
        '''Display classification performance, confusion matrix showing results of classification, and AUC of the ROC (below)
        '''
        print('Converted to PD vs non-PD')
        print(classification_report(y_all_labels_PD, y_pred_PD))
        print('Confusion Matrix PD vs non-PD')
        con_matrix = confusion_matrix(y_all_labels_PD, y_pred_PD)
        print(con_matrix)
        AUC = roc_auc_score(y_all_labels_PD, y_pred_PD)
        print('AUC: '+str(AUC))
        print('Accuracy: '+str(accuracy_score(y_all_labels_PD, y_pred_PD)))
        
        
        #Convert to ET-nonET
        y_all_labels_ET = []
        y_pred_ET = []
        for i in range(len(y_pred)):
            if y_all_labels[i] == 2:
                y_all_labels_ET.append(0)
            else:
                y_all_labels_ET.append(1)
            if y_pred[i] == 2:
                y_pred_ET.append(0)
            else:
                y_pred_ET.append(1)
        y_all_labels_ET = np.array(y_all_labels_ET)
        y_pred_ET = np.array(y_pred_ET)
        print('Converted to ET vs non-ET')
        print(classification_report(y_all_labels_ET, y_pred_ET))
        print('Confusion Matrix ET vs non-ET')
        con_matrix = confusion_matrix(y_all_labels_ET, y_pred_ET)
        print(con_matrix)
        AUC = roc_auc_score(y_all_labels_ET, y_pred_ET)
        print('AUC: '+str(AUC))
        print('Accuracy: '+str(accuracy_score(y_all_labels_ET, y_pred_ET)))
        
        #Convert to DT-nonDT
        y_all_labels_DT = []
        y_pred_DT = []
        for i in range(len(y_pred)):
            if y_all_labels[i] == 3:
                y_all_labels_DT.append(0)
            else:
                y_all_labels_DT.append(1)
            if y_pred[i] == 3:
                y_pred_DT.append(0)
            else:
                y_pred_DT.append(1)
        y_all_labels_DT = np.array(y_all_labels_DT)
        y_pred_DT = np.array(y_pred_DT)
        print('Converted to DT vs non-DT')
        print(classification_report(y_all_labels_DT, y_pred_DT))
        print('Confusion Matrix DT vs non-DT')
        con_matrix = confusion_matrix(y_all_labels_DT, y_pred_DT)
        print(con_matrix)
        AUC = roc_auc_score(y_all_labels_DT, y_pred_DT)
        print('AUC: '+str(AUC))
        print('Accuracy: '+str(accuracy_score(y_all_labels_DT, y_pred_DT)))

'''
------------------------------------------
----------PROGRAM OUTPUT EXAMPLE----------
------------------------------------------

- An example of output from this program is shown here.
- Here, a random forest classifier is used, initialized using the following parameters: RandomForestClassifier(n_estimators=200, max_depth=10,random_state=0)
- The confusion matrix showing the number of true vs. predicted classes is outputed:
Confusion Matrix
[[16  1  2]
 [ 1 37  4]
 [ 1 11  5]]

- The classification performance showing various metrics of classification are shown, including the weighted averages of these metrics across all classes
              precision    recall  f1-score   support

           1       0.89      0.84      0.86        19
           2       0.76      0.88      0.81        42
           3       0.45      0.29      0.36        17

   micro avg       0.74      0.74      0.74        78
   macro avg       0.70      0.67      0.68        78
weighted avg       0.72      0.74      0.73        78

- The overall average accuracy across all classes is also shown:
Ave. Accuracy: 74.35897435897436

- Lastly, the binary classifications derived from the above trinary classification are displayed for the user
- Below, the results of binary classification of PD vs non-PD is shown, including its classification performance, overall accuracy, and AUC of the ROC
---> converted to PD-nonPD:
              precision    recall  f1-score   support

           0       0.89      0.84      0.86        19
           1       0.95      0.97      0.96        59

   micro avg       0.94      0.94      0.94        78
   macro avg       0.92      0.90      0.91        78
weighted avg       0.94      0.94      0.94        78

Confusion Matrix
[[16 3]
 [2 57]]

AUC: 0.9041034790365745
Accuracy: 0.9358974358974359

'''