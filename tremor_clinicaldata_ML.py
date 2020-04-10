'''
File: tremor_clinicaldata_ML.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.

- This program is used to train various machine learning classifiers (using Sklearn) using clinical data obtained from tremor patients.
- Multiple types of machine learning methods such as random forest classifiers and logistic regression can be selected as the classifier of choice by the user.
- Specifically, this code trains and tests machine learning data on the same data set using leave-one-out cross-validation (LOOCV; although any form of K-fold Cross Validation (KFCV) can be selected by the user).
- This program uses as features clinical data alone (see tremor_accelerometer_ML.py for code for classifiers trained with accelerometer data).
- The performance of classification after LOOCV (or whatever form of KFCV) is computed and displayed in text for the user.
- Note: The features used to train the classifiers were obtained in a clinical environment separately and stored in .csv text files to be used here. These files are imported in this program, and various combinations of training features can be selected by the user.
'''

'''
Clinical Data alone for training machine learning classifiers
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
- using ORIGINAL VARS below (the only statistically significant features, as determined in a separate post-hoc analysis)  yielded the best classification results
- other possible combinations of training features (i.e. covars) are commented below, and can be uncommented and used to re-train classifiers
'''

#covars = ['Bradykinesia (R+L)','Tone R+L','Parkinsonism score','Mirrorig score','BFMDRS','SARA (out of 40)','nystagmus','cerebellar speech','saccades latency','saccades accuracy','saccades speed','floating door sign','Sniffing sticks (below 8 pathological, up to 12)','Spiral Coefficient (averaged R and L)','Spiral density (averaged R and L)','Midline tremor','Dominant body','Non-dominant body','Asymmetry dominant-nondominant','Rest tremor','Action tremor','Upper body','Lower body + trunk','Disability (TRS)']
#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant']
#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Asymmetry dominant-nondominant','Upper body']

#ORIGINAL VARS:
covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','Upper body']

#NO BFMDRS:
covars = ['Bradykinesia (R+L)','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','Upper body'] #72.9%, F1-0.7

#NO other vars:
#covars = ['Bradykinesia (R+L)','BFMDRS','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','Upper body'] #79.2% - F1-0.75
#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Asymmetry dominant-nondominant','Upper body'] #83.3% - F1 0.82
#covars = ['BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','Upper body'] #85.4% - F1-0.84
#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant'] #79.2%, F1-0.75
#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Upper body'] #81.25%, F1-0.78

#Data sets:
'''
binaryClass:
-binaryClass is a String to set which classes the machine learning classifiers will be trained to classify.
-depending on the value of binaryClass, different .csv files each containing different training data and training labels are read 
-The possible values for binaryClass that the user can choose are as follows:
    - '' (i.e. empty string/no value) -> perform trinary classification, i.e. train classifier to classify PD vs. ET vs. DT. If control=True (see below), controls are included as a fourth group and a quarternary classifier is trained
    - PD -> if want to train classifier to classify PD patients vs non-PD patients
    - ET -> classify ET vs. non-ET
    - DT -> classify DT vs. non-DT
'''
binaryClass = ''
if binaryClass=='PD':
    df_totalData = pd.read_csv('tremor_clinical_spreadsheet_forAnalysis_PDvsnonPD.csv', index_col=False)
elif binaryClass=='ET':
	df_totalData = pd.read_csv('tremor_clinical_spreadsheet_forAnalysis_ETvsnonET.csv', index_col=False)
elif binaryClass=='DT':	
    df_totalData = pd.read_csv('tremor_clinical_spreadsheet_forAnalysis_DTvsnonDT.csv', index_col=False)
else:
	df_totalData = pd.read_csv('tremor_clinical_spreadsheet_forAnalysis.csv', index_col=False)   

'''
- The columns from the input text file corresponding to the features selected to be used in 'covars' are obtained below
- All features are then normalized on a scale between -1 and 1 below
'''						
df_totalData.reindex(np.random.permutation(df_totalData.index))
df_y_train = df_totalData[['Diagnosis_num']]
names = np.array(df_totalData[['Name']])
df_x_train = df_totalData[covars]
scaler = RobustScaler() #scale data to normalize
df_x_train = scaler.fit_transform(df_x_train)
df_x_train = pd.DataFrame(df_x_train, columns=covars)
x_all = df_x_train.values
y_all_labels_pre = df_y_train.values
y_all_labels_list = []

#The training data is converted one-hot-encoded form, as required by the initiation functions for the machine learning classifiers 
for y in y_all_labels_pre:
	y_all_labels_list.append(y[0])
y_all = to_one_hot(df_y_train.values,num_tremor_types)
y_all_labels = np.array(y_all_labels_list)#convert to nparray

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.3, random_state=0)

#Define model:
num_inputs = len(covars)
DT_num = 0
DT_num_pred = 0
DT_num_pred_cor = 0

#Initiate K-fold Cross-Validation
num_split = y_all_labels.size #15
num_epochs = 150
#kfold = StratifiedKFold(n_splits=7)#, random_state=seed)
cvscores = []
tot_cvscores = []
tot_val_acc = np.zeros(num_epochs)
if binaryClass:
	tot_con_matrix = [[0,0],[0,0]]
else:
	tot_con_matrix = [[0,0,0],[0,0,0],[0,0,0]]
#splt = kfold.split(x_all, y_all_labels)
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


'''
- Code for KFCV is below, and by default is done specifically as LOOCV (i.e. n_splits equals the number of training points)
- The user can choose which machine learning classifier they wish to use amongst the selection below, uncommenting the line of code corresponding to the required classifier and inputting the required initialization variables
- By default, random forest classifier is chosen
'''
#---------------------------------------
#k-FOLD CROSS VALIDATION
#---------------------------------------

num_kcross_rand = 1
for i in range(num_kcross_rand):
    kfold = KFold(n_splits=num_split, shuffle=False)# random_state=seed)
    #kfold = StratifiedKFold(n_splits=num_split, shuffle=True)# random_state=seed)
    cvscores.clear()
    '''
    Machine learning classifier of choice can be selected or inputted below, by default Logistic Regression is chosen below
    '''
    #if binaryClass=='':
    #model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
    model = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    #model = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
        #model = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
        #model = DecisionTreeClassifier(random_state=0)
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

    #ROC - computed for instances of binary classification, e.g. PD vs. non-PD
    if binaryClass != '':
        AUC = roc_auc_score(y_all_labels, y_pred)
        print('AUC: '+str(AUC))
    
    print('Ave. Accuracy: '+str(np.mean(cvscores)))
    print(np.std(cvscores))

'''
------------------------------------------
----------PROGRAM OUTPUT EXAMPLE----------
------------------------------------------

- An example of output from this program is shown here.
- Here, a logistic regression classifier is used
- The confusion matrix showing the number of true vs. predicted classes is outputed:
Confusion Matrix
[[ 3  1  1]
 [ 1 32  0]
 [ 0  4  6]]

- The classification performance showing various metrics of classification are shown, including the weighted averages of these metrics across all classes
              precision    recall  f1-score   support

           1       0.75      0.60      0.67         5
           2       0.86      0.97      0.91        33
           3       0.86      0.60      0.71        10

   micro avg       0.85      0.85      0.85        48
   macro avg       0.82      0.72      0.76        48
weighted avg       0.85      0.85      0.85        48

- The overall average accuracy across all classes is also shown:
Ave. Accuracy: 85.41666666666667
'''