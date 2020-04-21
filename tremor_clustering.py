'''
File: tremor_clustering.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.

- This program uses unsupervised machine learning techniques to cluster data agnostically without using their known labels, and only uses known labels to color-code the data when being displayed for the user
- Specifically, the Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) is used (https://arxiv.org/abs/1802.03426), but also other methods such as t-SNE (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) can be used as well in this code
- The data features used for clustering analysis can be chosen by the user, and include accelerometer data alone, accelerometer data alone but only for patients who had clinical data, combined accelerometer and clinical data, and the use of control (non-tremor) data amongst these groups
- Note: The data features used were computed separately using another code (see README) and stored in .csv text files. These files are imported in this program, and various combinations of features to be used can be selected by the user.
'''

#Import all requires packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import umap

#PATH refers to the folder path where both this file and the requires input .csv files (containing tremor recording data used as features) are stored
#Note: user must set PATH according to where this file and required .csv files are stored on their computer
PATH = "/Users/rambalachandar/Desktop/University of Toronto/Med School/Fasano Lab/Cincinatti Data"

v = [[],[],[]]

'''
'filetype' is used below to choose features to use for clustering analysis, and hence which data file to load the required data from
-The possible values for fileType that the user can choose are as follows:
    - '' (i.e. empty string/no value) -> load all accelerometer data of PD, ET and DT but no controls
    - 'sameAsClinical' ->  load accelerometer data, but only for patients that also had clinical data obtained
    - 'sameAsClinical+clinicalVars' -> load both accelerometer and clinical data, but only for patients that also had clinical data obtained
    - 'includingControls' -> loads all accelerometer data INCLUDING for controls
    - 'includingControls_noET' -> loads all accelerometer data INCLUDING for controls, but WITHOUT ET to see effects of clustering removing ET data points
    - 'includingControlsEtc' -> loads all accelerometer data including controls AND a few recordings of non-PD/ET/DT/control patients who had other tremor syndromes (e.g. MS, FXTAS)
    - 'includingControlsEtc_noET' -> same as line above, but without ET data points
'''
fileType = 'sameAsClinical+clinicalVars' #'sameAsClinical+clinicalVars'

'''
The array 'covars' below contains the list of variables to be used as features for the clustering algorithm
- using ORIGINAL variables and KIN + REST variables (i.e. kinetic and rest variables alone) yielded the best results , shown below
- other possible combinations of training features (i.e. covars) are commented below, and can be uncommented and used to re-run the clustering algorithm
'''

if fileType == '':
    data = pd.read_csv(PATH+'/AllAnalysisData_training1.csv')
    #ORIGINAL VARIABLES:
    covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
        
    #KIN + REST - works better than just KIN
    #covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
elif fileType=='sameAsClinical':
    data = pd.read_csv(PATH+'/AllAnalysisData_training_sameAsClinical.csv')
    #ORIGINAL VARIABLES:
    #covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
        
    #KIN + REST - works better than just KIN
    covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
elif fileType=='sameAsClinical+clinicalVars':
    data = pd.read_csv(PATH+'/AllAnalysisData_training_sameAsClinical_+clinicalVars.csv')
    #covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    #covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Asymmetry dominant-nondominant','RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    
    #all clinical vars:
    covars = ['Bradykinesia (R+L)','Tone R+L','Parkinsonism score','Mirrorig score','BFMDRS','SARA (out of 40)','nystagmus','cerebellar speech','saccades latency','saccades accuracy','saccades speed','floating door sign','Sniffing sticks (below 8 pathological, up to 12)','Spiral Coefficient (averaged R and L)','Spiral density (averaged R and L)','Midline tremor','Dominant body','Non-dominant body','Asymmetry dominant-nondominant','Rest tremor','Action tremor','Upper body','Lower body + trunk','Disability (TRS)','RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    #Kin + rest only:
    #covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    #covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    #Less vars:
    #covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant','RPC_U_kin','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','TSI_rest','TSI_amplitude_rest']
elif fileType=='includingControls':
    data = pd.read_csv(PATH+'/AllAnalysisData_training1_+controls_noExtras.csv')
    #ORIGINAL VARIABLES:
    #covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    
    #NEW SIG. VARS:    
    #covars = ['Peak Freq_Y_rest','Peak Power_X_rest','Peak Power_Y_rest','Peak Power_Z_rest','Peak Power_U_rest','Peak Power_X_out','Peak Power_Y_out','Peak Power_Z_out','Peak Power_U_out','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','Peak Power_Y_kin','Peak Power_Z_kin','Peak Power_U_kin','RPC_X_bat','RPC_Z_bat','RPC_U_bat','Peak Freq_Y_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','StddevPeakAmplitude to MeanPeakAmplitude ratio_kin','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    #KIN + REST - works better than just KIN
    covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    v = [[],[],[],[]]
elif fileType=='includingControls_noET':
    #data = pd.read_csv(PATH+'/AllAnalysisData_training1_+controls_noExtras_noET.csv')
    data = pd.read_csv(PATH+'/AllAnalysisData_training1_+controls_noExtras.csv')
    #ORIGINAL VARIABLES:
    #covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
    
    #KIN + REST - works better than just KIN
    covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    v = [[],[],[],[]]
elif fileType=='includingControlsEtc':
    data = pd.read_csv(PATH+'/AllAnalysisData_training1_+controls.csv')
    #ORIGINAL VARIABLES:
    #covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
        
    #KIN + REST - works better than just KIN
    covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    v = [[],[],[],[],[],[]]
elif fileType=='includingControlsEtc_noET':
    data = pd.read_csv(PATH+'/AllAnalysisData_training1_+controls_noET.csv')
    #ORIGINAL VARIABLES:
    covars = ['RPC_X_bat','RPC_Y_bat','RPC_Z_bat','RPC_U_bat','RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_bat','TSI_amplitude_bat','Mean Peak Amplitude_bat','Stddev Peak Amplitude_+bat','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']
        
    #KIN + REST - works better than just KIN
    #covars = ['RPC_X_kin','RPC_Y_kin','RPC_Z_kin','RPC_U_kin','RPC_X_rest','RPC_Y_rest','RPC_Z_rest','RPC_U_rest','TSI_kin','TSI_amplitude_kin','Mean Peak Amplitude_kin','Stddev Peak Amplitude_+kin','TSI_rest','TSI_amplitude_rest','Mean Peak Amplitude_rest','Stddev Peak Amplitude_+rest']

    v = [[],[],[],[],[]]

numVars = len(covars) #number of features sued for clustering


'''
- Below, the data is vectorized and labelled in an array with their respective clinician-determined diagnosis in order to assign specific colours to each data point according to their diagnosis
- the colour scheme is different depending on which "diagnosis" (i.e. which of PD, ET, DT, controls, etc.) are included in the clustering
'''
for index,row in data.iterrows():
        a = np.zeros(numVars)
        for i in range(numVars):
            a[i] = row[covars[i]]
        v[row['Diagnosis_num']-1].append(a)
group = []
if fileType=='includingControlsEtc':
    vecs = np.concatenate((v[0],v[1],v[2],v[3],v[4],v[5]))
    for i in range(len(v[0])):
        group.append(0)
    for i in range(len(v[1])):
        group.append(1)
    for i in range(len(v[2])):
        group.append(2)
    for i in range(len(v[3])):
        group.append(3)
    for i in range(len(v[4])):
        group.append(4)
    for i in range(len(v[5])):
        group.append(5)
    topic = ['PD','ET','DT','FXTAS','Control','Undefined']
elif fileType=='includingControls':
    vecs = np.concatenate((v[0],v[1],v[2],v[3]))
    for i in range(len(v[0])):
        group.append(0)
    for i in range(len(v[1])):
        group.append(1)
    for i in range(len(v[2])):
        group.append(2)
    for i in range(len(v[3])):
        group.append(3)
    topic = ['PD','ET','DT','Control']
elif fileType=='includingControls_noET':
    vecs = np.concatenate((v[0],v[1],v[2],v[3]))
    for i in range(len(v[0])):
        group.append(0)
    for i in range(len(v[1])):
        group.append(1)
    for i in range(len(v[2])):
        group.append(2)
    for i in range(len(v[3])):
        group.append(3)
    topic = ['PD','ET','DT','Control']
elif fileType=='includingControlsEtc_noET':
    vecs = np.concatenate((v[0],v[1],v[2],v[3],v[4]))
    for i in range(len(v[0])):
        group.append(0)
    for i in range(len(v[1])):
        group.append(1)
    for i in range(len(v[2])):
        group.append(2)
    for i in range(len(v[3])):
        group.append(3)
    for i in range(len(v[4])):
        group.append(4)
    topic = ['PD','DT','FXTAS','Control','Undefined']
else:
    vecs = np.concatenate((v[0],v[1],v[2]))
    for i in range(len(v[0])):
        group.append(0)
    for i in range(len(v[1])):
        group.append(1)
    for i in range(len(v[2])):
        group.append(2)
    topic = ['PD','ET','DT']

vec2 = vecs.T #transpose the vector

'''Principal component analysis -> this is required for use in t-SNE clustering
'''
Y1 = PCA(n_components=numVars).fit_transform(vecs)

'''
- Below are 3 other methods of unsupervised learning that can be uncommented and used by the user, but not used in this project
- Refer to MAIN CLUSTERING ANALYSIS below if not using these 3 methods
'''
#1.-----K-Means Clustering-----
'''
kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
Y2 = kmeans.fit_transform(Y1)
#fitted = kmeans.fit(Y1)
Y = kmeans.predict(Y1)
'''

#2.-----Gaussian Mixture Model-----
'''
gmm = GaussianMixture(n_components=3).fit(Y1)
Y = gmm.predict(Y1)
probs = gmm.predict_proba(Y1)
size = 50 * probs.max(1) ** 2  # square emphasizes differences
plt.scatter(Y1[:, 0], Y1[:, 1], c=Y, cmap='viridis',s=size); #scale dot size based on prob
'''

#-3.----DSCOM-----:
#Y = DBSCAN(eps=3, min_samples=2).fit_predict(Y1)
#Y = clustering.predict(Y1)

y_true = np.array(group)
##-----NOTE: ORDER OF y_true vs Y BELOW MAY BE SWITCHED
#print(classification_report(y_true, Y))
#print('Confusion Matrix')
#con_matrix = confusion_matrix(y_true, Y)
#print(con_matrix)
#cluster_rand = metrics.adjusted_rand_score(y_true,Y)
#print(cluster_rand)

'''
MAIN CLUSTERING ANALYSIS:
- Below, UMAP is used for clustering analysis using the vectors defined above.
- Also, the t-SNE method can be used by uncommenting the corresponding line below, and commenting the line of code for UMAP
'''

#Y = TSNE(n_components=2).fit_transform(Y1) #can uncomment this to use t-SNE
Y = umap.UMAP(n_neighbors=15,min_dist=0.4,metric='correlation').fit_transform(vecs)

'''
GENERATING FIGURES:
- Plots are generated using matplotlib below, and data-points color-coded appropriately to be displayed for the user
'''
fig,ax = plt.subplots(figsize = (7,7))
ax.set_title('')#UMAP Clustering of Tremor Data')
scatter_x = Y.T[0] 
scatter_y = Y.T[1]

for g in np.unique(group):
    i = np.where(group == g)
    if fileType=='includingControls_noET':
        if topic[int(g)] != 'ET':
            ax.scatter(scatter_x[i],scatter_y[i],label=topic[int(g)],marker = "o")
        else:
            ax.scatter(scatter_x[i],scatter_y[i],label='',marker = 'None')
    else:
        ax.scatter(scatter_x[i],scatter_y[i],label=topic[int(g)],marker = "o")
ax.legend()
plt.show()