'''
File: tremor_clustering_clinicalData.py
Project: 'Machine Learning for Classifying Tremor Using Smartphone Accelerometers and Clinical Features', Balachandar et. al 2020

@author: Arjun Balachandar MD

Please refer to README file for more information.
 
- This program uses unsupervised machine learning techniques to cluster data agnostically using clinical data as features and without using their known labels, and only uses known labels to color-code the data when being displayed for the user
- Specifically, the Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) is used (https://arxiv.org/abs/1802.03426), but also other methods such as t-SNE (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) can be used as well in this code
- The data features used for clustering analysis can be chosen by the user, and includes features from clinical data alone
- Note: The data features used were obtained separately (see README) and stored in .csv text files. These files are imported in this program, and various combinations of features to be used can be selected by the user.
'''

#Import all requires packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

#PATH refers to the folder path where both this file and the requires input .csv files (containing clinical data used as features) are stored
#Note: user must set PATH according to where this file and required .csv files are stored on their computer
PATH = "/Users/rambalachandar/Desktop/University of Toronto/Med School/Fasano Lab/Cincinatti Data"
data = pd.read_csv(PATH+'/tremor_clinical_spreadsheet_forAnalysis.csv')


v = [[],[],[]]

'''The following array 'covars' contains the list of variables to be used as features for training the machine learning classifiers
- using ORIGINAL VARS below (the only statistically significant features, as determined in a separate post-hoc analysis)  yielded the best clustering results
- other possible combinations of training features (i.e. covars) are commented below, and can be uncommented and used for clustering analysis as well
'''

#covars = ['Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant']

#ORIGINAL VARS
covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Spiral density (averaged R and L)','Asymmetry dominant-nondominant']

#covars = ['Upper body','Bradykinesia (R+L)','BFMDRS','Sniffing sticks (below 8 pathological, up to 12)','Asymmetry dominant-nondominant']
#covars = ['Bradykinesia (R+L)','BFMDRS','SARA (out of 40)','floating door sign','Sniffing sticks (below 8 pathological, up to 12)','Spiral Coefficient (averaged R and L)','Spiral density (averaged R and L)','Midline tremor','Asymmetry dominant-nondominant','Rest tremor','Action tremor','Upper body']
#covars = ['Bradykinesia (R+L)','Tone R+L','Parkinsonism score','Mirrorig score','BFMDRS','SARA (out of 40)','nystagmus','cerebellar speech','saccades latency','saccades accuracy','saccades speed','floating door sign','Sniffing sticks (below 8 pathological, up to 12)','Spiral Coefficient (averaged R and L)','Spiral density (averaged R and L)','Midline tremor','Dominant body','Non-dominant body','Asymmetry dominant-nondominant','Rest tremor','Action tremor','Upper body','Lower body + trunk','Disability (TRS)']


'''
- Below, the data is vectorized and labelled in an array with their respective clinician-determined diagnosis in order to assign specific colours to each data point according to their diagnosis
'''
numVars = len(covars) #number of features
for index,row in data.iterrows():
    a = np.zeros(numVars)
    for i in range(numVars):
        a[i] = row[covars[i]]
    v[row['Diagnosis_num']-1].append(a)
    
vecs = np.concatenate((v[0],v[1],v[2]))
group = [];
for i in range(len(v[0])):
    group.append(0)
for i in range(len(v[1])):
    group.append(1)
for i in range(len(v[2])):
    group.append(2)

topic = ['PD','ET','DT']    
vec2 = vecs.T #transpose the vector

'''
MAIN CLUSTERING ANALYSIS:
- Below, UMAP is used for clustering analysis using the vectors defined above.
- Also, the t-SNE method can be used by uncommenting the corresponding line below, and commenting the line of code for UMAP
'''
Y1 = PCA(n_components=numVars).fit_transform(vecs)
#Y = TSNE(n_components=2).fit_transform(Y1)
Y = umap.UMAP(n_neighbors=15,
                      min_dist=0.4,
                      metric='correlation').fit_transform(vecs)


'''
GENERATING FIGURES:
- Plots are generated using matplotlib below, and data-points color-coded appropriately to be displayed for the user
'''
fig,ax = plt.subplots(figsize = (7,7))
ax.set_title('UMAP image of clinical data')
scatter_x = Y.T[0] 
scatter_y = Y.T[1]

for g in np.unique(group):
    i = np.where(group == g)
    ax.scatter(scatter_x[i],scatter_y[i],label=topic[int(g)],marker = 'o')
ax.legend()
plt.show()