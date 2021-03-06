#%% IMPORTS

import numpy as np
from matplotlib import pyplot as plt
import math
import os
import pandas as pd
import Functions 
import PreProcessing

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc as areaUnderCurve

from skimage.feature import shape_index
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

plt.close('all')
np.random.seed(0) # To avoid changes in random choice


#%% LOAD DATA  

images_indexes=list(range(100)) # Imagens que vamos abrir
val_images_indexes=132 #Imagens nas quais vamos validar
#Estas duas linhas depois vão ser substituidas por train_test_split()

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

#Load Images and Masks
flat_nodule=[]
flat_mask=[]
for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    #Extract middle slice of Image and its mask
    flat_nodule.append(Functions.getMiddleSlice(nodule)) #since we have a volume we must show only a slice
    flat_mask.append(Functions.getMiddleSlice(mask))

ground_truth = pd.read_excel('ground_truth.xls') #read the metadata

#%% NODULE SEGMENTATION (TRAINING)


# For feature extraction
features = []
labels = []
totalGauss=[]

for n in images_indexes:
    
    single_image=flat_nodule[n]
    single_mask=flat_mask[n]
    #=======PRE-PROCESSING============
    
    # Gaussian
    sigma=0.5
    gauss_image=PreProcessing.gaussFiltering(single_image,sigma)
    
    #Histogram Equalization, etc..........
    
    
    
   #=======FEATURE EXTRACTION============
    
    #Collect intensity (Feature 1)
    intensity = np.ravel(gauss_image)
    
    #Collect local entropy (Feature 2)
    entrop = np.ravel(entropy(gauss_image,disk(5)))
    
    
    #Hessian Matrix
    # We will be using 7 sigmas with 0.5 step, as seen on: * por o nome do paper aqui *
    sigmas = [0.5,0.60,0.70,0.8,0.9,1,1.2,2,4,5]
    shapeind, cv, eigValues=Functions.eigenValuesShapeIndexCurveness(sigmas, single_image)
    
    eigVal0=np.ravel(eigValues[0]) # Feature 3
    eigVal1=np.ravel(eigValues[1]) # Feature 4
    shapeind=np.ravel(shapeind) #Feature 5
    cv=np.ravel(cv) #Feature 6
    
    
    #Other Features... 

    
    #Convert labeled image into a one dimensional array
    label=np.ravel(single_mask)
    labels.append(label)
    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,entrop,eigVal0, eigVal1,shapeind,cv])
    total_labels=np.hstack(labels)
    total_features = np.hstack(features)
    
#Data Standerization (To ensure mean=0 and std=1)

scaler = StandardScaler().fit(total_features)
total_features=scaler.transform(total_features)

#Sampling 
sampled_features,sampled_labels=Functions.sampling2(total_features,total_labels,50000) #To make sure we have the same number of nodule and non-nodule pixels for Training




#=======CLASSIFICATOR TRAINING============
 
#K-Neighbors
n_neighbors=5
knn=Functions.KNeighbors(n_neighbors, sampled_features, sampled_labels) #Training K-neighbours
print(knn.score(sampled_features,sampled_labels))

#SVM


#%% NODULE SEGMENTATION (TESTING)

label_val=[]
features_val = []
total_features_val=[]

val_nodule = np.load(nodules[val_images_indexes])
val_mask = np.load(masks[val_images_indexes])

#Extract middle slice of Image and its mask
val_flat_nodule=Functions.getMiddleSlice(val_nodule) #since we have a volume we must show only a slice
val_flat_mask=Functions.getMiddleSlice(val_mask)
mask_array=np.ravel(val_flat_mask)
#Functions.show2DImages(flat_nodule, flat_mask)

#=======PRE-PROCESSING============

# Gaussian
sigma=0.5
gaussImage=PreProcessing.gaussFiltering(val_flat_nodule,sigma)


#=======FEATURE EXTRACTION============

intensity = np.ravel(gaussImage)
entrop = np.ravel(entropy(gaussImage,disk(5)))

#Hessian Matrix
# We will be using 7 sigmas with 0.5 step, as seen on: * por o nome do paper aqui *
sigmas = [0.5,0.60,0.70,0.8,0.9,1,1.2,2,4,5]
shapeind, cv, eigValues=Functions.eigenValuesShapeIndexCurveness(sigmas, single_image)

eigVal0=np.ravel(eigValues[0]) # Feature 3
eigVal1=np.ravel(eigValues[1]) # Feature 4
shapeind=np.ravel(shapeind) #Feature 5
cv=np.ravel(cv) #Feature 6

#Other Features...
  
#Convert labeled image into a one dimensional array
features_val.append([intensity,entrop,eigVal0, eigVal1,shapeind,cv])
total_features_val = np.hstack(features_val).T


#Data Standerization (To ensure mean=0 and std=1)
total_features_val = scaler.transform(total_features_val)# - Para o teste, apenas aplicamos o transform que criámos com o train

#Classification
prediction=knn.predict(total_features_val)
    
#%% PREFORMANCE EVALUATION

prediction_image=np.reshape(prediction,[51,51])
Functions.show2DImages(prediction_image, val_flat_mask)

TN, FP, FN, TP =confusion_matrix(mask_array, prediction).ravel()

Acc=(TP+TN)/(TP+TN+FP+FN)            
Sens_men=TP/(TP+FN)
Spec_men=TN/(TN+FP)
Sens_women=TN/(TN+FP)
Spec_women=TP/(TP+FN)
auc=areaUnderCurve([0,1-Spec_men,1], [0,Sens_men,1]) #It's the same for both men and women
print("sklearn's AUC=",auc)


