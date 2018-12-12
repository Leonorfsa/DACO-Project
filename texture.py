#%% IMPORTS

import numpy as np
from matplotlib import pyplot as plt
import math
import os
import pandas as pd
import Functions 
import PreProcessing
import gaborfilter as gbf
import texture_features

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc as areaUnderCurve
from sklearn import svm

from skimage.feature import shape_index
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import filters
from skimage.measure import regionprops


plt.close('all')
np.random.seed(0) # To avoid changes in random choice


#%% LOAD DATA  

images_indexes=list(range(5)) # Imagens que vamos abrir
val_images_indexes=121 #Imagens nas quais vamos validar
#Estas duas linhas depois vão ser substituidas por train_test_split()

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

#Load Images and Masks
flat_nodule=[]
flat_mask=[]
features = []
labels = []

for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    #Extract middle slice of Image and its mask
    flat_nodule.append(Functions.getMiddleSlice(nodule)) #since we have a volume we must show only a slice
    flat_mask.append(Functions.getMiddleSlice(mask))

ground_truth = pd.read_excel('ground_truth.xls') #read the metadata

#%% TEXTURE CLASSIFICATION (TRAINING)

texture=[]
labels = []
new_texture=[]
total_texture=[]
all_regions=[]

for n in images_indexes:
    
    single_image=flat_nodule[n]
    single_mask=flat_mask[n]
    
    #=======PRE-PROCESSING============
    
    single_image[single_mask==0]=0 # Só nos interessa os valores que existem na mask 
    
    # Gaussian
    sigma=0.5
    gauss_image=PreProcessing.gaussFiltering(single_image,sigma)
    
    #Histogram Equalization, etc..........
    
    
    #%% ========Texture classification:===========
    
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    new_texture.append(texture)
    if (new_texture[n]==5):
        new_texture[n]=2 # Equivalent to solid
    elif (new_texture[n]>2 and new_texture[n]<5):
        new_texture[n]=1 # Equivalent to semi-solid
    elif (new_texture[n]<=2):
        new_texture[n]=0 # Equivalent to non-solid
    
    #%% ============Features=============
    
     # Feature 1
    intensity = np.mean(np.ravel(gauss_image))  # Ravel põe a matriz sob a forma de uma só linha
                                        # Append acrescenta esses elementos 
    # Feature 2
    
    region = regionprops(single_mask.astype(int)) # avoids computing all props
    region_solidity=region[0].solidity
    region_convex = region[0].convex_area
    
    # Feature 3
    gabor_mean, gabor_dev=gbf.gaborFilter(gauss_image)
    gabor0=np.float64(gabor_mean[0])
    gabor1=np.float64(gabor_mean[1])
    gabor2=np.float64(gabor_mean[2])
    gabor3=np.float64(gabor_mean[3])
    gabor4=np.float64(gabor_mean[4])
    gabor5=np.float64(gabor_mean[5])
    gabor6=np.float64(gabor_mean[6])
    gabor7=np.float64(gabor_mean[7])
    gabor8=np.float64(gabor_mean[8])
    gabor9=np.float64(gabor_mean[9])
    gabor10=np.float64(gabor_mean[10])
    gabor11=np.float64(gabor_mean[11])
    gabor12=np.float64(gabor_mean[12])
    gabor13=np.float64(gabor_mean[13])
    gabor14=np.float64(gabor_mean[14])
    gabor15=np.float64(gabor_mean[15])
    
    gabor_dev=np.float64(gabor_dev)
    gabor_dev0=gabor_dev[0]
    gabor_dev1=gabor_dev[1]
    gabor_dev2=gabor_dev[2]
    gabor_dev3=gabor_dev[3]
    gabor_dev4=gabor_dev[4]
    gabor_dev5=gabor_dev[5]
    gabor_dev6=gabor_dev[6]
    gabor_dev7=gabor_dev[7]
    gabor_dev8=gabor_dev[8]
    gabor_dev9=gabor_dev[9]
    gabor_dev10=gabor_dev[10]
    gabor_dev11=gabor_dev[11]
    gabor_dev12=gabor_dev[12]
    gabor_dev13=gabor_dev[13]
    gabor_dev14=gabor_dev[14]
    gabor_dev15=gabor_dev[15]
    
    # Feature 4 
    int_gaus=x = np.uint8(255*(gauss_image))
    hu_moments=np.max(texture_features.fd_hu_moments(gauss_image))
    haralick=np.max(texture_features.fd_haralick(int_gaus))
    

    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,region_solidity, region_convex, gabor_dev0,gabor_dev1,gabor_dev2,gabor_dev3,gabor_dev4,gabor_dev5,
                     gabor_dev6,gabor_dev7,gabor_dev8,gabor_dev9,gabor_dev10,gabor_dev11,gabor_dev12,gabor_dev13,
                     gabor_dev14,gabor_dev15,gabor0,gabor1,gabor2,gabor3,gabor4,gabor5,gabor6,gabor7,gabor8,
                     gabor9,gabor10,gabor11,gabor12,gabor13,gabor14,gabor15,hu_moments,haralick])
    
    #total_features = np.hstack(features).T
    total_texture=np.hstack(new_texture).T
    #Data Standerization (To ensure mean=0 and std=1)

scaler = StandardScaler().fit(features)
total_features=scaler.transform(features)




# Create the SVC model object
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(total_features, total_texture)
f, axarr = plt.subplots(1,2)
axarr[0].plot_decision_boundary_iris(total_features, svc, 'Linear SVM')

# Create the SVC model object
C = 1.0 # SVM regularization parameter
svc_kernel = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(total_features, total_texture)
axarr[1].plot_decision_boundary_iris(total_features, svc_kernel, 'SVM with RBF kernel')