#%% IMPORTS

import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import Functions 
import PreProcessing
import texture_features as feat
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from skimage.measure import regionprops


plt.close('all')
np.random.seed(0) # To avoid changes in random choice


#%% LOAD DATA  

images_indexes=list(range(80)) # Imagens que vamos abrir
images_indexes_val=[114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133] #Imagens nas quais vamos validar
#Estas duas linhas depois vão ser substituidas por train_test_split()

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

#Load Images and Masks
flat_nodule=[]
flat_mask=[]
features = []
features_val=[]
labels = []
labels_val=[]

for n in range(134):
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
    intensity = np.mean(np.ravel(single_image))  # Ravel põe a matriz sob a forma de uma só linha
                                        # Append acrescenta esses elementos 
    # Features 2 and 3     
    maxIntensity=np.max(np.ravel(single_image))
    minIntensity=np.min(np.avel(single_image))
    
     # Feature 4
    std_dev=np.std(np.ravel(single_image))
    
    # Feature 5 to 36
    gabor_mean, gabor_dev=feat.gaborFilter(single_image)
    gabor_mean=np.dtype(float).type(gabor_mean)
    gabor0=gabor_mean[0]
    gabor1=gabor_mean[1]
    gabor2=gabor_mean[2]
    gabor3=gabor_mean[3]
    gabor4=gabor_mean[4]
    gabor5=gabor_mean[5]
    gabor6=gabor_mean[6]
    gabor7=gabor_mean[7]
    gabor8=gabor_mean[8]
    gabor9=gabor_mean[9]
    gabor10=gabor_mean[10]
    gabor11=gabor_mean[11]
    gabor12=gabor_mean[12]
    gabor13=gabor_mean[13]
    gabor14=gabor_mean[14]
    gabor15=gabor_mean[15]
    
    gabor_dev=np.dtype(float).type(gabor_dev)
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
    
    # Feature 37, 38
    int_image=x = np.uint8(255*(single_image)) # To obtain a 255 uint8 image
    hu_moments=feat.fd_hu_moments(single_image)
    hu_moments_max=np.max(hu_moments)
    hu_moments_mean=np.mean(hu_moments)
    
    # Feature 39, 40
    haralick=feat.fd_haralick(int_image)
    haralick_max=np.max(haralick)
    haralick_mean=np.mean(haralick)
    
    # Feature 41 to 46
    contrast, dissimilarity, homogeneity, energy, correlation, ASM =feat.GLCM_features(int_image)
    
    # Feature 47
    regions = regionprops(single_mask)
    eccentricity=regions[0].eccentricity
    solidity=regions[0].solidity
    
    
    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,maxIntensity, minIntensity, std_dev, gabor_dev0,gabor_dev1,gabor_dev2,gabor_dev3,gabor_dev4,gabor_dev5,
                     gabor_dev6,gabor_dev7,gabor_dev8,gabor_dev9,gabor_dev10,gabor_dev11,gabor_dev12,gabor_dev13
                     ,gabor_dev14,gabor_dev15,gabor0,gabor1,gabor2,gabor3,gabor4,gabor5,gabor6,gabor7,gabor8,
                     gabor9,gabor10,gabor11,gabor12,gabor13,gabor14,gabor15,hu_moments_max, hu_moments_mean,haralick_max, haralick_mean
                     ,contrast, dissimilarity, homogeneity, energy, correlation, ASM])

    total_texture=np.hstack(new_texture).T
    
#Data Standerization (To ensure mean=0 and std=1)
scaler = StandardScaler().fit(features)
features=scaler.transform(features)


#%% Classificators

# SVMs 

# Create the SVC model object
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(features, total_texture)
print('Linear SVM score for train: %f',svc.score(features,total_texture))

f, axarr = plt.subplots(1,3)
axarr[0].plot_decision_boundary_iris(features, svc, 'Linear SVM')

# Create the SVC model object
C = 1.0 # SVM regularization parameter
svc_kernel = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(features, total_texture)
print('RBF SVM score for train: %f',svc_kernel.score(features,total_texture))
axarr[1].plot_decision_boundary_iris(features, svc_kernel, 'SVM with RBF kernel')



n_neighbors=5
knn=Functions.KNeighbors(n_neighbors, features, total_texture) #Training K-neighbours
print('KNN score for train: %f',knn.score(features,total_texture))
axarr[2].plot_decision_boundary_iris(features, knn, 'KNN')


models = []
models.append(('LR', LogisticRegression(random_state=9))) # Logistic Regression
models.append(('KNN', KNeighborsClassifier()))#K-Neighbors

#%% TEXTURE CLASSIFICATION (VALIDATION)




texture_val=[]
labels_val = []
new_texture_val=[]
total_texture_val=[]
all_regions_val=[]

for n in images_indexes_val:
    
    single_image_val=flat_nodule[n]
    single_mask_val=flat_mask[n]
    
    #=======PRE-PROCESSING============
    
    single_image_val[single_mask_val==0]=0 # Só nos interessa os valores que existem na mask 
    
    # Gaussian
    sigma=0.5
    gauss_image_val=PreProcessing.gaussFiltering(single_image_val,sigma)
    
    #Histogram Equalization, etc..........
    
    
    #%% ========Texture classification:===========
    
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    if (texture==5):
        texture=2 # Equivalent to solid
    elif (texture>2 and texture<5):
        texture=1 # Equivalent to semi-solid
    elif (texture<=2):
        texture=0 # Equivalent to non-solid
    new_texture_val.append(texture)
    
    #%% ============Features=============
    
     # Feature 1
    intensity = np.mean(np.ravel(single_image_val))  # Ravel põe a matriz sob a forma de uma só linha
                                        # Append acrescenta esses elementos 
    std_dev=np.std(np.ravel(single_image_val))
    
    # Feature 2
    gabor_mean, gabor_dev=feat.gaborFilter(single_image_val)
    gabor_mean=np.dtype(float).type(gabor_mean)
    gabor0=gabor_mean[0]
    gabor1=gabor_mean[1]
    gabor2=gabor_mean[2]
    gabor3=gabor_mean[3]
    gabor4=gabor_mean[4]
    gabor5=gabor_mean[5]
    gabor6=gabor_mean[6]
    gabor7=gabor_mean[7]
    gabor8=gabor_mean[8]
    gabor9=gabor_mean[9]
    gabor10=gabor_mean[10]
    gabor11=gabor_mean[11]
    gabor12=gabor_mean[12]
    gabor13=gabor_mean[13]
    gabor14=gabor_mean[14]
    gabor15=gabor_mean[15]
    
    gabor_dev=np.dtype(float).type(gabor_dev)
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
    int_image_val=x = np.uint8(255*(single_image_val)) # To obtain a 255 uint8 image
    hu_moments=feat.fd_hu_moments(single_image_val)
    hu_moments_max=np.max(hu_moments)
    hu_moments_mean=np.mean(hu_moments)
    
    # Feature 5
    haralick=feat.fd_haralick(int_image_val)
    haralick_max=np.max(haralick)
    haralick_mean=np.mean(haralick)
    
    # Feature 6 
    contrast, dissimilarity, homogeneity, energy, correlation, ASM =feat.GLCM_features(int_image_val)
    
    #Concatenate all features for all Training Images in an ndarry
    features_val.append([intensity,std_dev, gabor_dev0,gabor_dev1,gabor_dev2,gabor_dev3,gabor_dev4,gabor_dev5,
                     gabor_dev6,gabor_dev7,gabor_dev8,gabor_dev9,gabor_dev10,gabor_dev11,gabor_dev12,gabor_dev13
                     ,gabor_dev14,gabor_dev15,gabor0,gabor1,gabor2,gabor3,gabor4,gabor5,gabor6,gabor7,gabor8,
                     gabor9,gabor10,gabor11,gabor12,gabor13,gabor14,gabor15,hu_moments_max, hu_moments_mean,haralick_max, haralick_mean
                     ,contrast, dissimilarity, homogeneity, energy, correlation, ASM])
    
    #total_features = np.hstack(features).T
    total_texture_val=np.hstack(new_texture_val).T
    #Data Standerization (To ensure mean=0 and std=1)

features_val=scaler.transform(features_val)
print('Linear SVM score for validation: %f',svc.score(features_val,total_texture_val))
print('RGF SVM score for validation: %f', svc_kernel.score(features_val,total_texture_val))
print('KNN score for validation: %f', knn.score(features,total_texture))

