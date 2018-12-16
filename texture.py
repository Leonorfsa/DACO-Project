#=============================================
#       FEATURES FOR TEXTURE CLASSIFICATION
#=============================================

#%% IMPORTS
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import Functions 
import PreProcessing
import texture_features as feat
from skimage.measure import regionprops
import pickle

plt.close('all')
np.random.seed(0) # To avoid changes in random choice

#%% LOAD DATA  

images_indexes=134 

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

ground_truth = pd.read_excel('ground_truth.xls') #read the metadata

#Load Images and Masks
images_3d=[]
masks_3d=[]
flat_nodule=[]
flat_mask=[]
features = []
new_texture=[]
total_labels=[]
all_regions=[]


for n in range(images_indexes):
    nodule_3d = np.load(nodules[n])
    mask_3d = np.load(masks[n])
    images_3d.append(nodule_3d)
    masks_3d.append(mask_3d)
    
    #Extract middle slice of Image and its mask
    flat_nodule.append(Functions.getMiddleSlice(nodule_3d)) #since we have a volume we must show only a slice
    flat_mask.append(Functions.getMiddleSlice(mask_3d))


#%% TEXTURE CLASSIFICATION (TRAINING)
    
    single_image=flat_nodule[n]
    single_mask=flat_mask[n]
    
    #=======PRE-PROCESSING============
    
    single_image[single_mask==0]=0 # We only want values that exist on the mask 
    
    
    
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
    regions = regionprops(np.uint8(single_mask),int_image)
    eccentricity=regions[0].eccentricity
    solidity=regions[0].solidity
    
    
    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,maxIntensity, std_dev, gabor_dev0,gabor_dev1,gabor_dev2,gabor_dev3,gabor_dev4,gabor_dev5,
                     gabor_dev6,gabor_dev7,gabor_dev8,gabor_dev9,gabor_dev10,gabor_dev11,gabor_dev12,gabor_dev13
                     ,gabor_dev14,gabor_dev15,gabor0,gabor1,gabor2,gabor3,gabor4,gabor5,gabor6,gabor7,gabor8,
                     gabor9,gabor10,gabor11,gabor12,gabor13,gabor14,gabor15,hu_moments_max, hu_moments_mean,haralick_max, haralick_mean
                     ,contrast, dissimilarity, homogeneity, energy, correlation, ASM,eccentricity, solidity])

    total_labels=np.float64(np.hstack(new_texture).T)
    
    
    
#%% SAVING FEATURE AND LABEL LISTS IN FILES

pickle.dump(features, open('texture_features.sav', 'wb'))
pickle.dump(total_labels, open('texture_labels.sav', 'wb'))
