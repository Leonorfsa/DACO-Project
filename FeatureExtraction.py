import os
import numpy as np
import Functions
import PreProcessing

from skimage.filters.rank import entropy
from skimage.morphology import disk

number_images=int(input('Enter the length of the Dataset (number of images):'))
amount_images=number_images

#%% LOADING OF ALL IMAGES AND MASKS
curr_path = os.getcwd() #find the current working directory

nodule_names, images = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

#Load all Images and Masks
sliced_images=[]
sliced_masks=[]
features = []
labels=[]

for n in range(0,(number_images)):
    single_image_3d = np.load(images[n])
    single_mask_3d = np.load(masks[n])
    
    #Extract middle slice of Image and its mask
    sliced_images.append(Functions.getMiddleSlice(single_image_3d)) #since we have a volume we must show only a slice
    sliced_masks.append(Functions.getMiddleSlice(single_mask_3d))
    
    #=======PRE-PROCESSING============
    
    # Gaussian
    sigma=0.5
    gauss_image=PreProcessing.gaussFiltering(sliced_images[n],sigma)
    
    #Histogram Equalization, etc..........
    
    
    
   #=======FEATURE EXTRACTION============
   
    #Collect intensity (Feature 1)
    intensity = np.ravel(gauss_image)
    
    #Collect local entropy (Feature 2)
    entrop = np.ravel(entropy(gauss_image,disk(5)))
    
    
    #Hessian Matrix
    sigmas = [0.5,0.60,0.70,0.8,0.9,1,1.2,2,4,5]
    shapeind, cv, eigValues=Functions.eigenValuesShapeIndexCurveness(sigmas, sliced_images[n])
    
    eigVal0=np.ravel(eigValues[0]) # Feature 3
    eigVal1=np.ravel(eigValues[1]) # Feature 4
    shapeind=np.ravel(shapeind) #Feature 5
    cv=np.ravel(cv) #Feature 6
    
    #Other Features...  
    
    
    #Convert labeled image into a one dimensional array
    label=np.ravel(sliced_masks[n])
    labels.append(label)
    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,entrop,eigVal0, eigVal1,shapeind,cv])
    total_labels=np.hstack(labels)
    total_features= np.hstack(features).T

#%% SAVING FEATURE NDARRAY AND LABEL ARRAY IN FILES

np.save('totalfeatures',total_features)
np.save('totallabels',total_labels)