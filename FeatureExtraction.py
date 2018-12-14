import os
import numpy as np
import Functions
import PreProcessing as prepros
import hessian_matrix as hessian
from skimage.filters.rank import entropy
from skimage.morphology import disk

number_images=134
              #int(input('Enter the length of the Dataset (number of images):'))

#%% LOADING OF ALL IMAGES AND MASKS
curr_path = os.getcwd() #find the current working directory

nodule_names, images = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

#Load all Images and Masks
images_3d=[]
masks_3d=[]
sliced_images=[]
sliced_masks=[]
features = []
labels=[]

for n in range(0,(number_images)):
    single_image_3d = np.load(images[n])
    single_mask_3d = np.load(masks[n])
    images_3d.append(single_image_3d)
    masks_3d.append(single_mask_3d)
    
    #Extract middle slice of Image and its mask
    sliced_images.append(Functions.getMiddleSlice(single_image_3d)) #since we have a volume we must show only a slice
    sliced_masks.append(Functions.getMiddleSlice(single_mask_3d))
    
    #=======PRE-PROCESSING============
    
    # Gaussian
    sigma=0.5
    gauss_image=prepros.gaussFiltering(sliced_images[n],sigma)
    
    #Histogram Equalization, etc..........
    
   #=======FEATURE EXTRACTION============
   
    #Collect intensity (Feature 1)
    intensity = np.ravel(gauss_image)
    
    #Collect local entropy (Feature 2)
    entrop = np.ravel(entropy(gauss_image,disk(5)))
    
    #Hessian Matrix
    sigmas = [0.5,0.60,0.70,0.8,0.9,1,1.2,2,4,5]
    shapeind, cv, eigValues=hessian.eigenValuesShapeIndexCurveness(sigmas, sliced_images[n])
    
    eigVal0=np.ravel(eigValues[0]) # Feature 3
    eigVal1=np.ravel(eigValues[1]) # Feature 4
    shapeind=np.ravel(shapeind) #Feature 5
    cv=np.ravel(cv) #Feature 6
    
    gabor_features=Functions.gaborFilter(sliced_images[n])
    gabor0=np.ravel(gabor_features[0])
    gabor1=np.ravel(gabor_features[1])
    gabor2=np.ravel(gabor_features[2])
    gabor3=np.ravel(gabor_features[3])
    gabor4=np.ravel(gabor_features[4])
    gabor5=np.ravel(gabor_features[5])
    gabor6=np.ravel(gabor_features[6])
    gabor7=np.ravel(gabor_features[7])
    gabor8=np.ravel(gabor_features[8])
    gabor9=np.ravel(gabor_features[9])
    gabor10=np.ravel(gabor_features[10])
    gabor11=np.ravel(gabor_features[11])
    gabor12=np.ravel(gabor_features[12])
    gabor13=np.ravel(gabor_features[13])
    gabor14=np.ravel(gabor_features[14])
    gabor15=np.ravel(gabor_features[15])

#    gabor_mult1=gabor0*gabor1
#    gabor_mult2=gabor2*gabor3
#    gabor_mult3=gabor4*gabor5
#    gabor_mult4=gabor6*gabor7
    
#    gabor_div1=gabor8/gabor9
#    gabor_div2=gabor10/gabor11
#    gabor_div3=gabor12/gabor13
#    gabor_div4=gabor14/gabor15
    
    #Other Features...  
    
    #Convert labeled image into a one-dimensional array
    label=np.ravel(sliced_masks[n])
    labels.append(label)
    
    #Concatenate all features for all Training Images in an ndarry
    features.append([intensity,entrop,eigVal0, eigVal1,shapeind,cv, gabor0, 
                     gabor1,gabor2, gabor3, gabor4, gabor5, gabor6, gabor7, gabor8, 
                     gabor9, gabor10, gabor11, gabor12, gabor13, gabor14, gabor15])#,
                     gabor_mult1,gabor_mult2,gabor_mult3])#, gabor_div1, gabor_div2, 
                     #gabor_div3, gabor_div4])
    total_labels=np.hstack(labels)
    total_features= np.hstack(features).T

#%% SAVING FEATURE NDARRAY AND LABEL ARRAY IN FILES

np.save('totalfeatures',total_features)
np.save('totallabels',total_labels)
np.save()