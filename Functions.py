import os
import numpy as np
import random
from matplotlib import pyplot as plt
from skimage.filters import gabor

random.seed(0)

#%% List of functions: 
#   - find extension 
#   - get middle slice
#   - shapeindex
#   - curvedness
#   - show2DImages
#   - sample / sampling2: returns sampled_labels, sampled_features  
#   - KNeighbors
#   - confusionMatrixCalculator: returns TP, TN, FP, FN

#_____________________________________
# FIND EXTENSION DIRECTORY
#_____________________________________  

def findExtension(directory,extension='.npy'):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path

#_____________________________________
# GET MIDDLE SLICE
#   Pega na fatia do meio de cada cubo 51x51x51 
#_____________________________________
    
def getMiddleSlice(volume):
    sh = volume.shape
    return volume[...,np.int(sh[-1]/2)]    


#_____________________________________
# SHOW IMAGES
#_____________________________________

def show2DImages(nodule, mask, addapt=0):
    # plot_args defines plot properties in a single variable
    plot_args={}
    plot_args['cmap']='gray'
    fig,ax = plt.subplots(1,2)
    plt.title('Middle slice')
    if addapt==1:
        plot_args['vmin']=np.min(nodule)
        plot_args['vmax']=np.max(nodule)
        ax[0].imshow(nodule,**plot_args)

    else:
        plot_args['vmin']=0
        plot_args['vmax']=1
        ax[0].imshow(nodule,**plot_args)
        
    plot_args['vmin']=0
    plot_args['vmax']=1
    ax[1].imshow(mask,**plot_args)
    plt.show()
    
    return


#_____________________________________
# Sample nodule and background
#_____________________________________

##sample points from a nodule mask
#np.random.seed(0)
#def sample(nodule,mask):
#    sampledmask = np.zeros(mask.shape)
#    sampled_background=np.zeros((mask==0).shape)
#    loc = np.nonzero(mask)
#    loc_zero=np.nonzero(mask==0)
#    indexes = [x for x in range(loc[0].shape[0])]
#    indexes_zeros=[x for x in range(loc_zero[0].shape[0])]
#    np.random.shuffle(indexes)
#    np.random.shuffle(indexes_zeros)
#    #get 10% of the points
#    indexes10perc = indexes[:int(len(indexes)*0.3)]
#    indexes10perc_background=indexes_zeros[:int(len(indexes)*0.3)]
#    
#    sampledmask[loc[0][indexes10perc],loc[1][indexes10perc]]=True
#    sampled_background[loc_zero[0][indexes10perc_background],loc_zero[1][indexes10perc_background]]=True
#    sampledtotal=sampledmask+sampled_background
#    samplednodule=nodule*sampledtotal
#    return samplednodule,sampledmask

def sampling2(total_features,total_labels, number_pixels_each_label):
    # This functions takes in the matrix containing all features (in which each 
    # line corresponds to a pixel in the original image) and the array 
    # containing all labels, and samples leaving full lines, assuring the number of 
    # nodule and non-nodule pixels is the same. The quantity of pixels sampled 
    # for each label is defined as an input
    # Returns: a matrix of the sampled features and an array of the according labels
    
    #These variables are created as lists to make append easier (Later will be converted to arrays)
    sampled_features=[]
    sampled_labels=[]
    
    #Select random pixels with label 1 (Quantity is as desired by user)
    loc_label1 = np.where(total_labels == 1)[0] #Store the indexes of the pixels which label is equal to 1
    random.shuffle(loc_label1)
    sampled_label1_indexes=loc_label1[:number_pixels_each_label] #Select only the quantity refered as an input argument
    
    #Select random pixels with label 0 (Quantity must be equal to the number of sampled pixels with label 1)
    loc_label0=np.where(total_labels == 0)[0] #Store the indexes of the pixels which label is equal to 0
    random.shuffle(loc_label0)
    sampled_label0_indexes=loc_label0[:number_pixels_each_label] #Select only the quantity refered as an input rgument
    
    for i in range(len(total_labels)):
        if (i in sampled_label1_indexes or i in sampled_label0_indexes):
           sampled_labels.append(total_labels[i])
           sampled_features.append(total_features[i][:])
    
    #Conversion back to arrays
    sampled_labels=np.asarray(sampled_labels)
    sampled_features = np.vstack(sampled_features)
    return sampled_features, sampled_labels

#______________________________________
# GABOR FILTERS
#_______________________________________

def gaborFilter(image):
# build Gabor filter bank kernels and apply to the image
    filtered_ims = []
   
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (2, 4):
            for frequency in (0.05, 0.1):
                filt_real, filt_imag = gabor(image, frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma)
                filtered_ims.append(filt_real)
                
#    print(len(filtered_ims))
#    f, axarr = plt.subplots(4,4)
#    for i in range(4):
#        for j in range(4):
#            axarr[i,j].imshow(filtered_ims[4*i + j], cmap='gray')
##    for i in range(4):
##        for j in range(4):
##            axarr[i,j+4].imshow(filtered_ims[4*i + j+4], cmap='gray')
#    plt.show()  
    
    return filtered_ims

