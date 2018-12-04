import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

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


def hessian(ndarray):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    grad = np.gradient(ndarray) 
    hessian = np.empty((ndarray.ndim, ndarray.ndim) + ndarray.shape, dtype=ndarray.dtype) 
    for k, grad_k in enumerate(grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


#_____________________________________
# SHOW IMAGES
#_____________________________________

def show2DImages(nodule, mask):
    # plot_args defines plot properties in a single variable
    plot_args={}
    plot_args['vmin']=0
    plot_args['vmax']=1
    plot_args['cmap']='gray'
    fig,ax = plt.subplots(1,2)
    plt.title('Middle slice')
    ax[0].imshow(nodule,**plot_args)
    ax[1].imshow(mask,**plot_args)
    plt.show()


#_____________________________________
# Sample nodule and background
#_____________________________________

#sample points from a nodule mask
np.random.seed(0)
def sample(nodule,mask):
    sampled = np.zeros(mask.shape)
    sampled_background=np.zeros(mask.shape)
    loc = np.nonzero(mask)
    loc_zero=np.nonzero(mask==0)
    indexes = [x for x in range(loc[0].shape[0])]
    index_zeros=[x for x in range(loc_zero[0].shape[0])]
    np.random.shuffle(indexes)
    np.random.shuffle(index_zeros)
    #get 10% of the points
    #indexes10perc = indexes[:int(len(indexes)*0.1)]
    
    sampled[loc[0][indexes],loc[1][indexes],loc[2][indexes]]=True
    sampled_background[loc[0][index_zeros],loc[1][index_zeros],loc[2][index_zeros]]=True
    return sampled, sampled_background

#_____________________________________
# K-NEIGHBORS
#_______________________________________

def KNeighbors(n_neighbors, X, y):
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(X,y)
    return knn

#_____________________________________
# SVM
#_______________________________________

gamma = 1 # SVM RBF radius
# fazer SVM