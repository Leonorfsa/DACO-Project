import os
import pandas as pd
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations_with_replacement
<<<<<<< HEAD
=======
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
>>>>>>> d6f8378427db43f01203a4f472ff5d8a49768ba7
from warnings import warn

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


<<<<<<< HEAD
def hessian_matrix_eigvals(H_elems, Hxy=None, Hyy=None, Hxx=None):
    """Compute Eigenvalues of Hessian matrix.
    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.
    Hxy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hyy : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Hxx : ndarray, deprecated
        Element of the Hessian matrix for each pixel in the input image.
    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).
    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc')
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    if Hxy is not None:
        if Hxx is None:
            Hxx = H_elems
        H_elems = [Hxx, Hxy, Hyy]
        warn('The API of `hessian_matrix_eigvals` has changed. Use a list of '
             'elements instead of separate arguments. The old version of the '
             'API will be removed in version 0.16.')
    if len(H_elems) == 3:  # Use fast Cython code for 2D
        eigvals = np.array(_image_orthogonal_matrix22_eigvals(*H_elems))
    else:
        matrices = _hessian_matrix_image(H_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigvals = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigvals.ndim - 1))
        eigvals = np.transpose(eigvals, (eigvals.ndim - 1,) + leading_axes)
    return eigvals

def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2


def _hessian_matrix_image(H_elems):
    """Convert the upper-diagonal elements of the Hessian matrix to a matrix.
    Parameters
    ----------
    H_elems : list of array
        The upper-diagonal elements of the Hessian matrix, as returned by
        `hessian_matrix`.
    Returns
    -------
    hessian_image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the Hessian matrix corresponding to each coordinate.
    """
=======
def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2


def _hessian_matrix_image(H_elems):
    """Convert the upper-diagonal elements of the Hessian matrix to a matrix.
    Parameters
    ----------
    H_elems : list of array
        The upper-diagonal elements of the Hessian matrix, as returned by
        `hessian_matrix`.
    Returns
    -------
    hessian_image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the Hessian matrix corresponding to each coordinate.
    """
>>>>>>> d6f8378427db43f01203a4f472ff5d8a49768ba7
    image = H_elems[0]
    hessian_image = np.zeros(image.shape + (image.ndim, image.ndim))
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        hessian_image[..., row, col] = H_elems[idx]
        hessian_image[..., col, row] = H_elems[idx]
    return hessian_image

<<<<<<< HEAD
=======

def eigValues(sigmas, flat_nodule):
    #já não é preciso fazer o filtro gaussiano porque esta função faz 
    #(Hrr, Hrc, Hcc) = hessian_matrix(flat_nodule, sigma=sigma, order='rc')
    #eigValues = hessian_matrix_eigvals((Hrr, Hrc, Hcc))
    eigValues = []
    h_elem_aux = []
    h_elem_max = ()
    
    for s in sigmas:
        #já não é preciso fazer o filtro gaussiano porque esta função faz 
        #para os vários sigmas usados vamos guardar apenas o tuplo com os maiores valores de hrr, hrc, hcc
        (Hrr, Hrc, Hcc) = hessian_matrix(flat_nodule, sigma = s, order='rc')
        h_elem_aux.append((Hrr, Hrc, Hcc))
        
    for i in range(len(h_elem_aux)-1):    
        h_elem_max = np.maximum(h_elem_aux[i],h_elem_aux[i+1])
        
    eigValues = hessian_matrix_eigvals(h_elem_max) 
    return eigValues

def shapeindex(eigValues):
    shapeindex = (2/math.pi)*np.arctan((eigValues[1]+eigValues[0])/(eigValues[1])-eigValues[0])
    return shapeindex

def curvedness(eigValues):
    cv = []
    for i in range(eigValues.shape[1]):
        for j in range(eigValues.shape[2]):
            cv.append(math.sqrt((math.pow(eigValues[0][i][j],2)+(math.pow(eigValues[1][i][j],2)))))
    return cv
>>>>>>> d6f8378427db43f01203a4f472ff5d8a49768ba7

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
    sampledmask = np.zeros(mask.shape)
    sampled_background=np.zeros((mask==0).shape)
    loc = np.nonzero(mask)
    loc_zero=np.nonzero(mask==0)
    indexes = [x for x in range(loc[0].shape[0])]
    indexes_zeros=[x for x in range(loc_zero[0].shape[0])]
    np.random.shuffle(indexes)
    np.random.shuffle(indexes_zeros)
    #get 10% of the points
    indexes10perc = indexes[:int(len(indexes)*0.3)]
    indexes10perc_background=indexes_zeros[:int(len(indexes)*0.3)]
    
    sampledmask[loc[0][indexes10perc],loc[1][indexes10perc]]=True
    sampled_background[loc_zero[0][indexes10perc_background],loc_zero[1][indexes10perc_background]]=True
    sampledtotal=sampledmask+sampled_background
    samplednodule=nodule*sampledtotal
    return samplednodule,sampledmask


def sampling2(total_features,total_labels, number_pixels_each_label):
    #This functions takes the matrix containing all features (in which each line corresponds to a pixel in the original
    #image) and the array containing all labels and samples full lines, assuring the number of nodule and non-nodule
    #pixels is the same.The quantity of pixels sampled for each label is defined as an inpt
    #Returns: a matrix of the sampled features and an array of the according labels
    
    #These variables are created as lists to make append easier (Later will be converted to arrays)
    sampled_features=[]
    sampled_labels=[]
    random.seed(0)
    
    #Select random pixels with label 1 (Quantity is as desired by user)
    loc_label1 = np.where(total_labels == 1)[0] #Store the indexes of the pixels which label is equal to 1
    random.shuffle(loc_label1)
    sampled_label1_indexes=loc_label1[:number_pixels_each_label] #Select only the quantity refered as an input rgument
    
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
    return sampled_labels,sampled_features       

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


#_____________________________________
# PREORMANCE EVALUATING
#_______________________________________

<<<<<<< HEAD
=======
def confusionMatrixCalculator(prediction,GT):
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(len(prediction)):
        if (prediction[i]==GT[i]):
            if (prediction[i]==1):
                TP+=1
            else:
                TN+=1
        else:
            if (prediction[i]==1):
                FP+=1
            else:
                FN+=1
        return TP, TN, FP, FN
>>>>>>> d6f8378427db43f01203a4f472ff5d8a49768ba7
    
    