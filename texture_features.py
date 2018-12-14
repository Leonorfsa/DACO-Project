from skimage.feature import greycomatrix, greycoprops
#from sklearn import preprocessing
import numpy as np
from skimage.filters import gabor
import mahotas
import cv2
#from matplotlib import pyplot as plt


# Feature 1: Mean and Standard Deviation of Gabor Filters 
def gaborFilter(image):
# build Gabor filter bank kernels and apply to the image
    
    filtered_ims = []
    filtered_dev=[]
    filtered_mean=[]
   
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
    
    for i in range(len(filtered_ims)):
        filtered_mean.append(np.mean(filtered_ims[i]))
        filtered_dev.append(np.std(filtered_ims[i]))
    return filtered_mean, filtered_dev 


# Feature 2: Hu Moments
def fd_hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Feature 3: Haralick Texture
def fd_haralick(image):
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    # return the result
    return haralick

# Feature 4: GLCM 
def GLCM_features(image):
    
    glcm = greycomatrix(image, [1], [0],  symmetric = True, normed = True )
    
    # Calculate texture features
    contrast     = np.dtype(float).type(greycoprops(glcm, 'contrast')[0])
    dissimilarity= np.dtype(float).type(greycoprops(glcm, 'dissimilarity')[0])
    homogeneity  = np.dtype(float).type(greycoprops(glcm, 'homogeneity')[0])
    energy       = np.dtype(float).type(greycoprops(glcm, 'energy')[0])
    correlation  = np.dtype(float).type(greycoprops(glcm, 'correlation')[0])
    ASM          = np.dtype(float).type(greycoprops(glcm, 'ASM')[0])
    
    return contrast, dissimilarity, homogeneity, energy, correlation, ASM
