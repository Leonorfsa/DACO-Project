#%% IMPORTS

import numpy as np
from skimage.filters import gabor
from matplotlib import pyplot as plt

# GABOR FILTERING 

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
        #filtered_total = np.hstack(filtered_fil)
       # filtered_dev_total.append(np.ravel(filtered_dev[i]))
    return filtered_mean, filtered_dev 