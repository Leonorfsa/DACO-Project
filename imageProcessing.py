## IMAGE PROCESSING

from skimage.filters import gaussian
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage import filters
from scipy import ndimage as ndi
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation, opening, closing, erosion


def gaussFiltering(nodule, sigma=1, output=None, mode='nearest', 
                   cval=0, multichannel=None, preserve_range=False, truncate=4.0):
    gaussImage=gaussian(nodule, sigma, output,mode,cval,multichannel,preserve_range,truncate)
    return gaussImage


def watershedfilt(image):
    
    k=disk(1)
    image=dilation(image,k)
    distance = ndi.distance_transform_edt(image)
    
    num_peaks=5
    # To obtain the local peaks
    local_matrix= peak_local_max(distance, indices=False, num_peaks=num_peaks ,footprint=np.ones((7, 7)),
                                labels=image) 
    markers = ndi.label(local_matrix)[0]
    labels = watershed(-distance, markers, mask=image)
    
#    fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
#    ax = axes.ravel()
#    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#    ax[0].set_title('Overlapping objects')
#    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
#    ax[1].set_title('Distances')
#    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
#    ax[2].set_title('Separated objects')

    nodule_x = 0
    nodule_y = 0
   
    center=np.float64(np.floor(len(image)/2))
    dist1=10
    closest_pos=[]
    
    properties = regionprops(labels)
    for centre in properties:
        y0, x0 = centre.centroid
        x0 = np.uint8(x0)
        y0 = np.uint8(y0)
        x0_=np.float64(x0)
        y0_=np.float64(y0)
#        ax[2].plot(x0,y0,'r+')
        dist=math.sqrt(((x0_-center)**2)+((y0_-center)**2))
        if dist<dist1:
            dist1=dist
            closest_pos=[x0,y0]
        
    nodule_x = closest_pos[0]
    nodule_y = closest_pos[1]
    
    area = np.zeros([51,51])   
    color = labels[nodule_x,nodule_y]
 
    for k in range(len(labels)):
        for l in range(len(labels[k])):
            if labels[k,l] == color:
                area[k,l]=1
#    ax[3].imshow(area)
#    ax[3].set_title('Seg')
#    ax[3].plot(nodule_x,nodule_y,'r+')
    return area

def removeLung(image, no_nodule):
    
    val = filters.threshold_otsu(no_nodule)
    for i in range(len(no_nodule)):
        for j in range(len(no_nodule)):
            if (no_nodule[i,j]<val):
                no_nodule[i,j]=0
            elif (no_nodule[i,j]>val):
                no_nodule[i,j]=1
    #plt.imshow(no_nodule)
    kernel = disk(2)
    no_nodule = erosion(no_nodule, kernel)
    while (np.sum(no_nodule)>(np.sum(np.ones(no_nodule.size))//2)):
        kernel = disk(5)
        no_nodule = erosion(no_nodule, kernel)
    while(no_nodule[25,25]==1):
        kernel = disk(5)
        no_nodule = erosion(no_nodule, kernel)
    k2=disk(1)
    no_nodule = erosion(no_nodule, k2)
        # We want to avoid excess removal, so in these cases we must reduce the lung wall
    if (np.sum(no_nodule)<(np.sum(np.ones(no_nodule.size))//2) and no_nodule[25,25]==0):
        image[no_nodule==1]=0
    
#    th = 0.20
# 
#    hist, bin_edges = np.histogram(removed_lung, bins=60)
#    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#    binary_img = image >= th
#   
#    kernel = disk(1)
#    binary_img = opening(binary_img, kernel)
#    
#    kernel = disk(3)
#    binary_img = closing(binary_img, kernel)
#
#    #print(binary_img)
#    plt.figure(figsize=(5,2))
#
#    plt.subplot(131)
#    plt.imshow(image)
#    plt.axis('off')
#    plt.subplot(132)
#    plt.plot(bin_centers, hist, lw=2)
#    plt.axvline(th, color='r', ls='--', lw=2)
#    plt.text(0.57, 0.8, 'histogram', fontsize=20, transform = plt.gca().transAxes)
#    plt.yticks([])
#    #plt.imshow(gauss_upimage)    
#    #plt.axis('off')
#    plt.subplot(133)
#    plt.imshow(binary_img, cmap=plt.cm.gray, interpolation='nearest')
#    plt.axis('off')
#                
#
#    plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
#    plt.show()
    return image
    
    
    
    