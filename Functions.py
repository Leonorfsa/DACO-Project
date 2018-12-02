import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries

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

def showImages(nb, nodule, mask):
    # plot_args defines plot properties in a single variable
    plot_args={}
    plot_args['vmin']=0
    plot_args['vmax']=1
    plot_args['cmap']='gray'

    #since we have a volume we must show only a slice
    fig,ax = plt.subplots(1,2)
    plt.title('Middle slice')
    ax[0].imshow(getMiddleSlice(nodule),**plot_args)
    ax[1].imshow(getMiddleSlice(mask),**plot_args)
    plt.show()

#_____________________________________
# LOAD DATA
#_____________________________________  

images_indexes=[0,10,15]

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = findExtension(os.path.join(curr_path,'masks'))   #Find the masks

ground_truth = pd.read_excel('ground_truth.xls') #read the metadata


for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    flat_nodule=getMiddleSlice(nodule)
    flat_mask=getMiddleSlice(mask)
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    
    
