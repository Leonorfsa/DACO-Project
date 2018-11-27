import os
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
# LOAD DATA
#_____________________________________  

def loadData(images_indexes, filename):
    #find the current working directory
    curr_path = os.getcwd()
    
    #find the files
    nodule_names, nodules = findExtension(os.path.join(curr_path,'images'))
    #remove the extension from the nodule names
    nodule_names = [os.path.splitext(x)[0] for x in nodule_names]
    
    #Find the masks
    mask_names, masks = findExtension(os.path.join(curr_path,'masks'))
    
    #read the metadata
    metadata = pd.read_excel('ground_truth.xls')
    
   #         #to load an images you can simply do
   #         index = 20
   #         example = np.load(nodules[index])
   #to get the nodule texture simply
    texture = int(metadata[metadata['filename']==nodule_names[images_indexes]]['texture'])
            
    
    nb = [images_indexes] #list of the image indexes to study/run
    # Este nb vai ter de ser a lista inteira depois
    return nodule_names, mask_names, nodules, masks, texture, nb



#_____________________________________
# GET MIDDLE SLICE
#
#   Pega na fatia do meio de cada cubo de 
#_____________________________________
    
def getMiddleSlice(volume):
    sh = volume.shape
    
    return volume[...,np.int(sh[-1]/2)]


#_____________________________________
# CREATE OVERLAY
#   
#  Cria uma imagem que contém uma sobreposição da máscara no nódulo.
#  contour=False faz com que a máscara apareça por inteiro, enquanto 
#  que contour=True faz com que só apareçam os limites 
#_____________________________________
    

def createOverlay(im,mask,color=(0,1,0),contour=True):
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
    elif len(im.shape)==3:
        if im.shape[-1] != 3:
            ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)

    else:
        ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)
   
    if contour:
        bw = find_boundaries(mask,mode='thick') #inner
    else:
        bw = mask
    for i in range(0,3):
        im_temp = im[:,:,i]
        im_temp = np.multiply(im_temp,np.logical_not(bw)*1)
        im_temp += bw*color[i]
        im[:,:,i] = im_temp
    return im


#_____________________________________
# SHOW IMAGES
#_____________________________________

def showImages(nb, nodules, masks):
    # plot_args defines plot properties in a single variable
    plot_args={}
    plot_args['vmin']=0
    plot_args['vmax']=1
    plot_args['cmap']='gray'
    
    
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        #since we have a volume we must show only a slice
        fig,ax = plt.subplots(1,2)
        plt.title('Middle slice')
        ax[0].imshow(getMiddleSlice(nodule),**plot_args)
        ax[1].imshow(getMiddleSlice(mask),**plot_args)
        plt.show()
    
    #if instead you want to overlay (Imagem com os limites a verde)
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        over = createOverlay(getMiddleSlice(nodule),getMiddleSlice(mask))
        #since we have volume we must show only a slice
        fig,ax = plt.subplots(1,1)
        ax.imshow(over,**plot_args)
        plt.title('Overlay')
        plt.show()

