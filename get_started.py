import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
import pandas as pd
from skimage import exposure # Para usar .equalize_hist(img)
from skimage import filters


#%% _____________________________________
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

#%% FIND EXTENSION
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

#%% To open just the middle slice for each nodule
def getMiddleSlice(volume):
    size_vol = volume.shape
    return volume[...,np.int(size_vol[-1]/2)]

#%%
#_____________________________________
# LOAD DATA
#_____________________________________  


#find the current working directory
current_directory = os.getcwd()

#find the files
nodule_names, nodules = findExtension(os.path.join(current_directory,'images'))
#remove the extension from the nodule names
nodule_names = [os.path.splitext(x)[0] for x in nodule_names]

#Find the masks
mask_names, masks = findExtension(os.path.join(current_directory,'masks'))

#read the metadata
ground_truth = pd.read_excel('ground_truth.xls')

#to load an images you can simply do
#index = 20
#example = np.load(nodules[index])
# to get the nodule texture simply
#texture = int(ground_truth[ground_truth['Filename']==nodule_names[index]]['texture'])




nb = [0,10,50] #list of the image indexes to study/run
# Este nb vai ter de ser a lista inteira depois

#%%
#_____________________________________
# SHOW IMAGES
#_____________________________________


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
    #getMidleSlice- since we have volume we must show only a slice
    #createOverlay- Does the "Overlay" image, puts mask on top of nodule. 
    #If contour=False the overlay is complete
    fig,ax = plt.subplots(1,1)
    ax.imshow(over,**plot_args)
    plt.title('Overlay')
    plt.show()


#%%
#________________________________
# PRE-PROCESSING
#________________________________
    
    


#%%
#________________________________
# APPLY A MASK TO A NODULE 
#________________________________
    
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    masked = nodule*mask
    #since we have volume we must show only a slice
    fig,ax = plt.subplots(1,1)
    ax.imshow(getMiddleSlice(masked),**plot_args)
    plt.title('Mask')
    plt.show()

#%%
#________________________________
# OTHER ALGORITHMS TO HELP YOU GET STARTED
#________________________________
    
from skimage.filters.rank import entropy
from skimage.morphology import disk


#mean intensity of a nodule
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    intens = np.mean(nodule[mask!=0])
    print('The intensity of nodule',str(n),'is',intens)
    
#sample points from a nodule mask
np.random.seed(0)
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    sampled = np.zeros(mask.shape)
    
    loc = np.nonzero(mask)
    
    indexes = [x for x in range(loc[0].shape[0])]
    np.random.shuffle(indexes)
    
    #get 10% of the points
    indexes = indexes[:int(len(indexes)*0.1)]
    
    sampled[loc[0][indexes],loc[1][indexes],loc[2][indexes]]=True
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(getMiddleSlice(nodule),**plot_args)
    ax[1].imshow(getMiddleSlice(sampled),**plot_args)
    plt.title('Sampling from a mask')
    plt.show()    


#create a simple 2 feature vector for 2D segmentation
np.random.seed(0)
features = []
labels = []
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    nodule = getMiddleSlice(nodule)
    mask = getMiddleSlice(mask)

    
    #collect intensity and local entropy
    
    entrop = np.ravel(entropy(nodule,disk(5)))
    inten = np.ravel(nodule)
    
    
    labels.append([1 for x in range(int(np.sum(mask)))])
    
    
    features.append([entrop,inten])

    entrop = np.ravel(entropy(nodule==0,disk(5)))
    inten = np.ravel(nodule==0)
    features.append([entrop,inten])
    labels.append([0 for x in range(int(np.sum(mask==0)))])

    
X = np.hstack(features).T
labels = np.hstack(labels)
    
   
#create a simple 2 feature vector for 2D texture analysis
np.random.seed(0)
features = []
labels = []
for n in nb:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    
    nodule = getMiddleSlice(nodule)
    mask = getMiddleSlice(mask)
    
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])

    
    #collect intensity and local entropy
    
    entrop = np.mean(entropy(nodule,disk(5)))
    inten = np.mean(nodule)
    
    
    labels.append(texture)
    
    
    features.append([entrop,inten])

#features_tex = np.vstack(features)
#labels_tex = np.hstack(labels) 
    

#_____________________________________
# HISTOGRAM FILTERING
#_____________________________________
# Isto ta a dar um erro na imagem, n sei pq
    
#    hist,bins = np.histogram(nodule.flatten(),256,[0,1])
#    cdf = hist.cumsum()
#    cdf_normalized = cdf * hist.max()/ cdf.max()
#    plt.plot(cdf_normalized, color = 'b')
#    plt.hist(nodule.flatten(),256,[0,1], color = 'r')
#    plt.xlim([0,1])
#    plt.legend(('cdf','histogram'), loc = 'upper left')
#    plt.show()
#    
#    img_adapteq = exposure.equalize_adapthist(nodule)
#    fig,ax = plt.subplots(1,2)
#    ax[0].imshow(nodule,**plot_args)
#    ax[1].imshow(img_adapteq,**plot_args)
#    

#_____________________________________
# GAUSSIAN FILTERING
#_____________________________________

def gaussFiltering(image,sigma):
    gaussFilt = filters.gaussian(image, sigma)
    return gaussFilt

for sigma in [0.3,0.4,0.5,1.0, 1.5, 2.0]:
    gaussian=gaussFiltering(nodule, sigma)
    plt.imshow(gaussFilt,**plot_args)
    plt.show()
