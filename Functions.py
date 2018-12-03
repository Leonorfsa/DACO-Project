import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import find_boundaries
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split

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

# For feature extraction
features = []
labels = []

for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    flat_nodule=getMiddleSlice(nodule)
    flat_mask=getMiddleSlice(mask)
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    
    #_____________________________________
    # FEATURE EXTRACTION
    #_______________________________________
    #collect intensity and local entropy
    
    # primeiro guarda-se o objeto
    entrop = np.ravel(entropy(flat_nodule,disk(5)))
    inten = np.ravel(flat_nodule)
    labels.append([1 for x in range(int(np.sum(flat_mask)))])
    features.append([entrop,inten])
    
    
    # Depois poe-se o background
    entrop = np.ravel(entropy(flat_nodule==0,disk(5)))
    inten = np.ravel(flat_nodule==0)
    features.append([entrop,inten])
    labels.append([0 for x in range(int(np.sum(flat_mask==0)))])


X = np.hstack(features).T
labels = np.hstack(labels)

X = StandardScaler().fit_transform(X) # Para ter a certeza que tudo tem a mesma escala


#X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.3)


#_____________________________________
# SVM AND K-NEIGHBORS
#_______________________________________

#from sklearn import svm
#gamma = 1 # SVM RBF radius
## fazer SVM
#from sklearn.neighbors import KNeighborsClassifier
#
#knn=KNeighborsClassifier(n_neighbors=1)
#knn.fit(X,labels)
#print(knn.score(X, labels))