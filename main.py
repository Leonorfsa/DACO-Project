import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split
import Functions 
import PreProcessing
from matplotlib import pyplot as plt
plt.close('all')
np.random.seed(0) # To avoid changes in random choice
#_____________________________________
# LOAD DATA
#_____________________________________  

images_indexes=[0,10,15] # Imagens que vamos abrir

curr_path = os.getcwd() #find the current working directory

nodule_names, nodules = Functions.findExtension(os.path.join(curr_path,'images')) #find the files
nodule_names = [os.path.splitext(x)[0] for x in nodule_names] #remove the extension from the nodule names
mask_names, masks = Functions.findExtension(os.path.join(curr_path,'masks'))   #Find the masks

ground_truth = pd.read_excel('ground_truth.xls') #read the metadata

# For feature extraction
features = []
labels = []
totalGauss=[]
for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    flat_nodule=Functions.getMiddleSlice(nodule) #since we have a volume we must show only a slice
    flat_mask=Functions.getMiddleSlice(mask)
    Functions.show2DImages(flat_nodule, flat_mask)
    
    # Gaussian
    sigma=0.5
    gaussImage=PreProcessing.gaussFiltering(flat_nodule,sigma)
    print(type(gaussImage))
    h = Functions.hessian(gaussImage)
    print(h)
    #gaussImage2=PreProcessing.gaussFiltering(images_indexes,nodules,0.2)
    Functions.show2DImages(gaussImage, flat_nodule)
    totalGauss.append(gaussImage)
    
    
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    #_____________________________________
    # FEATURE EXTRACTION
    #_______________________________________
    #collect intensity and local entropy
    
    intensity = np.ravel(gaussImage)
    entrop = np.ravel(entropy(gaussImage,disk(5)))
    
    label=np.ravel(flat_mask) # Para pôr em linha
    labels.append(label)
    features.append([intensity,entrop]) # 

    total_labels=np.hstack(labels)
    total_features = np.hstack(features).T
    

total_features = StandardScaler().fit_transform(total_features) # Para ter a certeza que tudo tem a mesma escala

# Tirar o mesmo número de amostras de background e de nodulo

#sampledimage,sampledbackground=Functions.sample(gaussImage,flat_mask)
#Functions.show2DImages(sampledimage,sampledbackground)
X_train, X_val, y_train, y_val = train_test_split(total_features, total_labels, test_size=0.3)


# CLASSIFICADORES
# _______________
# K-Neighbors
n_neighbors=1
knn=Functions.KNeighbors(n_neighbors, X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_val, y_val))

