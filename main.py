import os
import pandas as pd
import numpy as npS
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.model_selection import train_test_split
import Functions 
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
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
eigValues=[]
for n in images_indexes:
    nodule = np.load(nodules[n])
    mask = np.load(masks[n])
    flat_nodule=Functions.getMiddleSlice(nodule) #since we have a volume we must show only a slice
    flat_mask=Functions.getMiddleSlice(mask)
    #Functions.show2DImages(flat_nodule, flat_mask)
    
    #já não é preciso fazer o filtro gaussiano porque esta função faz 
    sigma=0.5
    (Hrr, Hrc, Hcc) = hessian_matrix(flat_nodule, sigma=sigma, order='rc')
    eigValues = hessian_matrix_eigvals((Hrr, Hrc, Hcc))
    #print(eigValues[0])
    #.print(eigValues[1])
    
    # Gaussian
    
   #gaussImage=PreProcessing.gaussFiltering(flat_nodule,sigma)
   # print(type(gaussImage))
   # h = Functions.hessian(gaussImage)
   # print(h)
    #gaussImage2=PreProcessing.gaussFiltering(images_indexes,nodules,0.2)
    #Functions.show2DImages(nodule, flat_nodule)
    
    #To make sure we have the same number of pixels for nodule and background
    nodule,flat_mask=Functions.sample(flat_nodule,flat_mask)
    Functions.show2DImages(flat_nodule, flat_mask)
    
    #totalGauss.append(gaussImage)
    
    texture = int(ground_truth[ground_truth['Filename']==nodule_names[n]]['texture'])
    
    #_____________________________________
    # FEATURE EXTRACTION
    #_______________________________________
    #collect intensity and local entropy
    
    intensity = np.ravel(flat_nodule)
    entrop = np.ravel(entropy(flat_nodule,disk(5)))
    
    label=np.ravel(flat_mask) # Para pôr em linha
    labels.append(label)
    features.append([intensity,entrop]) # 
    total_labels=np.hstack(labels)
    total_features = np.hstack(features).T
    

total_features = StandardScaler().fit_transform(total_features) # Para ter a certeza que tudo tem a mesma escala

X_train, X_val, y_train, y_val = train_test_split(total_features, total_labels, test_size=0.3)


# CLASSIFICADORES
# _______________
# K-Neighbors
n_neighbors=5
knn=Functions.KNeighbors(n_neighbors, X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_val, y_val))

