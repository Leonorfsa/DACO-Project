import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import shape_index
from sklearn.model_selection import train_test_split
import Functions 
from guidedFilter import guidedFilt
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

    #utilizar 7 sigmas como referido no paper com 0.5 de step
    sigmas = [0.5,1.0,1.5,2.0,2.5,3.0,3.5]
    eigValues = []
    h_elem_aux = []
    h_elem_max = ()
    
    #já não é preciso fazer o filtro gaussiano porque esta função faz 
    #(Hrr, Hrc, Hcc) = hessian_matrix(flat_nodule, sigma=sigma, order='rc')
    #eigValues = hessian_matrix_eigvals((Hrr, Hrc, Hcc))
    
    for s in sigmas:
        #já não é preciso fazer o filtro gaussiano porque esta função faz 
        #para os vários sigmas usados vamos guardar apenas o tuplo com os maiores valores de hrr, hrc, hcc
        (Hrr, Hrc, Hcc) = hessian_matrix(flat_nodule, sigma = s, order='rc')
        h_elem_aux.append((Hrr, Hrc, Hcc))
        
    for i in range(len(h_elem_aux)-1):    
        h_elem_max = np.maximum(h_elem_aux[i],h_elem_aux[i+1])
        
    eigValues = hessian_matrix_eigvals(h_elem_max) 
    #fazer o plot para o primeiro valor e segundo valor de eig values de cada pixel 
    Functions.show2DImages(eigValues[0],flat_nodule)
    Functions.show2DImages(eigValues[1],flat_nodule)
        
    shapeind0_aux = []
    shapeind1_aux = []
    #calcular shape index
    for s in sigmas:
        shapeind0_aux.append(shape_index(eigValues[0],s))
        shapeind1_aux.append(shape_index(eigValues[1],s))
    
    #shape index FINAIS
    #extrair o maiores shape index resultantes da aplicação dos diferentes sigmas
    for i in range(len(shapeind0_aux)-1):
        shapeind0 = np.maximum(shapeind0_aux[i],shapeind0_aux[i+1])    
        shapeind1 = np.maximum(shapeind1_aux[i],shapeind1_aux[i+1])
   
    #calcular curvedness
    cv = []
    for i in range(eigValues.shape[1]):
        for j in range(eigValues.shape[2]):
            cv.append(math.sqrt((math.pow(eigValues[0][i][j],2)+(math.pow(eigValues[1][i][j],2)))))
   
        
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






I=flat_nodule
p=I
r=16
eps=0.01

q=np.zeros(I.shape)

q=guidedFilt(I,p,r,eps)
#q=guidedFilt(I(...,...,2),p(...,...,2),r,eps)
#q=guidedFilt(I(...,...,3),p(...,...,3),r,eps)

I_enhanced=(I-q)*5+q

fig,ax = plt.subplots(1,3)
ax[0].imshow(I,[0,1])
ax[1].imshow(q,[0,1])
ax[2].imshow(I_enhanced,[0,1])