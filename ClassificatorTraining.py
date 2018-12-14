#%% IMPORTS AND LOADS

import numpy as np
import random
import Functions
import Classifiers
import pickle
import time
from FeatureExtraction import number_images
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

total_features=np.load('totalfeatures.npy')
total_labels=np.load('totallabels.npy')

np.random.seed(0)

#%%Data Standerization, Dataset Split, Sampling and Training


#Data Standerization (To ensure mean=0 and std=1)
scaler = StandardScaler().fit(total_features)
total_features=scaler.transform(total_features)

image_indexes=np.array(range(0,number_images))

#%% TRAIN TEST SPLIT
#X_train, X_test, Y_train, Y_test = train_test_split(total_features, total_labels, 
#                                                    test_size=0.20,random_state=0, stratify=total_labels)


#TRAIN_TEST_SPLIT NÃO ESTAVA A FUNCIONAR
train_size=70
            #int(input('What is the percentage of the Dataset used for Training (0-100): '))/100
start = time.time()
np.random.shuffle(image_indexes)
X_train_indexes=image_indexes[:int(len(image_indexes)*train_size)]
X_val_indexes=image_indexes[:int(len(image_indexes)*train_size)]

X_train=np.zeros((1,int(len(total_features[0]))))
Y_train=[]
Train_pixel_index=[]
for i in X_train_indexes:
    first_line=i*2601
    last_line=((i+1)*2601)
    Train_pixel_index.extend(range(first_line, last_line))
    Train_pixel_index.sort()
    
for j in Train_pixel_index:
       X_train=np.vstack((X_train,total_features[j]))
       Y_train.append(total_labels[j])

Y_train=np.asarray(Y_train)#X_train_index, X_test_index=train_test_split(image_indexes, 0.3, random_state=42)
X_train = np.delete(X_train, (0), axis=0)

X_val=np.zeros((1,6))
Y_val=[]
Val_pixel_index=[]
for i in X_val_indexes:
    first_line=i*2601
    last_line=((i+1)*2601)
    Val_pixel_index.extend(range(first_line, last_line))
    Val_pixel_index.sort()
    
for j in Val_pixel_index:
       X_val=np.vstack((X_val,total_features[j]))
       Y_val.append(total_labels[j])
       
Y_val=np.asarray(Y_val)#X_train_index, X_test_index=train_test_split(image_indexes, 0.3, random_state=42)
X_val=np.delete(X_val, (0), axis=0)

#%%

#Sampling (To ensure we train with the same number of nodule and non-nodule pixels)
sampled_features,sampled_labels=Functions.sampling2(X_train,Y_train,1000) 

#=======CLASSIFIER TRAINING============

# Naive Bayes
maxAccuracy_gnb=Classifiers.naiveBayes(sampled_features,sampled_labels)

# Logistic Regression
regularization_params = [0.0001, 0.001, 0.01, 1, 10]
clf_LR,maxAccuracy_lr=Classifiers.logReg(sampled_features, sampled_labels, regularization_params)

#K-Neighbors
n_neighbors = [1,3,5,7,9,11,13,15]
knn,maxAccuracy_knn=Classifiers.KNeighbors(n_neighbors, sampled_features, sampled_labels) #Training K-neighbours
print(knn.score(sampled_features,sampled_labels)) #Porque é que isto não dá 1??

#SVM Grid search or Random Search
parameters = [{'kernel': ['rbf'],
               'gamma': [0.001, 0.01, 0.1, 1, 10],
                'C': [0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]
#clf_SVM, maxAccuracy_SVM=Classifiers.SVMs_grid(sampled_features, sampled_labels, parameters)
clf_SVM_rand, maxAccuracy_rand=Classifiers.SVMs_rand(sampled_features, sampled_labels, parameters)

# Decision Trees 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
grid_search, maxAccuracy_trees=Classifiers.decision_trees(sampled_features, sampled_labels, param_grid)

#%% FINAL RESULTS

clf_LR.score(X_test, Y_test)
knn.score(X_test, Y_test)
#clf_SVM.score(X_test, Y_test)
clf_SVM_rand.score(X_test, Y_test)
grid_search.score(X_test, Y_test)

print("Gaussian Naive Bayes accuracy: ",maxAccuracy_gnb)
print("Logistic Regression: ", clf_LR.score(X_test, Y_test))
print("kNN", knn.score(X_test, Y_test))
#print("SVM",clf_SVM.score(X_test, y_test))
print("SVM",clf_SVM_rand.score(X_test, Y_test))
print("trees",grid_search.score(X_test, Y_test))



#%%SAVE TRAINED CLASSIFIERS INTO FILES

filename1 = 'trainedknn.sav'
filename2='Xval.sav'
filename3='Yval.sav'
filename4='SVM.sav'
pickle.dump(knn, open(filename1, 'wb'))
pickle.dump(X_val, open(filename2, 'wb'))
pickle.dump(Y_val, open(filename3, 'wb'))
pickle.dump(svm_classifier, open(filename4, 'wb'))
end = time.time()
elapsed_time=end-start
#print("Execution time: ",end-start)
pickle.dump(elapsed_time, open('time.sav', 'wb'))