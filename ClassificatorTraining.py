#%% IMPORTS AND LOADS

import numpy as np
import random
import Functions
import Classifiers
import pickle
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

start_time = time.time()
total_features=pickle.load(open('totalfeatures.sav', 'rb'))
total_labels=pickle.load(open('totallabels.sav', 'rb'))

np.random.seed(0)

#%% DATASET SPLIT, DATA STANDARIZATION SAMPLING

#TrainTestSplit
X_train,X_test,Y_train,Y_test=train_test_split(total_features, total_labels,test_size=0.20,random_state=0)

#Conversion to ndarray
X_train_ndarray=np.vstack(X_train[:])
Y_train_ndarray=np.hstack(Y_train[:])

#Data Standerization (To ensure mean=0 and std=1)
scaler = StandardScaler().fit(X_train_ndarray)
X_train_ndarray=scaler.transform(X_train_ndarray)

#Sampling (To ensure we train with the same number of nodule and non-nodule pixels)
sampled_features,sampled_labels=Functions.sampling2(X_train_ndarray,Y_train_ndarray,10000) 


#%%
#=======CLASSIFIER TRAINING============

## Naive Bayes
#gnb, maxAccuracy_gnb=Classifiers.naiveBayes(sampled_features,sampled_labels)
#
## Logistic Regression
#regularization_params = [0.0001, 0.001, 0.01, 1, 10]
#clf_LR,maxAccuracy_lr=Classifiers.logReg(sampled_features, sampled_labels, regularization_params)

#K-Neighbors
n_neighbors = [1,3,5,7,9,11,13,15]
knn,maxAccuracy_knn=Classifiers.KNeighbors(n_neighbors, sampled_features, sampled_labels) #Training K-neighbours
print(knn.score(sampled_features,sampled_labels))

#SVM Grid search or Random Search
#parameters = [{'kernel': ['rbf'],
#               'gamma': [0.001, 0.01, 0.1, 1, 10],
#                'C': [0.1, 1, 10, 100]},
#              {'kernel': ['linear'], 'C': [1, 10, 100]}]
##clf_SVM, maxAccuracy_SVM=Classifiers.SVMs_grid(sampled_features, sampled_labels, parameters)
#svm_classifier, maxAccuracy_rand=Classifiers.SVMs_rand(sampled_features, sampled_labels, parameters)
##
## Decision Trees 
#param_grid = {"criterion": ["gini", "entropy"],
#              "min_samples_split": [2, 10, 20],
#              "max_depth": [None, 2, 5, 10],
#              "min_samples_leaf": [1, 5, 10],
#              "max_leaf_nodes": [None, 5, 10, 20],
#              }
#grid_search, maxAccuracy_trees=Classifiers.decision_trees(sampled_features, sampled_labels, param_grid)

#%% FINAL RESULTS

#clf_LR.score(X_test, Y_test)
#clf_SVM.score(X_test, Y_test)
#clf_SVM_rand.score(X_test, Y_test)
#grid_search.score(X_test, Y_test)

#print("Gaussian Naive Bayes accuracy: ",maxAccuracy_gnb)
#print("Logistic Regression: ", clf_LR.score(X_test, Y_test))
#print("kNN", knn.score(X_val, Y_val))
#print("SVM",clf_SVM.score(X_test, y_test))
#print("SVM",clf_SVM_rand.score(X_test, Y_test))
#print("trees",grid_search.score(X_test, Y_test))



#%%SAVE TRAINED CLASSIFIERS INTO FILES

#elapsed_time = time.time() - start_time
#pickle.dump(elapsed_time, open('elapsedtime.sav', 'wb'))
pickle.dump(X_test, open('Xtrain.sav', 'wb'))
pickle.dump(Y_train, open('Ytrain.sav', 'wb'))
pickle.dump(X_test, open('Xtest.sav', 'wb'))
pickle.dump(Y_test, open('Ytest.sav', 'wb'))
pickle.dump(knn, open('trainedknn.sav', 'wb')) # KNN
#pickle.dump(gnb, open('trainedNB.sav', 'wb')) # GNB
#pickle.dump(svm_classifier, open('trainedsvm.sav', 'wb')) # SVM