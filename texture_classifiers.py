import numpy as np
from matplotlib import pyplot as plt
import Classifiers
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as areaUnderCurve

#%% Load files

features=pickle.load(open('texture_features.sav', 'rb'))
total_texture=pickle.load(open('texture_labels.sav', 'rb'))

np.random.seed(0)

#%% Data Split

#TrainTestSplit
X_train,X_test,Y_train,Y_test=train_test_split(features, total_texture,test_size=0.20,random_state=0)

#Conversion to ndarray
X_train_ndarray=np.vstack(X_train[:])
Y_train_ndarray=np.hstack(Y_train[:])

#Data Standerization (To ensure mean=0 and std=1)
scaler = StandardScaler().fit(X_train_ndarray)
X_train_ndarray=scaler.transform(X_train_ndarray)


#%% Classifiers


#SVM Grid search
parameters = [{'kernel': ['rbf'],
               'gamma': [0.001, 0.01, 0.1, 1, 10],
                'C': [0.1, 1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]
clf_SVM, maxAccuracy_SVM=Classifiers.SVMs_grid(X_train_ndarray, Y_train_ndarray, parameters)
print('SVMs grid: %f',clf_SVM.score(X_train_ndarray, Y_train_ndarray))

# KNN 
n_neighbors = [1,3,5,7,9,11,13,15]
knn, knnaccuracy=Classifiers.KNeighbors(n_neighbors, X_train_ndarray, Y_train_ndarray) #Training K-neighbours
print('KNN score for train: %f',knn.score(X_train_ndarray,Y_train_ndarray))

# Decision Trees 
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
tree, maxAccuracy_trees=Classifiers.decision_trees(X_train_ndarray, Y_train_ndarray, param_grid)

#Let's save these files to avoid running them again 
pickle.dump(X_test, open('Xtrain_texture.sav', 'wb'))
pickle.dump(Y_train, open('Ytrain_texture.sav', 'wb'))
pickle.dump(X_test, open('Xtest_texture.sav', 'wb'))
pickle.dump(Y_test, open('Ytest_texture.sav', 'wb'))
pickle.dump(knn, open('trainedknn_texture.sav', 'wb'))
pickle.dump(clf_SVM, open('trainedSVM_texture.sav', 'wb'))
pickle.dump(tree,open('trainedtree_texture.sav', 'wb'))

#%%=====================================
#  PERFORMANCE ANALYSIS
#=======================================

# Loading the files to avoid running them again: 
#tree=pickle.load(open('trainedtree_texture.sav', 'rb'))
#X_test=pickle.load(open('Xtest_texture.sav', 'rb'))
#Y_test=pickle.load(open('Ytest_texture.sav', 'rb'))

# Create each variable for performance analysis
X_test_ndarray=np.vstack(X_test[:])
X_test_ndarray=scaler.transform(X_test_ndarray)

Y_nonsolid=np.zeros(len(Y_test))
Y_subsolid=np.zeros(len(Y_test))
Y_solid=np.zeros(len(Y_test))

for j in range(len(Y_test)):
    if (Y_test[j]==0):
        Y_nonsolid[j]=1
    elif (Y_test[j]==1):
        Y_subsolid[j]=1
    elif (Y_test[j]==2):
        Y_solid[j]=1

#%% PREDICTIONS
        
prediction=[]
prediction.append(tree.predict(X_test_ndarray))
prediction=prediction[0]
nonsolid=np.zeros(len(Y_test))
subsolid=np.zeros(len(Y_test))
solid=np.zeros(len(Y_test))

for j in range(len(prediction)):
    if (prediction[j]==0):
        nonsolid[j]=1
    elif (prediction[j]==1):
        subsolid[j]=1
    elif (prediction[j]==2):
        solid[j]=1
        

#%% PREFORMANCE EVALUATION
        
TN=np.zeros(3)
TP=np.zeros(3)
FN=np.zeros(3)
FP=np.zeros(3)
Sens=np.zeros(3)
Acc=np.zeros(3)
Spec=np.zeros(3)

TN[0], FP[0], FN[0], TP[0]=confusion_matrix(Y_nonsolid, nonsolid).ravel()
TN[1], FP[1], FN[1], TP[1]=confusion_matrix(Y_subsolid, subsolid).ravel()
TN[2], FP[2], FN[2], TP[2]=confusion_matrix(Y_solid, solid).ravel()

for i in range(3):
    Acc[i]=(TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i])            
    Sens[i]=TP[i]/(TP[i]+FN[i])
    Spec[i]=TN[i]/(TN[i]+FP[i])

    auc=areaUnderCurve([0,1-Spec[i],1], [0,Sens[i],1]) #It's the same for both men and women
    print("sklearn's AUC= ",auc)

    jaccard=TP[i]/(TP[i]+FP[i]+FN[i])
    print("Jaccard Coefficient= ",jaccard)
    
# ROC Curve
fpr2, tpr2, thresholds = roc_curve(Y_solid, solid)
fpr1, tpr1, thresholds = roc_curve(Y_subsolid, subsolid)
fpr0, tpr0, thresholds = roc_curve(Y_nonsolid, nonsolid)

plt.figure(figsize=(6, 6))
plt.plot(fpr0,  tpr0)
plt.plot(fpr1, tpr1)
plt.plot(fpr2, tpr2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
