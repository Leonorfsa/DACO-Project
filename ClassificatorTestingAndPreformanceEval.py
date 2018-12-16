import numpy as np
import pickle
import Functions
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as areaUnderCurve
from sklearn.metrics import jaccard_similarity_score
from matplotlib import pyplot as plt
from ClassificatorTraining import scaler


trained_knn = pickle.load(open('trainedknn_texture.sav', 'rb'))
#trained_NB=pickle.load(open('trained____.sav', 'rb'))

X_test=pickle.load(open('Xtest_texture.sav', 'rb'))
Y_test=pickle.load(open('Ytest_texture.sav', 'rb'))



# Data Standerization
X_test_ndarray=np.vstack(X_test[:])
X_test_ndarray=scaler.transform(X_test_ndarray)
Y_test_array=np.hstack(Y_test[:])


#%% PREDICTIONS

prediction=[]
prediction.append(trained_knn.predict(X_test_ndarray))
prediction=prediction[0]

#%% PREFORMANCE EVALUATION

prediction_list=[]
for i in range(0,len(Y_test)):
    prediction_image=np.reshape(prediction[i*2601:(i+1)*2601],[51,51])
    GT_image=np.reshape(Y_test[i],[51,51])
    #Functions.show2DImages(prediction_image, GT_image, 1)
    prediction_list.append(prediction[i*2601:(i+1)*2601])
    
TN, FP, FN, TP =confusion_matrix(Y_test_array, prediction).ravel()

Acc=(TP+TN)/(TP+TN+FP+FN)            
Sens=TP/(TP+FN)
Spec=TN/(TN+FP)

auc=areaUnderCurve([0,1-Spec,1], [0,Sens,1])
print("sklearn's AUC= ",auc)

jaccard=TP/(TP+FP+FN)
print("Jaccard Coefficient= ",jaccard)
#AUC for KNN 5 neighbours with 70% for test was 0.826 and Jaccard coefficient 0.85
#AUC for SVM with 70% for test was 0.846 and Jaccard coefficient 0.828

