import numpy as np
import pickle
import Functions
import imageProcessing as proc
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as areaUnderCurve
from sklearn.metrics import jaccard_similarity_score
from matplotlib import pyplot as plt

scaler=pickle.load(open('scaler.pkl','rb'))
trained_knn = pickle.load(open('trainedknn.sav', 'rb'))
#trained_NB=pickle.load(open('trained____.sav', 'rb'))

X_test=pickle.load(open('Xtest.sav', 'rb'))
Y_test=pickle.load(open('Ytest.sav', 'rb'))

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
post_prediction=[]
for i in range(len(Y_test)):
    prediction_image=np.reshape(prediction[i*2601:(i+1)*2601],[51,51])
    
    processed_image=proc.watershedfilt(prediction_image)
    GT_image=np.reshape(Y_test[i],[51,51])
    Functions.show2DImages(processed_image, GT_image, 1)
    post_prediction.append(np.ravel(processed_image))


prediction_list=np.hstack(post_prediction)
TN, FP, FN, TP =confusion_matrix(Y_test_array, prediction_list).ravel()

Acc=(TP+TN)/(TP+TN+FP+FN)            
Sens=TP/(TP+FN)
Spec=TN/(TN+FP)

auc=areaUnderCurve([0,1-Spec,1], [0,Sens,1])
print("sklearn's AUC= ",auc)

jaccard=TP/(TP+FP+FN)
print("Jaccard Coefficient= ",jaccard)
#AUC for KNN 5 neighbours with 70% for test was 0.826 and Jaccard coefficient 0.85
#AUC for SVM with 70% for test was 0.846 and Jaccard coefficient 0.828

