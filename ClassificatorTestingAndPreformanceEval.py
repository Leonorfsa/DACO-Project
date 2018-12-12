import numpy as np
import pickle
import Functions
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc as areaUnderCurve
from sklearn.metrics import jaccard_similarity_score

total_features=np.load('totalfeatures.npy')
total_labels=np.load('totallabels.npy')
#load dos indices de val
trained_knn = pickle.load(open('trainedknn.sav', 'rb'))
X_val=pickle.load(open('Xval.sav', 'rb'))
Y_val=pickle.load(open('Yval.sav', 'rb'))
trained_SVM=pickle.load(open('SVM.sav', 'rb'))
elapsed_time=pickle.load(open('time.sav','rb'))
print('Time passed: ',elapsed_time)

#%% PREDICTIONS
prediction=trained_knn.predict(X_val)
prediction2=trained_SVM.predict(X_val)


#%% PREFORMANCE EVALUATION

for i in range(int(len(Y_val)/2601)):
    prediction_single=prediction[i*2601:(i+1)*2601]
    prediction2_single=prediction2[i*2601:(i+1)*2601]
    mask_single=Y_val[i*2601:(i+1)*2601]
    prediction_image=np.reshape(prediction_single,[51,51])
    prediction2_image=np.reshape(prediction2_single,[51,51])
    GT_image=np.reshape(mask_single,[51,51])
    Functions.show2DImages(prediction2_image, GT_image)

TN, FP, FN, TP =confusion_matrix(Y_val, prediction2).ravel()

Acc=(TP+TN)/(TP+TN+FP+FN)            
Sens1=TP/(TP+FN)
Spec1=TN/(TN+FP)
Sens0=TN/(TN+FP)
Spec0=TP/(TP+FN)

auc=areaUnderCurve([0,1-Spec1,1], [0,Sens1,1]) #It's the same for both men and women
print("sklearn's AUC= ",auc)

jaccard=jaccard_similarity_score(Y_val,prediction)
print("Jaccard Coefficient= ",jaccard)
#AUC for KNN 5 neighbours with 70% for test was 0.826 and Jaccard coefficient 0.85
#AUC for SVM with 70% for test was 0.846 and Jaccard coefficient 0.828
