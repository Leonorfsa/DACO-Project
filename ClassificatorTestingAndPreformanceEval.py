import numpy as np
import pickle
import Functions
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as areaUnderCurve
from sklearn.metrics import jaccard_similarity_score
from matplotlib import pyplot as plt
from ClassificatorTraining import scaler

#total_features=np.load('totalfeatures.npy')
#total_labels=np.load('totallabels.npy')
#load dos indices de val
trained_knn = pickle.load(open('trainedknn.sav', 'rb'))
#trained_NB=pickle.load(open('trainedNB.sav', 'rb'))

X_test=pickle.load(open('Xtest.sav', 'rb'))
Y_test=pickle.load(open('Ytest.sav', 'rb'))
#elapsed_time=pickle.load(open('time.sav','rb'))
#print('Time passed: ',elapsed_time)

X_test_ndarray=np.vstack(X_test[:])
X_test_ndarray=scaler.transform(X_test_ndarray)

Y_test_array=np.hstack(Y_test[:])


#%% PREDICTIONS
prediction=[]
#for i in range(0,len(X_test)):
#    prediction.append(trained_knn.predict(X_test[i]))
prediction.append(trained_knn.predict(X_test_ndarray))
prediction=prediction[0]
# FALTAM AS OUTRAS PREDICTIONS


#%% PREFORMANCE EVALUATION

prediction_list=[]
for i in range(0,len(Y_test)):
    prediction_image=np.reshape(prediction[i*2601:(i+1)*2601],[51,51])
    GT_image=np.reshape(Y_test[i],[51,51])
    #Functions.show2DImages(prediction_image, GT_image, 1)
    prediction_list.append(prediction[i*2601:(i+1)*2601])
    
TN, FP, FN, TP =confusion_matrix(Y_test_array, prediction).ravel()

Acc=(TP+TN)/(TP+TN+FP+FN)            
Sens1=TP/(TP+FN)
Spec1=TN/(TN+FP)
Sens0=TN/(TN+FP)
Spec0=TP/(TP+FN)

auc=areaUnderCurve([0,1-Spec1,1], [0,Sens1,1]) #It's the same for both men and women
print("sklearn's AUC= ",auc)

jaccard=TP/(TP+FP+FN)
print("Jaccard Coefficient= ",jaccard)
#AUC for KNN 5 neighbours with 70% for test was 0.826 and Jaccard coefficient 0.85
#AUC for SVM with 70% for test was 0.846 and Jaccard coefficient 0.828

## ROC Curve
#fpr, tpr, thresholds = roc_curve(Y_val, prediction2)
#plt.figure(figsize=(6, 6))
#plt.plot(fpr,  tpr)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.title('ROC curve for classifier')
#plt.xlabel('False Positive Rate (1 - Specificity)')
#plt.ylabel('True Positive Rate (Sensitivity)')
#plt.grid(True)


##%%  APRESENTAR NUMA FIGURA
#classifiers = ['NB', 'LR', 'kNN', 'SVM', 'RF']
#x_pos = np.arange(len(classifiers))
#CTEs = [0.6, 0.83, 0.82, 0.859, 0.809]
#error = [0.53/2, 0.36/2, 0.5/2, 0.342/2, 0.400/2]
#
## Build the plot
#fig, ax = plt.subplots()
#ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
#ax.set_ylabel('ROC AUC')
#ax.set_xticks(x_pos)
#ax.set_xticklabels(classifiers)
#ax.set_title('AUC for different classifiers (10-fold CV)')
#ax.yaxis.grid(True)
#
## Save the figure and show
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
#plt.show()
