import numpy as np
import pickle
import Functions
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc as areaUnderCurve
from sklearn.metrics import jaccard_similarity_score
from matplotlib import pyplot as plt

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

# FALTAM AS OUTRAS PREDICTIONS

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

# ROC Curve
fpr, tpr, thresholds = roc_curve(Y_val, prediction2)
plt.figure(figsize=(6, 6))
plt.plot(fpr,  tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


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
