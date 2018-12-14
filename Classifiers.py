from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np



#%% CLASSIFIERS 

###################################NAIVE BAYES###############################################

def naiveBayes(X_train, Y_train):
    gnb = naive_bayes.GaussianNB();
    cv_scores = cross_val_score(gnb, X_train, Y_train, cv=10,scoring='roc_auc')
    # Usually, NB classifiers don't overfit, but cross validation was performed to assure a better estimate
    # on all folds
    
    print("Accuracy Gaussian Naive Bayes: ",(cv_scores.mean(), cv_scores.std() * 2))
    maxAccuracy_gnb=cv_scores.mean(); 
    return maxAccuracy_gnb

###############################LOGISTIC REGRESSION############################################

def logReg(X_train, y_train, regularization_params):
    scores_lr=[]
    for C in regularization_params:
        clf_LR = LogisticRegression(C=C)
        cv_scores = cross_val_score(clf_LR, X_train, y_train, cv=10,scoring='roc_auc')
        print("C=",C,"Accuracy Logistic Regression",(cv_scores.mean(), cv_scores.std() * 2))
        scores_lr.append(cv_scores.mean())
        
    maxIndex_lr=np.argmax(scores_lr)
    maxAccuracy_lr=cv_scores[maxIndex_lr]
    
    clf_LR = LogisticRegression(C=regularization_params[maxIndex_lr])
    clf_LR.fit(X_train, y_train)
    return clf_LR, maxAccuracy_lr

###############################K NEAREST NEIGHBOUR############################################

def KNeighbors(n_neighbors, X_train, y_train):
    scores_knn=[]
    for n in n_neighbors:
        knn = KNeighborsClassifier(n_neighbors=n) #=n)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=10,scoring='roc_auc')
        print("n=",n,"Accuracy kNN",(cv_scores.mean(), cv_scores.std() * 2))
        scores_knn.append(cv_scores.mean())
    
    maxIndex_knn=np.argmax(scores_knn)
    maxAccuracy_knn=cv_scores[maxIndex_knn]
    
    knn = KNeighborsClassifier(n_neighbors = n_neighbors[maxIndex_knn])
    knn.fit(X_train, y_train)
    return knn, maxAccuracy_knn

###############################SVM grid search################################
# SÓ VAMOS USAR ESTE NO FIM, POR ENQUANTO BASTA O RANDOM (EM BAIXO)
    
def SVMs_grid(X_train, y_train, parameters):
    #y = label_binarize(y_train, classes=[0, 1, 2])
    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=10)
    print("# Tuning hyper-parameters")
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    maxAccuracy_SVM=np.amax(means)
    
    clf=svm.SVC(probability=True, **clf.best_params_)
    clf.fit(X_train, y_train)
    
    return clf, maxAccuracy_SVM

###############################SVM random search###############################

def SVMs_rand(X_train, y_train, parameters):

    clf = RandomizedSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=10)
    print("# Tuning hyper-parameters")
    clf.fit(X_train, y_train);
    
    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    maxAccuracy_SVM=np.amax(means); 
    
    clf=svm.SVC(probability=True, **clf.best_params_)
    clf.fit(X_train, y_train)
    
    return clf, maxAccuracy_SVM

###############################DECISION TREE############################################

def decision_trees(X_train, y_train, param_grid):

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, param_grid,cv=10,scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    print("Best parameters set found on training set:")
    print()
    print(grid_search.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means_gs = grid_search.cv_results_['mean_test_score']
    stds_gs = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means_gs, stds_gs, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    maxAccuracy_trees=np.amax(means_gs)
    grid_search=DecisionTreeClassifier(**grid_search.best_params_)
    grid_search.fit(X_train, y_train)
    return grid_search, maxAccuracy_trees

