#%% CLASSIFIERS 

#_____________________________________
# K-NEIGHBORS
#_______________________________________

def KNeighbors(n_neighbors, X, y):
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(X,y)
    return knn

