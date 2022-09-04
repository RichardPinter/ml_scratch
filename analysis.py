#### Creating KNN Class


import numpy as np
from collections import Counter
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from toolz.sandbox import fold


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))



def split(data,k_fold=5):
    n = len(data)//k_fold
    return [ data[i*n:n*(i+1)] if i !=(k_fold-1) else data[(i)*n:] for i in range(k_fold)]


def knn_algorithm(x_train,x_test,y_train,y_test,number_of_neighbours=3):
    clf = KNN(number_of_neighbours)
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)

    return np.sum( predictions == y_test) / len(y_test)

def cross_validation(x,y,k_fold,number_of_neighbours):
    x_value = split(x, k_fold)
    y_value = split(y, k_fold)
    acc_value_llist = []
    for i in range(k_fold):
        temp_x = x_value.copy()
        temp_y = y_value.copy()

        x_test = temp_x.pop(i)
        y_test = temp_y.pop(i)

        x_train = np.concatenate(temp_x)
        y_train = np.concatenate(temp_y)
        acc = knn_algorithm(x_train, x_test, y_train, y_test,number_of_neighbours)
        acc_value_llist.append(acc)

    return np.mean(acc_value_llist)


def grid_search(x,y,k_fold,neighbours_llist):
    acc_value_llist = []
    for number_of_neighbours in neighbours_llist:
        acc = cross_validation(x, y, k_fold, number_of_neighbours)
        acc_value_llist.append(acc)
    return zip(acc_value_llist,neighbours_llist)



class KNN():

    def __init__(self,k=3):
        self.k = k
        self.k_nearest = None
        self.predicted_labelels = None

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,x_test):
        self.predicted_labelels = np.array([self._predict_single(x)[0] for x in x_test])
        self.k_nearest = {idx: self._predict_single(x)[1] for idx,x in enumerate(x_test)}
        return self.predicted_labelels


    def _predict_single(self,x):
        ## Compute distances
        distances  = [euclidean_distance(x,y) for y in self.x_train]
        ## Sort distances
        k_indices = np.argsort(distances)[:self.k]
        ## Labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        ## Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common, k_nearest_labels

    def performance_tester(self,x_test,cutoff=0.1):

        for idx,x in enumerate(x_test):
            predicted = self.predicted_labelels(idx)
            neighbours =  self.k_nearest[idx]




if __name__ == "__main__":

    iris = datasets.load_iris()
    x,y = iris.data, iris.target
    k_fold = 4
    neighbours_llist = [3,5,7,9,11]
    grid_result = grid_search(x, y, k_fold, neighbours_llist)
    for i,j, in grid_result:
        print("Number of neighbours :",i,"Accuracy:",j)



    #### Built in functions
    from sklearn.neighbors import KNeighborsClassifier

    ### Splitting the data into training and testing
    xtrain,xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.25,random_state = 12345)

    #
    ### Built in model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(xtrain, ytrain)

    ytrain_pred = knn.predict(xtrain)
    ytest_pred = knn.predict(xtest)

    acc_train = (ytrain == ytrain_pred).mean()
    acc_test = (ytest == ytest_pred).mean()

    print(f"Train set: {acc_train}")
    print(f"Test set: {acc_test}")


    ### Hand Crafted
    clf = KNN(5)
    clf.fit(xtrain,ytrain)

    predictions = clf.predict(xtest)
    acc = (predictions == ytest).mean()
    print(f"My test predictions: {acc}")
    # print(f'This is the closest neighbours {clf.k_nearest[4]}')



    # # ### Built in Tuner
    # from sklearn.neighbors import KNeighborsClassifier
    # knn2 = KNeighborsClassifier()


    # from sklearn.model_selection import RandomizedSearchCV
    # distributions = dict(n_neighbors=list(range(1,50)))
    # clf = RandomizedSearchCV(knn2, distributions, random_state=0, n_iter = 10)    #
    # search = clf.fit(xtrain, ytrain)
    # best_param= search.best_params_['n_neighbors']
    # print(f'Best results with the following tune parameter{best_param}')
    #
    # from sklearn.model_selection import GridSearchCV
    #
    # parameters = dict(n_neighbors=list(range(1, 50)))
    # clf2 = GridSearchCV(knn2, parameters)
    # # run
    # search2 = clf2.fit(xtrain, ytrain)
    # search2.best_params_['n_neighbors']
    # print(f'Best results with the following tune parameter{ search2.best_params_["n_neighbors"]}')
