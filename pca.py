import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets


class PCA:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.eigenvector = None
        self.overal_variability = None

    def fit(self,X):

        self.mean = np.mean(X,axis=0)
        X = X - self.mean

        cov = np.cov(X.T)
        self.overal_variability = np.sum(np.diag(cov))
        print(cov,self.overal_variability)

        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        idx = np.argsort(eigenvalues)[::-1]
        eigenvector = eigenvectors[idx]


        print(eigenvector.shape)
        print(eigenvalues,idx)
        self.eigenvector = eigenvector[0:self.n_components]
        print(self.eigenvector.shape)

    def transform(self,X):
        X = X - self.mean
        return np.dot(X,self.eigenvector.T)

    def calculate_variability_captured(self,X):

        new_cov =  np.cov(X.T)
        if X.shape[1] == 1:
            return new_cov/self.overal_variability
        new_variability =  np.sum(np.diag(new_cov))/self.overal_variability
        print( np.sum(np.diag(new_cov)))
        return new_variability

# import numpy as np
# from collections import Counter
# import pandas as pd
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from toolz.sandbox import fold
# import pandas as pd



#
iris = datasets.load_iris()
X,y = iris.data, iris.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)
variability_captured = pca.calculate_variability_captured(X_projected)
print(f'This is athe variability captured {variability_captured}')

print(f'Shape of original data {X.shape}')
print(f'Shape of transformed data {X_projected.shape}')


x1 = X_projected[:,0]
x2 = X_projected[:,1]
plt.scatter(x1,x2,c=y, edgecolor = 'none', alpha = 0.8, cmap = plt.cm.get_cmap('viridis',3))
plt.show()



#### Pizza dataset PCA

pizza_csv = pd.read_csv('pizza.csv')
pizza_target = pizza_csv['brand'].tolist()
pizza_data  = pizza_csv.drop('brand',axis=1)

pca = PCA(2)
pca.fit(pizza_data)
X_projected = pca.transform(pizza_data)
variability_captured = pca.calculate_variability_captured(X_projected)
print(f'This is athe variability captured {variability_captured}')


x1 = X_projected[:,0]
x2 = X_projected[:,1]
plt.scatter(x1,x2,c=pizza_target, edgecolor = 'none', alpha = 0.8, cmap = plt.cm.get_cmap('viridis',3))
plt.show()y