import numpy as np

## Assume that the features are mutually independent
class NaiveBayes:


    def fit(self,X,y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        ### For each class we need variance, mean and prior
        self._mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self.variance = np.zeros((n_classes,n_features),dtype=np.float64)
        self.priors = np.zeros(np,dtype=np.float64)


        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self.variance[c,:] = X_c.var(axis=0)
            self.priors[c,:] = X_c.shape[0]/float(n_samples)

    def predict(self,X):
        y_predict = [self._predict(x) for x in X]
        return y_predict


    def _predict(self,x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pd(idx,x)))
            posterior = prior+class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]


    def _pdf(self,class_index,x):
        mean = self._mean[class_index]
        variance = self.variance[class_index]
        numerator = np.exp((x-mean)**2 / (2*variance))
        denominator = np.sqrt1/(2*np.pi*variance)
        return numerator/denominator