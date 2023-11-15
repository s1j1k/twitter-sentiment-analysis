"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

# KMeans clustering
from sklearn.cluster import KMeans

# logistic regression model
from sklearn.linear_model import LogisticRegression

from scipy import stats


class KMeansClustering1NN(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y) # assume unlabelled data have None in the y field
        # Store the classes seen during fit
        #self.classes_ = unique_labels(y)
        self.classes_ = unique_labels(y[y != 'None'])
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=0) # 2 clusters is good (???)
        self._logit = LogisticRegression(C=5e1, solver='lbfgs', multi_class='ovr', random_state=17, n_jobs=4) # binary classification problem (positive/negative)
        X_new = self._kmeans.fit_transform(X)
        X_labelled = X_new[y != 'None']
        y_labels = y[y != 'None']
        X_clusters = np.array(self._kmeans.predict(X))

        for n in range(self.n_clusters):
            cluster_y = y[X_clusters == n]
            cluster_y = cluster_y[cluster_y != 'None']
            label = stats.mode(cluster_y)

            if label==None: # there was no labelled data in the cluster 
                labels = []
                indx = np.argmin(X_labelled[:,n], axis=0) # nearest labelled neighbour to the cluster 
                label = y_labels[indx]

            # label the unlabelled data using 1-NN rule (not majority voting)
            cluster_y = y[X_clusters == n]
            y[X_clusters == n] = np.where(cluster_y == 'None', label, cluster_y)[0]

        # cluster_labels = np.array(self._kmeans.fit_predict(X))
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)


        # predict - todo implement as simply clustering 
        #X_test = np.array(test_emb['TFIDF'].tolist())
        #_logit.fit(X,y)
        #y_pred = _logit.predict(X_test)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

