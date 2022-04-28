from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.pi_ = np.zeros((n_classes,))
        self.mu_ = np.zeros((n_classes, n_features))

        for (k, label) in enumerate(self.classes_):
            n_k = np.count_nonzero(y == label)
            label_inds = np.array(np.where(y == label)).flatten()
            labeled_samples_sum = np.sum(X[label_inds, :], axis=0)
            self.pi_[k] = n_k / n_samples
            self.mu_[k] = labeled_samples_sum / n_k

        self.cov_ = np.zeros((n_features, n_features))
        for (i, x) in enumerate(X):
            y_label_ind = np.where(self.classes_ == y[i])[0]
            y_mu = self.mu_[y_label_ind, :]
            self.cov_ += np.outer(x - y_mu, x - y_mu)
        self.cov_ = self.cov_ / (n_samples - n_classes)
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        max_k_per_sample = self.likelihood(X).argmax(1)
        k_to_label = lambda k: self.classes_[k]
        return k_to_label(max_k_per_sample)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]
        likelihoods = np.zeros((n_samples, n_classes))
        for k in range(n_classes):
            mu_k = self.mu_[k, :]
            mahalanobis = np.einsum("bi,ij,bj->b", X - mu_k,
                                    self._cov_inv, X - mu_k)
            col = np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi) ** X.shape[1] * det(self.cov_))
            col = col * self.pi_[k]
            likelihoods[:, k] = col
        return likelihoods


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        from IMLearn.metrics import misclassification_error
        y_pred = self.predict(X)  # TODO change to predict with under?
        return misclassification_error(y, y_pred)
