from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        n_classes = self.classes_.shape[0]

        self.pi_ = np.zeros((n_classes,))
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for (k, label) in enumerate(self.classes_):
            n_k = np.count_nonzero(y == label)
            self.pi_[k] = n_k / n_samples
            for j in range(n_features):
                label_inds = np.array(np.where(y == label)).flatten()
                labeled_sample_j_feature = X[label_inds, j]
                labeled_samples_sum = np.sum(labeled_sample_j_feature,
                                             axis=0)
                self.mu_[k][j] = labeled_samples_sum / n_k
                self.vars_[k][j] = np.sum((labeled_sample_j_feature -
                                           self.mu_[k][j]) ** 2) / (n_k -1)
                # TODO what is the unbaised version here?

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

        # for i in range(n_samples):
        #     for k in range(n_classes):
        #         mu_k = self.mu_[k, :]
        #         pi_k = self.pi_[k]
        #         sigma_k = self.vars_[k, :]
        #         x_i = X[i, :]
        #         j_dependent = -(x_i - mu_k ** 2) / (
        #                 2 * (sigma_k ** 2)) - np.log(
        #             sigma_k * np.square(2 * np.pi))
        #         likelihoods[i][k] = j_dependent.sum() + np.log(pi_k)

        for k in range(n_classes):
            cov = np.diag(self.vars_[k, :])
            mu_k = self.mu_[k, :]
            mahalanobis = np.einsum("bi,ij,bj->b", X - mu_k,
                                    np.linalg.inv(cov), X - mu_k)
            col = np.exp(-0.5 * mahalanobis) / np.sqrt((2 * np.pi) ** X.shape[1] ** np.linalg.det(cov))
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
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)

