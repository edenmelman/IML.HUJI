from __future__ import annotations
from typing import NoReturn

import numpy

#from ...base import BaseEstimator
#from ...metrics import mean_square_error
import numpy as np
from numpy.linalg import pinv

from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error


class LinearRegression (BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self,
                 include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            X = np.insert(X, 0, [1], axis=1)

        X = numpy.asmatrix(X)
        self.weights_ = np.matmul(np.linalg.pinv(X), y).transpose()

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
        if self.include_intercept_:
            X = np.insert(X, 0, [1],
                          axis=1)  # weights contain intercept
        return np.asarray(np.matmul(X, self.weights_)).flatten()
    # TODO should I reshape?

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return mean_square_error(y, self._predict(X))
        # TODO check if should handle not fitted yet situations

if __name__ == '__main__':
    lin_reg = LinearRegression(include_intercept=False)
    X = np.array([[2, 4, 5], [7, 8, 9], [1, 2, 3], [2, 4, 6]])
    y = np.array([25, 50, 14, 28])
    lin_reg.fit(X, y)
    print(lin_reg.predict(X))
    print(lin_reg._loss(X, y))
