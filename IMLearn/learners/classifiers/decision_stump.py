from __future__ import annotations
from typing import Tuple, NoReturn

import numpy as np
from itertools import product

from IMLearn import BaseEstimator

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        signs = [-1, 1]
        cur_err = y.shape[0] + 1
        cur_feature = -1
        cur_sign = 0
        cur_thr = 0
        for j in range(0, X.shape[1]):
            for sign in signs:
                potential_thr, potential_err = self._find_threshold(
                    X[:, j], y, sign)
                if potential_err <= cur_err:
                    cur_feature = j
                    cur_sign = sign
                    cur_thr = potential_thr
                    cur_err = potential_err
        self.sign_ = cur_sign
        self.j_ = cur_feature
        self.threshold_ = cur_thr


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.where(X[:, self.j_] < self.threshold_, -self.sign_,
                          self.sign_)
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        p = values.argsort()
        values = values[p]
        labels = labels[p]
        best_thr = np.NINF
        cur_thr_err = self._weighted_misclassification_error(labels, np.full(shape=(values.shape[0],), fill_value=sign))
        best_thr_err = cur_thr_err
        for i in range(1, values.shape[0]):
            cur_thr_err -= (labels[i-1] * (-sign))
            if values[i-1] == values[i]: #  if the previous value equals current value, cur value should not be the threshold.
                continue
            if cur_thr_err < best_thr_err:
                best_thr_err = cur_thr_err
                best_thr = values[i]
        total_sum = np.sum(np.abs(labels))
        return best_thr, best_thr_err/total_sum


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
        return self._weighted_misclassification_error(y, y_pred)


    def _weighted_misclassification_error(self, weighted_y_true:np.ndarray,y_pred:np.ndarray) -> float:
        true_pred_match = np.array(np.sign(weighted_y_true)==y_pred)
        mismatch_ind = np.where(true_pred_match == False)[0]
        return np.sum(np.absolute(weighted_y_true[mismatch_ind]))