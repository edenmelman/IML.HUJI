from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = np.zeros((cv,))
    validation_score = np.zeros((cv,))
    indices = np.arange(X.shape[0])
    folds_indices = np.array_split(indices, cv)
    for i in range(cv):
        X_validation, y_validation = X[folds_indices[i]], y[folds_indices[i]]
        X_train, y_train = X[np.setdiff1d(indices, folds_indices[i])], y[np.setdiff1d(indices, folds_indices[i])]
        estimator_copy = deepcopy(estimator)
        estimator_copy.fit(X=X_train, y=y_train)
        train_score[i] = scoring(y_train, estimator_copy.predict(X_train))
        validation_score[i] = scoring(y_validation, estimator_copy.predict(X_validation))
    return train_score.mean(), validation_score.mean()

