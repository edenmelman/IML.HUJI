from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, \
    LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    no_noise_response = lambda x: (x + 3) * (x + 2) * (x + 1) * (
            x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples, endpoint=True)
    y = no_noise_response(X) + np.random.normal(0, noise, (n_samples,))
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X),
                                                        y, 0.66)
    train_x, train_y, test_x, test_y = train_x.to_numpy().flatten(), \
                                       train_y.to_numpy().flatten(), \
                                       test_x.to_numpy().flatten(), \
                                       test_y.to_numpy().flatten()

    go.Figure([go.Scatter(x=X, y=no_noise_response(X), mode='lines',
                          name='Noiseless Samples',
                          marker=dict(color='black')),
               go.Scatter(x=train_x, y=train_y,
                          mode='markers',
                          name='Train Samples',
                          marker=dict(color='blue')),
               go.Scatter(x=test_x, y=test_y,
                          mode='markers',
                          name='Test Samples',
                          marker=dict(color='orange'))]).update_layout(yaxis_title='y', xaxis_title='x',
                  title=f"Noisless Model and Train & Test Samples with Noise. Noise level: {noise}. Number of samples: {n_samples}").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_range = np.arange(0, 11)
    train_losses = np.zeros((11,))
    validation_losses = np.zeros((11,))
    for k in k_range:
        train_losses[k], validation_losses[k] = cross_validate(
            PolynomialFitting(k), train_x,
            train_y, mean_square_error)
    go.Figure([go.Scatter(x=k_range, y=train_losses,
                          mode='lines',
                          name='Train Error',
                          marker=dict(color='blue')),
               go.Scatter(x=k_range, y=validation_losses,
                          mode='lines',
                          name='Validation Error',
                          marker=dict(color='orange'))]).update_layout(yaxis_title='Error', xaxis_title='k',
                  title=f"Train and Validation Errors for each polinomial degree. Noise level: {noise}. Number of samples: {n_samples}").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_losses)
    print(f"****num of samples: {n_samples}. Noise: {noise}****")
    print(f"Best polinomial degree: {best_k}")
    test_error = PolynomialFitting(best_k).fit(
        train_x, train_y).loss(test_x, test_y)
    print(f"Test error using test degree: {round(test_error,2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes(as_frame=True)
    X = diabetes['data'].to_numpy()
    y = diabetes['target']
    train_x, train_y, test_x, test_y = X[0:n_samples], y[
                                                       0:n_samples], X[
                                                                     n_samples:], y[
                                                                                  n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_lam_range = np.linspace(0.001, 0.2, n_evaluations)
    lasso_lam_range = np.linspace(0.001, 1, n_evaluations)
    ridge_train_losses = np.zeros((n_evaluations,))
    lasso_train_losses = np.zeros((n_evaluations,))
    ridge_validation_losses = np.zeros((n_evaluations,))
    lasso_validation_losses = np.zeros((n_evaluations,))

    for i in range(n_evaluations):
        ridge_train_losses[i], ridge_validation_losses[
            i] = cross_validate(
            RidgeRegression(ridge_lam_range[i]), train_x, train_y,
            mean_square_error)
        lasso_train_losses[i], lasso_validation_losses[
            i] = cross_validate(
            Lasso(lasso_lam_range[i]), train_x, train_y,
            mean_square_error)

    go.Figure([go.Scatter(x=ridge_lam_range, y=ridge_train_losses,
                          mode='lines',
                          name='ridge_train_losses',
                          marker=dict(color='blue')),
               go.Scatter(x=ridge_lam_range, y=ridge_validation_losses,
                          mode='lines',
                          name='ridge_validation_losses',
                          marker=dict(color='orange'))]).update_layout(
        yaxis_title='Error', xaxis_title='Lamda',
        title=f"Ridge Regularization: Train and Validation Errors for each lamda degree. Number of samples used for training: {n_samples}").show()

    go.Figure([go.Scatter(x=lasso_lam_range, y=lasso_train_losses,
                          mode='lines',
                          name='lasso_train_losses',
                          marker=dict(color='blue')),
               go.Scatter(x=lasso_lam_range, y=lasso_validation_losses,
                          mode='lines',
                          name='lasso_validation_losses',
                          marker=dict(color='orange'))
               ]).update_layout(yaxis_title='Error', xaxis_title='Lamda',
                  title=f"Lasso Regularization: Train and Validation Errors for each lamda. Number of samples used for training: {n_samples}").show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lam = ridge_lam_range[np.argmin(ridge_validation_losses)]
    ridge_test_error = RidgeRegression(ridge_best_lam).fit(train_x,
                                                           train_y).loss(
        test_x, test_y)

    lasso_best_lam = lasso_lam_range[np.argmin(lasso_validation_losses)]
    lasso_test_error = mean_square_error(test_y,
                                         Lasso(lasso_best_lam).fit(
                                             train_x, train_y).predict(
                                             test_x))

    lin_reg_test_error = LinearRegression().fit(train_x, train_y).loss(
        test_x, test_y)

    print("***Ridge***", f"Best Lamda:{round(ridge_best_lam, 3)}",
          f"Test Error:{round(ridge_test_error, 3)}", sep='\n')
    print("\n***Lasso***", f"Best Lamda:{round(lasso_best_lam, 3)}",
          f"Test Error:{round(lasso_test_error, 3)}", sep='\n')
    print("\n***Least Squares***",
          f"Test Error: {round(lin_reg_test_error, 3)}", sep='\n')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
