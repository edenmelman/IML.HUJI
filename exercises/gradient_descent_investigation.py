import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.model_selection

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker=dict(color="black"))],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []
    def recorder(solver, weight, val, grad, t, eta, delta):
        values.append(val)
        weights.append(weight)
    return recorder, values, weights


def plot_convergence_rate(values: List[np.ndarray], title: str):
    return go.Figure([go.Scatter(x=np.arange(len(values)),
                                 y=values,
                                 mode="markers+lines",
                                 marker=dict(color="black"))],
                     layout=go.Layout(title=f"Convergence Rate as a Function of Gradient Descent Iteration {title}", xaxis_title="Iteration", yaxis_title="Norm"))


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        lr = FixedLR(eta)
        for module in [L1, L2]:
            init_model = module(init)
            callback, values, weights = get_gd_state_recorder_callback()
            GradientDescent(learning_rate=lr, callback=callback).fit(f=init_model, X=None, y=None)
            plot_descent_path(module, descent_path=np.vstack(weights), title=f" | Module:{module.__name__} | Eta:{eta}").show()
            plot_convergence_rate(values, title=f" | Module:{module.__name__} | Eta:{eta}").show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # Plot algorithm's convergence for the different values of gamma
    # Plot descent path for gamma=0.95
    fig = go.Figure()
    l1 = L1(init)
    for gamma in gammas:
        l1.weights = init  # initialize in every iteration
        lr = ExponentialLR(eta, gamma)
        callback, values, weights = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=lr, callback=callback).fit(f=l1, X=None, y=None)
        fig.add_scatter(x=np.arange(len(values)), y=values, name=f"gamma-{gamma}")
        if gamma == 0.95:
            plot_descent_path(module=L1, descent_path=np.vstack(weights), title=f" | Module: L1| Eta:{eta}").show()

    fig.update_layout(title="Convergence Rate as a Function of Gradient Descent Iteration",xaxis_title="Iteration", yaxis_title="Norm").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return sklearn.model_selection.train_test_split(df.drop(['chd', 'row.names'], axis=1), df.chd, train_size=train_portion)
    #return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)
    #TODO decide on the relevant split test


def fit_logistic_regression():
    from sklearn.metrics import roc_curve, auc
    # Load and split SA Heard Disease dataset
    #X_train, y_train, X_test, y_test = load_data() my implementaion
    X_train, X_test, y_train, y_test = load_data()  # sklearn
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    y_prob = LogisticRegression().fit(X_train, y_train).predict_proba(X_train)
    # TODO align test train wtv

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines',
                         text=thresholds, name="", showlegend=False,
                         marker = dict(size=5),
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title="ROC Curve Of Fitted Model",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_alpha = thresholds[np.argmax(tpr-fpr)]
    test_error = LogisticRegression(alpha=best_alpha).fit(X_train,y_train).loss(X_test,y_test)
    print(f"**Logistic** | Bast Alpha: {best_alpha} | Test Error: {test_error}")




    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    from IMLearn.model_selection import cross_validate
    from IMLearn.metrics import mean_square_error
    from IMLearn.metrics import misclassification_error
    for penalty in ["l1", "l2"]:
        train_losses = np.zeros((7,))
        validation_losses = np.zeros((7,))
        for ind, lam in enumerate(lamdas):
            train_losses[ind], validation_losses[ind] = cross_validate(LogisticRegression(penalty=penalty, lam=lam, alpha=0.5), X_train,
                                                                       y_train, misclassification_error)
        best_lam = lamdas[np.argmin(validation_losses)]
        test_error = LogisticRegression(penalty=penalty, lam=best_lam, alpha=0.5).fit(X_train, y_train).loss(X_test, y_test)
        print(f"**Regularized Using {penalty} penalty**| Bast Lamda: {best_lam} | Test Error: {test_error}")








if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    #compare_exponential_decay_rates()
    #fit_logistic_regression()
