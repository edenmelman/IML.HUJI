import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

USED_ADABOOST = False


def generate_data(n: int, noise_ratio: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = None

    if USED_ADABOOST:
        adaboost = pickle.load(open("save.p", "rb"))

    else:
        adaboost = AdaBoost(wl=DecisionStump, iterations=n_learners)
        adaboost.fit(train_X, train_y)
        pickle.dump(adaboost, open("save.p", "wb"))

    train_losses = np.zeros((n_learners,))
    test_losses = np.zeros((n_learners,))
    n_learners_for_predict = np.arange(1, n_learners+1)
    for i in n_learners_for_predict:
        train_losses[i-1] = adaboost.partial_loss(train_X, train_y, i)
        test_losses[i-1] = adaboost.partial_loss(test_X, test_y, i)
    go.Figure([
        go.Scatter(x=n_learners_for_predict, y=train_losses,
                   mode='markers + lines',
                   name=r'$Train Loss$'),
        go.Scatter(x=n_learners_for_predict, y=test_losses,
                   mode='markers + lines',
                   name=r'$Test Loss$')]).update_layout(
        title="Train and Test Errors as Function of Number of Learners Used for prediction").show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig_1 = make_subplots(rows=2, cols=2,
                          subplot_titles=[rf"$\textbf{{{t}}} stumps$"
                                          for t in
                                          T],
                          horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig_1.add_traces([decision_surface(
            lambda X: adaboost.partial_predict(X, t), lims[0],
            lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y,
                                   colorscale=[custom[0],
                                               custom[
                                                   -1]],
                                   line=dict(
                                       color="black",
                                       width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig_1.update_layout(
        title="Predictions of the ensemble up to different sizes:", ) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_1.show()

    # Question 3: Decision surface of best performing ensemble
    from IMLearn.metrics.loss_functions import accuracy
    best_k = np.argmin(test_losses) + 1
    fig_2 = go.Figure(data=[
        decision_surface(lambda X: adaboost.partial_predict(X, best_k),
                         lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y,
                               colorscale=[
                                   custom[0],
                                   custom[-1]],
                               line=dict(
                                   color="black",
                                   width=1)))
    ])
    fig_2.update_layout(
        title=f"Best Performing Ensamble Size: {best_k} . Accuracy: {accuracy(test_y, adaboost.predict(test_X))}")
    fig_2.show()

    # Question 4: Decision surface with weighted samples

    normalized_weights = (adaboost.D_ / np.max(adaboost.D_)) * 25
    fig_3 = go.Figure(data=[
        decision_surface(lambda X: adaboost.partial_predict(X, n_learners),
                         lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=train_y,
                               size=normalized_weights,
                               colorscale=[
                                   custom[0],
                                   custom[-1]],
                               line=dict(
                                   color="black",
                                   width=1)))
    ])
    fig_3.update_layout(
        title="Fitted Adaboost Samples Weights")
    fig_3.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
