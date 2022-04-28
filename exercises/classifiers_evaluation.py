import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, \
    GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset('C:/IML.HUJI/datasets/' + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def extract_loss(fit: Perceptron, x: np.ndarray, response: int):
            losses.append(fit.loss(X, y))

        Perceptron(callback=extract_loss).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(
            go.Scatter(x=list(range(1, len(losses) + 1)), y=losses))
        fig.update_layout(
            title=f"Loss as function of fitting iteration for {n} samples",
            xaxis_title="Iteration",
            yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (
                l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (
                l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    from IMLearn.metrics import accuracy
    for f in ["gaussian1.npy", "gaussian2.npy"]:

        # load data
        X, y = load_dataset('C:/IML.HUJI/datasets/' + f)
        models = [GaussianNaiveBayes(), LDA()]
        symbols = np.array(["circle-open-dot", "star-diamond-open-dot",
                            "triangle-up-open-dot"])
        # Fit models and predict over training set
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["GNB", "LDA"],
                            horizontal_spacing=0.01,
                            vertical_spacing=.03)
        for i, m in enumerate(models):
            y_pred = m.fit(X, y).predict(X)
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1],
                                     mode="markers", showlegend=False,
                                     marker=dict(color=y_pred, symbol=symbols[y],
                                                 size=10,
                                                 line=dict(color="black", width=1))), row=1,
                          col=i + 1)

            fig.add_trace(
                go.Scatter(mode="markers", x=m.mu_[:, 0], y=m.mu_[:, 1],
                           marker=dict(symbol='x', color='black')),
                row=1, col=i + 1)

            model_name = 'GNB' if i == 0 else 'LDA'
            fig.layout.annotations[i].update(
                text=f"{model_name},accuracy:{accuracy(y, y_pred)}")

            for k in range(m.classes_.shape[0]):
                if i == 0:
                    cov = np.diag(m.vars_[k, :])
                else:
                    cov = m.cov_
                fig.add_trace(get_ellipse(m.mu_[k, :], cov), row=1,
                              col=i + 1, )

        fig.update_layout(margin=dict(t=100), showlegend=False,
                          title=rf"$\textbf{{Prediction of Classifiers - {f} Dataset}}$")
        fig.show()
        # TODO redactor code change models to dictironary for i, (k, v) in enumerate(example_dict.items())
        #     print(i, k, v)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        # raise NotImplementedError()

        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
