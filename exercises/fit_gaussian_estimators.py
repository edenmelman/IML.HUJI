from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_gaussian = UnivariateGaussian()
    X = np.random.normal(10, 1, 1000)
    uni_gaussian.fit(X)
    print(uni_gaussian.mu_, uni_gaussian.var_, sep=",")

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1001, 10)
    estimated_exp = np.zeros((100,))
    for i in range(0, 100):
        estimated_exp[i] = uni_gaussian.fit(X[:sample_sizes[i]]).mu_
    fig_1 = go.Figure(
        go.Scatter(x=sample_sizes,
                   y=np.abs(estimated_exp - 10)))
    fig_1.update_layout(
        title_text="Estimated Expectation As a Function of Sample Size",
        xaxis_title="Sample Size",
        yaxis_title="ABS distance between"
                    " estimated and real Expectation")
    fig_1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fig_2 = go.Figure(
        # at this point, uni_gaussian is fitted based the full sample-
        # size (1000)
        go.Scatter(x=X, y=uni_gaussian.pdf(X), mode='markers'))
    fig_2.update_layout(title="PDF per sample value",
                        xaxis_title="Sample Value",
                        yaxis_title="PDF Value")
    fig_2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multi_gaussian = MultivariateGaussian()
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    mu = np.array([0, 0, 4, 0])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multi_gaussian.fit(X)
    print(multi_gaussian.mu_)
    print(multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    likelihood = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            mu = np.array([f1[i], 0, f3[j], 0])
            likelihood[i][j] = multi_gaussian.log_likelihood(mu, cov, X)
    fig_5 = go.Figure(go.Heatmap(x=f3, y=f1, z=likelihood),
                      layout=go.Layout(
                          title='Log Likelihood Evaluation'))
    fig_5.update_layout(xaxis_title="f3 Values",
                        yaxis_title="f1 values")
    fig_5.show()

    # Question 6 - Maximum likelihood
    max_val_ind = np.argmax(likelihood)
    max_val_x = max_val_ind % likelihood.shape[1]
    max_val_y = max_val_ind // likelihood.shape[1]
    max_f1_rounded = round(f1[max_val_y], 3)
    max_f3_rounded = round(f3[max_val_x], 3)
    print("max log likelihood model is ({}, {})".format(max_f1_rounded,
                                                        max_f3_rounded))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
