import numpy

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    # adjust fields
    df['date'] = pd.to_datetime(df['date'], errors="coerce")

    # clean invalid fields
    df = df.dropna()
    df.drop(df[df.price <= 0].index, inplace=True)
    df.drop(df[df.bedrooms <= 0].index, inplace=True)
    df.drop(df[df.bathrooms <= 0].index, inplace=True)
    df.drop(df[df.sqft_living <= 0].index, inplace=True)
    df.drop(df[df.floors <= 0].index, inplace=True)
    df.drop(df[df.sqft_above <= 0].index, inplace=True)
    df.drop(df[df.sqft_basement < 0].index, inplace=True)
    df.drop(df[(df.yr_built < 1900) | (df.yr_built > 2020)].index,
            inplace=True)
    df.drop(df[df.date.isnull()].index, inplace=True)

    # add fields
    df['sqft-sqft15_living'] = df.apply(
        lambda row: row.sqft_living - row.sqft_living15, axis=1)
    df['sqft-sqft15_lot'] = df.apply(
        lambda row: row.sqft_lot - row.sqft_lot15, axis=1)
    df['house_age'] = df.apply(lambda row: row.date.year - row.yr_built,
                               axis=1)
    df = pd.get_dummies(df, columns=['zipcode'])
    # TODO add zipcodes

    # remove fields
    df = df.drop(columns=['id', 'lat', 'long', 'date', 'sqft_living15',
                          'sqft_lot15'])
    response = df.pop('price')
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for (featureName, featureData) in X.iteritems():
        pc = round(pearson_correlation_calc(X[featureName], y), 3)
        fig = px.scatter(x=featureData, y=y, trendline='ols')
        fig.update_layout(
            title="Correlation between {} and price. Pearson Correlation is: {}".format(
                featureName, pc),
            xaxis_title=featureName,
            yaxis_title="price")
        # TODO change to y.name?
        pio.write_image(fig,
                        output_path + "/{}.png".format(featureName))


def pearson_correlation_calc(feature: pd.Series,
                             response: pd.Series) -> float:
    cov = numpy.cov(feature, response)[0, 1]
    return cov / (numpy.std(feature) * numpy.std(response))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, res = load_data("C:/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, res, "C:/IML.HUJI/ex_temp")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(df, res, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lin_reg = LinearRegression()
    training_set = train_x.assign(response=train_y)  # joining for sampling
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    loss_per_p = np.zeros((91,))
    loss_std_per_p = np.zeros((91,))
    p = np.arange(10, 101)

    for i in range(0, len(p)):
        print("entered")
        losses = np.zeros((10,))
        for j in range(0, 10):
            partial_training_set = training_set.sample(
                frac=(p[i] / 100))
            response = partial_training_set.pop('response')
            partial_training_set = partial_training_set.to_numpy()
            partial_response = response.to_numpy()

            losses[j] = lin_reg.fit(partial_training_set, response)._loss(test_x, test_y)

        loss_per_p[i] = losses.mean()
        loss_std_per_p[i] = losses.std()

    upper_error_bound = loss_per_p + (2 * loss_std_per_p)
    lower_error_bound = loss_per_p - (2 * loss_std_per_p)

    fig = go.Figure([go.Scatter(x=p, y=loss_per_p),
                    go.Scatter(x=p, y=upper_error_bound, marker=dict(color="#444")),
                    go.Scatter(x=p, y=lower_error_bound, marker=dict(color="#444"), fill='tonexty')])
    fig.update_layout(
        title="MSE per Training Sample Size Percentage",
        xaxis_title="Training Sample Size Percentage",
        yaxis_title="MSE")
    fig.show()
