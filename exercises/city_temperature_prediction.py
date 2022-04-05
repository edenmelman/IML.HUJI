import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])

    # drop invalid values
    df = df.dropna()
    df.drop(df[(df.Day < 1) | (df.Day > 31)].index, inplace=True)
    df.drop(df[(df.Year < 1900) | (df.Year > 2022)].index, inplace=True)
    df.drop(df[(df.Month < 1) | (df.Month > 12)].index, inplace=True)
    df.drop(df[df.Temp < -30].index, inplace=True)

    # adjust fields
    df["Year"] = df["Year"].astype(str)

    # add fields
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_subset = df.loc[df.Country == 'Israel']
    fig_1 = px.scatter(israel_subset, x="DayOfYear", y="Temp",
                       color='Year')
    fig_1.update_layout(title='Temprature as function of Day of Year')
    #fig_1.show()

    israel_month_temp_std = israel_subset.groupby(['Month']).std()
    israel_month_temp_std = israel_month_temp_std.reset_index()
    fig_2 = px.bar(israel_month_temp_std, x='Month', y='Temp')
    fig_2.update_layout(title='Temperature STD per Month')
    #fig_2.show()

    # Question 3 - Exploring differences between countries
    temp_per_country_month = df.groupby(['Country', 'Month'],
                                        as_index=False).Temp.agg(['mean', 'std'])
    temp_per_country_month = temp_per_country_month.reset_index()
    fig_3 = px.line(temp_per_country_month, x="Month", y="mean",
                    color='Country', error_y='std')
    fig_3.update_layout(title="Montly Avarage Temprature per Country")
    #fig_3.show()

    # Question 4 - Fitting model for different values of `k`
    temp = israel_subset.pop('Temp')
    train_x, train_y, test_x, test_y = split_train_test(israel_subset, temp, 0.75)
    polinom_degree = np.arange(1, 11)
    losses = np.zeros((10,))
    for i in range(10):
        polifit_model = PolynomialFitting(polinom_degree[i])
        polifit_model.fit(train_x['DayOfYear'].to_numpy(), train_y.to_numpy())
        losses[i] = polifit_model.loss(test_x['DayOfYear'].to_numpy(), test_y.to_numpy())
    fig_4 = px.bar(x=polinom_degree, y=losses)
    fig_4.update_layout(title='Loss per Polinom Degree',
                        xaxis_title="Polinom Degree",
                        yaxis_title="Loss")
    fig_4.show()

    # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
