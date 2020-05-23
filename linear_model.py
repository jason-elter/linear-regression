# Made by Jason Elter (ID: 318634110) for IML 2020.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def fit_linear_regression(design_matrix, response):
    """Solves a linear regression problem and returns the coefficients
    vector and the singular values of x.

    Parameters:
        design_matrix: a design matrix (numpy array with p rows and n columns).
        response: a response vector (numpy array with n rows).

    Returns:
        The coefficients vector (numpy array with p rows), a numpy array of the singular values of design_matrix.
    """
    singular_values = np.linalg.svd(design_matrix, compute_uv=False)
    new_design_matrix = np.insert(design_matrix, 0, np.ones(design_matrix.shape[1]), axis=0)  # Add row of ones.
    coefficients_vector = np.linalg.pinv(new_design_matrix).T.dot(response)
    return coefficients_vector, singular_values


def predict(design_matrix, w):
    """Returns the predicted value by the given model.

    Parameters:
        design_matrix: a design matrix (numpy array with p rows and n columns).
        w: a coefficients vector (numpy array with p + 1 rows).

    Returns:
        A vector with the predicted value by the model (numpy array with n rows).
    """
    new_design_matrix = np.insert(design_matrix, 0, np.ones(design_matrix.shape[1]), axis=0)  # Add row of ones.
    return new_design_matrix.T.dot(w)


def mse(y, y_hat):
    """Returns the MSE (Mean Square Error) over the received samples.

    Parameters:
        y: a response vector (numpy array with n rows).
        y_hat: a prediction vector (numpy array with n rows).

    Returns:
        The MSE over the received samples.
    """
    return np.square(y_hat - y).sum() / y.size


def load_data(path):
    """Loads the dataset from a csv file at the given path,
    preforms all the needed preprocessing and returns a valid design matrix.

    Parameters:
        path: the path to the csv file to load.

    Returns:
        A valid design matrix representing the dataset after preprocessing(pandas DataFrame), a response vector(numpy).
    """
    # Load.
    dataset = pd.read_csv(path, usecols=list(range(1, 21)), parse_dates=['date'],
                          date_parser=(
                              lambda x: pd.to_datetime(x, format="%Y%m%dT000000") if str(x).endswith(
                                  "T000000") else np.nan))

    # Preprocess.
    del dataset['sqft_living']  # sqft_living is unnecessary since sqft_living = sqft_above + sqft_basement.
    dataset.dropna(inplace=True)  # Remove incomplete rows.
    drop_indices = dataset[(dataset['price'] < 0) | (dataset['bedrooms'] < 0) | (dataset['bathrooms'] < 0) |
                           (dataset['sqft_lot'] <= 0) | (dataset['floors'] < 1) |
                           (~(dataset['waterfront'] == 0) & ~(dataset['waterfront'] == 1)) |
                           (dataset['view'] < 1) | (dataset['condition'] < 1) | (dataset['grade'] < 1) |
                           (dataset['sqft_above'] < 0) | (dataset['sqft_basement'] < 0) |
                           (dataset['zipcode'] <= 0) | (dataset['sqft_living15'] < 0) |
                           (dataset['sqft_lot15'] < 0)].index
    dataset.drop(drop_indices, inplace=True)

    # Handle categorical features.
    dataset['date'] = dataset['date'].astype('int64') // 1e9  # Change date to Epoch time. (gives numerical meaning)
    dataset['zipcode'] = dataset['zipcode'].astype('int64')
    dataset = pd.get_dummies(dataset, columns=['zipcode'], drop_first=True)  # One Hot Encoding.

    # Split into design matrix and response vector.
    response = dataset['price'].to_numpy()
    del dataset['price']
    return dataset.T, response


def plot_singular_values(singular_values):
    """Receives a collection of singular values and plots them in descending order.

    Parameters:
        singular_values: a numpy array of singular values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Scree Plot of singular values", fontsize=16)
    ax.plot(np.arange(1, singular_values.size + 1), singular_values, 'o-', label="Singular Values")
    ax.set_yscale('log')
    ax.set_xlabel("Running Index")
    ax.set_ylabel("Singular Values")
    ax.legend()
    plt.show()


def feature_evaluation(design_matrix, response):
    """Plots for every non-categorical feature a graph (scatter plot) of the feature values and the response values.
    It then also computes and shows on the graph the Pearson Correlation between the feature and the response.

    Parameters:
        design_matrix: a design matrix (pandas DataFrame with p rows and n columns).
        response: a response vector (numpy array with n rows).

    Returns:
        The coefficients vector (numpy array with p rows), a numpy array of the singular values of design_matrix.
    """
    new_design_matrix = design_matrix.T
    for column in new_design_matrix:
        if column.startswith('zipcode'):
            break
        pearson = ((new_design_matrix[column] * response).mean() - new_design_matrix[
            column].mean() * response.mean()) / (np.std(new_design_matrix[column]) * np.std(response))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Prices as a function of " + column + " values\nPearson correlation: " + str(pearson), fontsize=16)
        ax.plot(new_design_matrix[column].to_numpy(), response, 'o', label="Response values (prices)")
        ax.set_xlabel(column + " values")
        ax.set_ylabel("Response values (prices)")
        ax.legend()

    plt.show()


# Helper function for calculating result for question 16 efficiently.
def question_16_helper(split_index):
    new_training_set, new_response = x_train[:, :split_index], y_train[:split_index]
    result = fit_linear_regression(new_training_set, new_response)[0]
    return mse(y_test, predict(x_test, result))


if __name__ == '__main__':
    # 15
    data_pd, response_vector = load_data("kc_house_data.csv")
    data = data_pd.to_numpy()
    singulars = fit_linear_regression(data, response_vector)[1]
    plot_singular_values(singulars)

    # 16
    # Create random 1 to 3 ratio split.
    x_train, x_test, y_train, y_test = train_test_split(data.T, response_vector, test_size=0.25)
    x_train, x_test = x_train.T, x_test.T

    # Calculate and show plot.
    p_array = np.arange(1, 101)
    split_sizes = (p_array * (x_train.shape[1] / 100)).astype(np.int)
    mse_array = np.vectorize(question_16_helper)(split_sizes)
    graph = plt.figure()
    sub = graph.add_subplot(111)
    sub.set_title("MSE over the test-set as a function of p%", fontsize=16)
    sub.plot(p_array, mse_array, label="MSE")
    sub.set_yscale('log')
    sub.set_xlabel("p% (percentage of training-set used)")
    sub.set_ylabel("MSE (Mean Square Error)")
    sub.legend()
    plt.show()

    # 17 test code
    feature_evaluation(data_pd, response_vector)
