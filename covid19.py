# Made by Jason Elter (ID: 318634110) for IML 2020.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Similiar\same functions in this file and in linear_model.py since 
# they were asked to be able to run not in the same directory.

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
    coefficients_vector = np.linalg.pinv(design_matrix).T.dot(response)
    return coefficients_vector, singular_values


def predict(design_matrix, w):
    """Returns the predicted value by the given model.

    Parameters:
        design_matrix: a design matrix (numpy array with p rows and n columns).
        w: a coefficients vector (numpy array with p + 1 rows).

    Returns:
        A vector with the predicted value by the model (numpy array with n rows).
    """
    return design_matrix.T.dot(w)


def load_data(path):
    """Loads the dataset from a csv file at the given path and returns a pandas DataFrame.

    Parameters:
        path: the path to the csv file to load.

    Returns:
        A pandas DataFrame representing the dataset.
    """
    dataset = pd.read_csv(path)  # 18
    dataset['log_detected'] = np.log(dataset['detected'])  # 19
    return dataset


if __name__ == '__main__':
    # 18, 19
    data_pd = load_data("covid19_israel.csv")

    # 20
    day_num_matrix, log_detected_vector = data_pd['day_num'].to_numpy(), data_pd['log_detected'].to_numpy()
    new_design_matrix = np.array([np.ones(day_num_matrix.size), day_num_matrix])  # Add row of ones.
    log_detected_fit, _ = fit_linear_regression(new_design_matrix, log_detected_vector)

    # 21
    log_prediction = predict(new_design_matrix, log_detected_fit)
    for y_label, data_vector, prediction in [("log_detected", log_detected_vector, log_prediction),
                                             ("detected", data_pd['detected'].to_numpy(), np.exp(log_prediction))]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(y_label + " as a function of day_num", fontsize=16)
        ax.plot(day_num_matrix, data_vector, 'o', label="Data")
        ax.plot(day_num_matrix, prediction, '-', label="Fitted Curve")
        ax.set_xlabel("day_num")
        ax.set_ylabel(y_label)
        ax.legend()
    plt.show()
