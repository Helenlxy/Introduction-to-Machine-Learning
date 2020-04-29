from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

SEED = 123


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # Plot feature i against y
        plt.scatter([data[i] for data in X], y)
        plt.xlim(left=0)
        plt.xlabel(features[i])
        plt.ylabel('target')
    
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # Linear regression
    # Remember to use np.linalg.solve instead of inverting!
    bias_X = np.asarray([np.insert(x, 0, 1) for x in X])
    xtx = np.dot(bias_X.transpose(), bias_X)
    xtt = np.dot(bias_X.transpose(), Y)
    return np.linalg.solve(xtx, xtt)


def mean_percentage_error(y_test, y_predict):
    sum = 0.0
    for i in range(len(y_test)):
        sum += abs((y_test[i] - y_predict[i])/y_test[i])
    return sum/len(y_test)


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # Normalize data
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    # Fit regression model
    w = fit_regression(X_train, y_train)

    feature_count = X.shape[1]
    print("feature      weight")
    for i in range(feature_count+1):
        if i == 0:
            print("BIAS       {}".format(w[i]))
        else:
            print("{}       {}".format(features[i-1], w[i]))

    # Compute fitted values, MSE, etc.
    bias_X_test = np.asarray([np.insert(x, 0, 1) for x in X_test])
    fitted_test = np.dot(bias_X_test, w)
    mse = mean_squared_error(y_test, fitted_test)
    mae = mean_absolute_error(y_test, fitted_test)
    mpe = mean_percentage_error(y_test, fitted_test)
    print("Mean squared error on test set:", mse)
    print("Mean absolute error on test set:", mae)
    print("Mean percentage error on test set:", mpe*100, "%")


if __name__ == "__main__":
    main()

