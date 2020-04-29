'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(10):
        same_label_digits = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(same_label_digits, axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    mean_mles = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        same_label_digits = data.get_digits_by_label(train_data, train_labels, i)
        sum = np.zeros((64, 64))
        for digit in same_label_digits:
            temp = np.reshape(digit - mean_mles[i], (64, 1))
            sum += np.dot(temp, np.transpose(temp))
        covariances[i] = np.divide(sum, len(same_label_digits))
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diags = []
    for i in range(0, 10):
        cov_diag = np.diag(covariances[i])
        log_cov_diag = np.log(cov_diag + 0.01)
        cov_diags.append(log_cov_diag.reshape((8, 8)))

    all_concat = np.concatenate(cov_diags, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = len(digits)
    answer = np.zeros((n, 10))
    for index, digit in enumerate(digits):
        d = len(digit)
        for i in range(10):
            mean = means[i]
            cov = covariances[i] + 0.01*np.identity(64)
            temp = digit - mean
            answer[index, i] = ((2*np.pi)**(-d/2))*(np.linalg.det(cov)**(-1/2))*np.exp((-1/2)*np.linalg.multi_dot([np.transpose(temp), np.linalg.inv(cov), temp]))
    return np.log(answer)


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative_likelihoods = generative_likelihood(digits, means, covariances)
    denominator = np.log(np.sum(np.exp(generative_likelihoods) * 0.1, axis=1).reshape(len(digits), 1))
    return generative_likelihoods + np.log(0.1) - denominator


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    cl_sum = 0.0
    for index, digit in enumerate(digits):
        cl_sum += cond_likelihood[index, int(labels[index])]

    return cl_sum / len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    plot_cov_diagonal(covariances)

    avg_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print("The average conditional log-likelihood for train set:")
    print(avg_train)
    print("The average conditional log-likelihood for test set:")
    print(avg_test)

    train_predict_labels = classify_data(train_data, means, covariances)
    test_predict_labels = classify_data(test_data, means, covariances)
    train_predict_accuracy = sum(1 for index, label in enumerate(train_labels) if train_predict_labels[index] == label) / len(train_labels)
    test_predict_accuracy = sum(1 for index, label in enumerate(test_labels) if test_predict_labels[index] == label) / len(test_labels)
    print("The accuracy on the train set:")
    print(train_predict_accuracy)
    print("The accuracy on the test set:")
    print(test_predict_accuracy)


if __name__ == '__main__':
    main()