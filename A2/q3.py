from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))


# helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # The list of -||x-x^(i)||^2/2tau^2
    A = l2(test_datum.transpose(), x_train)
    A = A[0]
    A = [(-1) * i / (2 * (tau ** 2)) for i in A]
    B = max(A)
    denominator = np.sum(np.exp(A - B))
    # The list of a^(i)
    ais = []
    for i in range(x_train.shape[0]):
        ais.append(np.exp(A[i] - B) / denominator)
    ais = np.asarray(ais)
    # The diagonal matrix A
    d = np.diag(ais)
    xtax = np.dot(np.dot(x_train.transpose(), d), x_train)
    xtay = np.dot(np.dot(x_train.transpose(), d), y_train)
    w = np.linalg.solve(xtax + 1e-8 * np.eye(xtax.shape[0]), xtay)
    return np.dot(test_datum.transpose(), w)


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i, :].reshape(d, 1), x_train, y_train, tau) \
                                for i in range(N_test)])
        losses[j] = ((predictions.flatten() - y_test.flatten()) ** 2).mean()
    return losses


def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    losses = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        losses.append(run_on_fold(X_test, y_test, X_train, y_train, taus))
    return np.mean(losses, axis=0)


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(taus, losses)
    plt.xlabel('taus')
    plt.ylabel('losses')
    plt.show()
    print("min loss = {}".format(losses.min()))
