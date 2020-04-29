'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import collections
import random
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import label_binarize


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = self.l2_distance(test_point)
        # Find the index of the k minimum distances
        idx = np.argpartition(distances, k)[:k]
        # Find the digit that occurs the most
        counts = collections.Counter(self.train_labels[idx].astype(int)).most_common()
        most_occurrence = counts[0][1]
        tie_labels = []
        for label in counts:
            if label[1] == most_occurrence:
                tie_labels.append(label[0])
            else:
                break
        return random.choice(tie_labels)

    def get_label_counts(self, test_point, k):
        distances = self.l2_distance(test_point)
        # Find the index of the k minimum distances
        idx = np.argpartition(distances, k)[:k]
        # Find the digit that occurs the most
        counts = collections.Counter(self.train_labels[idx].astype(int)).most_common()
        return counts

    def decision_function(self, test_data, k):
        probabilities = []
        for x in test_data:
            counts = self.get_label_counts(x, k)
            digits = np.zeros(10)
            for count in counts:
                digits[count[0]] = count[1] / k
            probabilities.append(list(digits))
        return np.array(probabilities)


# Helper function for cross validation
def run_10_fold(x, y, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           K is the number of kNN
    output is accuracy a vector of 10-fold cross validation losses one for each k value
    '''
    accuracy = []
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn = KNearestNeighbor(X_train, y_train)
        accuracy.append(classification_accuracy(knn, k, X_test, y_test))
    return np.mean(accuracy, axis=0)


def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    max_avg_accuracy = 0
    max_accu_k = 0
    for k in k_range:
        accuracy = run_10_fold(train_data, train_labels, k)
        if accuracy > max_avg_accuracy:
            max_avg_accuracy = accuracy
            max_accu_k = k
    return max_accu_k, max_avg_accuracy


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predicted_labels = []
    for data_point in eval_data:
        predicted_label = knn.query_knn(data_point, k)
        predicted_labels.append(predicted_label)
    return accuracy_score(eval_labels, predicted_labels)


class MLPNeuralNetworkClassifier(object):
    '''
    Multi-Layer Perceptron Neural Network
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.mlp = MLPClassifier(max_iter=500)
        self.mlp.fit(self.train_data, self.train_labels)

    def query_mlp(self, test_data):
        '''
        Query test data using the MLP Neural Network

        Return the digit labels provided by the algorithm
        '''
        return self.mlp.predict(test_data)

    def score(self, test_data, test_labels):
        return self.mlp.score(test_data, test_labels)

    def decision_function(self, test_data):
        return self.mlp.predict_proba(test_data)


class SVMClassifier(object):
    '''
    SVM classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.1, 1, 10]}
        self.svm = GridSearchCV(svm.SVC(), parameters)
        self.svm.fit(self.train_data, self.train_labels)

    def query_svm(self, test_data):
        '''
        Query test data using the MLP Neural Network

        Return the digit labels provided by the algorithm
        '''
        return self.svm.predict(test_data)

    def score(self, test_data, test_labels):
        return self.svm.score(test_data, test_labels)

    def decision_function(self, test_data):
        return self.svm.decision_function(test_data)


class AdaBoost(object):
    '''
    AdaBoost classifier with a weak classifier as a decision tree with depth 1
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.ada = AdaBoostClassifier()
        self.ada.fit(self.train_data, self.train_labels)

    def query_ada(self, test_data):
        '''
        Query test data using the MLP Neural Network

        Return the digit labels provided by the algorithm
        '''
        return self.ada.predict(test_data)

    def score(self, test_data, test_labels):
        return self.ada.score(test_data, test_labels)

    def decision_function(self, test_data):
        return self.ada.decision_function(test_data)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # Example usage:
    # predicted_label = knn.query_knn(test_data[0], 15)

    # Q3.1.1
    print("For K = 1, the train classification accuracy:")
    print(classification_accuracy(knn, 1, train_data, train_labels))
    print("For K = 1, the test classification accuracy:")
    print(classification_accuracy(knn, 1, test_data, test_labels))
    print("For K = 15, the train classification accuracy:")
    print(classification_accuracy(knn, 15, train_data, train_labels))
    print("For K = 15, the test classification accuracy:")
    print(classification_accuracy(knn, 15, test_data, test_labels))

    # Q3.1.3
    k, avg_accuracy = cross_validation(train_data, train_labels)
    print('The optimal K is', k)
    print("Its train classification accuracy is", classification_accuracy(knn, k, train_data, train_labels))
    print("Its average accuracy across folds is", avg_accuracy)
    print("Its test accuracy is", classification_accuracy(knn, k, test_data, test_labels))

    optimized_knn_predicted_label = []
    for data_point in test_data:
        predicted_label = knn.query_knn(data_point, k)
        optimized_knn_predicted_label.append(predicted_label)

    # Q3.2.1
    mlp = MLPNeuralNetworkClassifier(train_data, train_labels)
    mlp_predicted_label = mlp.query_mlp(test_data)

    # Q3.2.2
    svm = SVMClassifier(train_data, train_labels)
    svm_predicted_label = svm.query_svm(test_data)

    # Q3.2.3
    ada = AdaBoost(train_data, train_labels)
    ada_predicted_label = ada.query_ada(test_data)

    # Q 3.3: ROC
    test_label_binarize = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    knn_fpr = dict()
    knn_tpr = dict()
    knn_roc_auc = dict()
    for i in range(10):
        knn_fpr[i], knn_tpr[i], _ = roc_curve(test_label_binarize[:, i],
                                              knn.decision_function(test_data, k)[:, i])
        knn_roc_auc[i] = auc(knn_fpr[i], knn_tpr[i])

    mlp_fpr = dict()
    mlp_tpr = dict()
    mlp_roc_auc = dict()
    for i in range(10):
        mlp_fpr[i], mlp_tpr[i], _ = roc_curve(test_label_binarize[:, i],
                                              mlp.decision_function(test_data)[:, i])
        mlp_roc_auc[i] = auc(mlp_fpr[i], mlp_tpr[i])

    svm_fpr = dict()
    svm_tpr = dict()
    svm_roc_auc = dict()
    for i in range(10):
        svm_fpr[i], svm_tpr[i], _ = roc_curve(test_label_binarize[:, i],
                                              svm.decision_function(test_data)[:, i])
        svm_roc_auc[i] = auc(svm_fpr[i], svm_tpr[i])

    ada_fpr = dict()
    ada_tpr = dict()
    ada_roc_auc = dict()
    for i in range(10):
        ada_fpr[i], ada_tpr[i], _ = roc_curve(test_label_binarize[:, i],
                                              ada.decision_function(test_data)[:, i])
        ada_roc_auc[i] = auc(ada_fpr[i], ada_tpr[i])

    models = [(mlp_fpr, mlp_tpr, mlp_roc_auc, 'darkorange', 'MLP'),
              (svm_fpr, svm_tpr, svm_roc_auc, 'green', 'SVM'),
              (ada_fpr, ada_tpr, ada_roc_auc, 'red', 'ADA'),
              (knn_fpr, knn_tpr, knn_roc_auc, 'purple', 'KNN')]
    for i in range(10):
        plt.figure()
        for model in models:
            plt.plot(model[0][i], model[1][i], color=model[3],
                     label=model[4]+' ROC curve (area = %0.2f)' % model[2][i])
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of class ' + str(i))
        plt.legend(loc="lower right")
        plt.show()

    # Q3.3: confusion matrix
    print("Confusion matrix:")
    print("KNN: \n", confusion_matrix(test_labels, optimized_knn_predicted_label))
    print("MLP: \n", confusion_matrix(test_labels, mlp_predicted_label))
    print("SVM: \n", confusion_matrix(test_labels, svm_predicted_label))
    print("ADA: \n", confusion_matrix(test_labels, ada_predicted_label))

    # Q3.3: Accuracy
    print("Accuracy:")
    print("KNN: ", accuracy_score(test_labels, optimized_knn_predicted_label))
    print("MLP: ", accuracy_score(test_labels, mlp_predicted_label))
    print("SVM: ", accuracy_score(test_labels, svm_predicted_label))
    print("ADA: ", accuracy_score(test_labels, ada_predicted_label))

    # Q3.3: Precision
    print("Precision:")
    print("KNN: ", precision_score(test_labels, optimized_knn_predicted_label, average=None))
    print("MLP: ", precision_score(test_labels, mlp_predicted_label, average=None))
    print("SVM: ", precision_score(test_labels, svm_predicted_label, average=None))
    print("ADA: ", precision_score(test_labels, ada_predicted_label, average=None))

    # Q3.3: Recall
    print("Recall:")
    print("KNN: ", recall_score(test_labels, optimized_knn_predicted_label, average=None))
    print("MLP: ", recall_score(test_labels, mlp_predicted_label, average=None))
    print("SVM: ", recall_score(test_labels, svm_predicted_label, average=None))
    print("ADA: ", recall_score(test_labels, ada_predicted_label, average=None))


if __name__ == '__main__':
    main()