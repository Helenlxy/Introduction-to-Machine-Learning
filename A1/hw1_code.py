import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

real_addr = "./clean_real.txt"
fake_addr = "./clean_fake.txt"
SEED = 123


# Q(a)
def load_data(real, fake):
    """
    Loads the data, preprocesses it using a vectorizer
    and splits the entire dataset randomly into 70% training, 15% validation, and 15% test examples.
    """
    real_data = np.loadtxt(real, dtype=str, delimiter="\n")
    fake_data = np.loadtxt(fake, dtype=str, delimiter="\n")
    real_indicator = np.ones(len(real_data))
    fake_indicator = np.zeros(len(fake_data))
    data = np.concatenate((real_data, fake_data), axis=0)
    indicators = np.concatenate((real_indicator, fake_indicator), axis=0)
    count_vectorizer = CountVectorizer()
    data_matrix = count_vectorizer.fit_transform(data)
    x_train, x_temp, y_train, y_temp = train_test_split(data_matrix, indicators, test_size=0.3, random_state=SEED)
    x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED)
    return x_train, y_train, x_validation, y_validation, x_test, y_test, count_vectorizer


# Q(b)
def select_tree_model(x_train, y_train, x_validation, y_validation):
    """
    trains the decision tree
    classifier using at least 5 different sensible values of max_depth, as well as two different split
    criteria (Information Gain and Gini coefficient)
    """
    depths = [16, 32, 64, 128, 256]
    criterias = ['entropy', 'gini']
    max_accuracy = -1000
    best_tree = None
    for criteria in criterias:
        for depth in depths:
            new_model = DecisionTreeClassifier(criterion=criteria, max_depth=depth, random_state=SEED)
            new_model.fit(x_train, y_train)

            accuracy = new_model.score(x_validation, y_validation)
            print('Accuracy for depth', depth, "and criteria", criteria, ": ", accuracy)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_tree = new_model
    return best_tree


print("Output for Q(b)")
x_train, y_train, x_validation, y_validation, x_test, y_test, count_vectorizer = load_data(fake_addr, real_addr)
best_tree = select_tree_model(x_train, y_train, x_validation, y_validation)

# Q(c)
print("Output for Q(c)")
best_accuracy = best_tree.score(x_test, y_test)
print("Accuracy of the best hyperparameter on the test dataset: ", best_accuracy)

plt.figure(figsize=(12, 12))
sklearn.tree.plot_tree(best_tree, max_depth=2, feature_names=count_vectorizer.get_feature_names(),
                       class_names=['fake', 'real'])
plt.show()


# Q(d)
def compute_information_gain(x_train, y_train, split, vocabularies):
    """
    Computes the information gain of split on x_train and y_train
    """
    index_of_split = vocabularies[split]
    left = []
    right = []
    for i in range(len(y_train)):
        if x_train[i, index_of_split] <= 0.5:
            left.append(y_train[i])
        else:
            right.append(y_train[i])
    root_entropy = compute_entropy(y_train)
    left_entropy = compute_entropy(left)
    right_entropy = compute_entropy(right)
    left_probability = len(left) / len(y_train)
    right_probability = len(right) / len(y_train)
    information_gain = root_entropy - left_entropy * left_probability - right_entropy * right_probability
    return information_gain


def compute_entropy(node):
    """
    Computes the entropy of node
    """
    total = len(node)
    appearance = sum(node)
    not_appearance = len(node) - sum(node)
    entropy = 0
    if appearance > 0:
        entropy -= (appearance / total) * math.log(appearance / total, 2)
    if not_appearance > 0:
        entropy -= (not_appearance / total) * math.log(not_appearance / total, 2)
    return entropy


print("Output for Q(d)")
print("The information gain of 'donald':",
      compute_information_gain(x_train, y_train, 'donald', count_vectorizer.vocabulary_))
print("The information gain of 'trumps':",
      compute_information_gain(x_train, y_train, 'trumps', count_vectorizer.vocabulary_))
print("The information gain of 'china':",
      compute_information_gain(x_train, y_train, 'china', count_vectorizer.vocabulary_))
print("The information gain of 'good':",
      compute_information_gain(x_train, y_train, 'good', count_vectorizer.vocabulary_))


# Q(e)
def select_knn_model(x_train, y_train, x_validation, y_validation):
    """
    uses a KNN classifier to classify between real vs. fake news
    """
    max_accuracy = -1000
    best_knn = None
    validation_errors = []
    training_errors = []
    for i in range(1, 21):
        new_model = KNeighborsClassifier(n_neighbors=i)
        new_model.fit(x_train, y_train)
        accuracy = new_model.score(x_validation, y_validation)
        validation_error = 1 - accuracy
        validation_errors.append(validation_error)
        training_error = 1 - new_model.score(x_train, y_train)
        training_errors.append(training_error)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_knn = new_model
    return best_knn, validation_errors, training_errors


print("Output for Q(e)")
knn_model, validation_errors, training_errors = select_knn_model(x_train, y_train, x_validation, y_validation)
best_knn_accuracy = knn_model.score(x_test, y_test)
print("Accuracy of the best KNN model on the test dataset: ", best_knn_accuracy)
x = [i for i in range(1, 21)]
fig, ax = plt.subplots()
ax.set_xlim(21, 0)
ax.scatter(x, validation_errors, label='Validation')
ax.plot(x, validation_errors)
ax.scatter(x, training_errors, c='red', label='Train')
ax.plot(x, training_errors, c='red')
plt.xlabel('k - Number of Nearest Neighbors')
plt.ylabel('Test Error')
ax.legend()
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.show()
