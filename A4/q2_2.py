'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))

    for i in range(10):
        same_label_digits = data.get_digits_by_label(train_data, train_labels, i)
        N = len(same_label_digits)
        for j in range(64):
            N_H = np.sum(same_label_digits[:, j])
            # +1 and +2 ->  add two training cases: one of which has every pixels OFF and the other has every pixels ON
            # equivalent to using a prior which is alpha - 1 = 1 and alpha + beta - 2 = 2
            eta[i][j] = (N_H + 1) / (N + 2)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    images = []
    for i in range(10):
        img_i = class_images[i]
        images.append(img_i.reshape((8, 8)))
    all_concat = np.concatenate(images, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for i in range(10):
        for j in range(64):
            generated_data[i][j] = np.random.binomial(1, eta[i][j])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    n = len(bin_digits)
    answer = np.zeros((n, 10))
    for index, digit in enumerate(bin_digits):
        for i in range(10):
            # likelihood = 1
            # for j in range(64):
            #     like
            answer[index][i] = np.product((eta[i]**digit) * ((1-eta[i])**(1-digit)))

    return np.log(answer)

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative_likelihoods = generative_likelihood(bin_digits, eta)
    denominator = np.log(np.sum(np.exp(generative_likelihoods) * 0.1, axis=1).reshape(len(bin_digits), 1))
    return generative_likelihoods + np.log(0.1) - denominator

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    cl_sum = 0.0
    for index, digit in enumerate(bin_digits):
        cl_sum += cond_likelihood[index, int(labels[index])]

    return cl_sum / len(bin_digits)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)

    avg_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test = avg_conditional_likelihood(test_data, test_labels, eta)
    print("The average conditional log-likelihood for train set:")
    print(avg_train)
    print("The average conditional log-likelihood for train set:")
    print(avg_test)

    train_predict_labels = classify_data(train_data, eta)
    test_predict_labels = classify_data(test_data, eta)
    train_predict_accuracy = sum(
        1 for index, label in enumerate(train_labels) if train_predict_labels[index] == label) / len(train_labels)
    test_predict_accuracy = sum(
        1 for index, label in enumerate(test_labels) if test_predict_labels[index] == label) / len(test_labels)
    print("The accuracy on the train set:")
    print(train_predict_accuracy)
    print("The accuracy on the test set:")
    print(test_predict_accuracy)
if __name__ == '__main__':
    main()
