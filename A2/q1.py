import matplotlib.pyplot as plt


def bias(var_lambda, var_mu):
    return ((var_lambda / (var_lambda + 1)) * var_mu) ** 2


def variance(var_sigma, var_n, var_lambda):
    return (var_sigma ** 2) / (var_n * ((var_lambda + 1) ** 2))


if __name__ == "__main__":
    mu = 1
    sigma = 3
    n = 10
    x = [i for i in range(0, 11)]
    biases = [bias(i, mu) for i in range(0, 11)]
    variances = [variance(sigma, n, i) for i in range(0, 11)]
    exp_err = [bias(i, mu)+variance(sigma, n, i) for i in range(0, 11)]
    fig, ax = plt.subplots()
    ax.scatter(x, biases, label='bias')
    ax.plot(x, biases)
    ax.scatter(x, variances, label='variance')
    ax.plot(x, variances)
    ax.scatter(x, exp_err, label='expected squared error')
    ax.plot(x, exp_err)
    plt.xlabel('lambda')
    plt.ylabel('value')
    plt.title('mu='+str(mu)+', sigma='+str(sigma)+', n='+str(n))
    ax.legend()
    plt.show()