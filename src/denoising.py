""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import medmnist
from matplotlib.backends.backend_pdf import PdfPages


RANDOM_SEED = 41


def task2():
    """ Bayesian Denoising - 2D Toytask

        Requirements for the plots:
        - ax[0] should contain training data, the test data point, and the conditional mean/MAP using a Dirac prior.
        - ax[1-3] should contain training data, the test data point, the estimated pdf, and the conditional mean/MAP using the KDE prior for 3 different bandwidths h. 
    """
    fig, ax = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle('Task 2 - Bayesian Denoising of 2D Toytask', fontsize=16)
    ax[0].set_title(r'Dirac Prior')

    """ Start of your code
    """

    N = 900
    n = int(N / 3)

    Y = __create_clean_data(n)
    bandwidths = [0.1, 0.25, 1]  # [0.1, 0.25, 0.5]
    grid_positions, xx, yy = __create_2d_grid()
    kde_results = []
    x = __select_out_of_distribution_test_dataset(Y)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    for i, h in enumerate(bandwidths):
        ax[i].plot(x[0], x[1], "x", color="black", markersize="8")
        ax[i].plot(Y[:n, 0], Y[:n, 1], 'o', markersize="2", color="firebrick")
        ax[i].plot(Y[n:2*n, 0], Y[n:2*n, 1], 'o', markersize="2", color="lightgreen")
        ax[i].plot(Y[2*n:, 0], Y[2*n:, 1], 'o', markersize="2", color="blue")
        kde_results.append(__calculate_kde(grid_positions, Y, h))

        Y_expected = np.mean(Y, axis=0)
        ax[i].plot(Y_expected[0], Y_expected[1], 'H', markersize="8", color="purple")

        # # conditional mean
        # likelihood = __calculate_likelihood_of_2d_gaussian(x, Y, h).reshape((Y.shape[0], -1))
        # prior = __calculate_kde(Y, Y, h)
        # cond_mean = np.sum(Y * prior * likelihood, axis=0) / np.sum(prior * likelihood, axis=0)
        #
        # # map
        # map = np.argmax(prior * likelihood, axis=0)
        #
        # # cond_mean = -0.43629,0.09552
        # # Y[108] = [-0.64147761  0.14104598]; Y[257] = [-0.93260094  0.16206101]

        ax[i].contourf(xx, yy, kde_results[i], cmap='terrain')
        ax[i].contour(xx, yy, kde_results[i], colors='gold')
        ax[i].set_title(r'KDE Prior $h=$' + str(h))
    # plt.show()

    # conditional mean
    likelihood = __calculate_likelihood_of_2d_gaussian(x, Y, bandwidths[2]).reshape((Y.shape[0], -1))
    prior = __calculate_kde(Y, Y, h)
    cond_mean = np.sum(Y * prior * likelihood, axis=0) / np.sum(prior * likelihood, axis=0)
    ax[2].plot(cond_mean[0], cond_mean[1], "*", color="black", markersize="8")
    print('conditional mean: ', cond_mean[0], cond_mean[1])

    # map
    map = np.argmax(prior * likelihood, axis=0)
    ax[2].plot(Y[map[0], 0], Y[map[1], 1], "+", color="black", markersize="8")
    print('map: ', Y[map[0], 0], Y[map[1], 1])

    # cond_mean = -0.43629,0.09552
    # Y[108] = [-0.64147761  0.14104598]; Y[257] = [-0.93260094  0.16206101]

    plt.show()

    """ End of your code
    """

    return fig


# def __calculate_conditional_mean(Y, bandwidths, h, x):
#     likelihood = __calculate_likelihood_of_2d_gaussian(x, Y, bandwidths[1]).reshape((Y.shape[0], -1))
#     prior = __calculate_kde(Y, Y, h)
#     cond_mean = np.sum(Y * prior * likelihood) / np.sum(prior * likelihood)
#     return cond_mean


def __select_out_of_distribution_test_dataset(Y):
    # x_dimensions = Y[:, 0]
    # outlier_indices = np.argmin(x_dimensions)
    # return Y[outlier_indices]

    return [2, -0.5]

    # return x.reshape((x.shape[0], -1))


def __create_2d_grid():
    num_x = 100
    num_y = 100
    x = np.linspace(-1, 2.5, num_x)
    y = np.linspace(-1, 2.5, num_y)
    xx, yy = np.meshgrid(x, y)
    grid_positions = np.concatenate((xx.reshape((num_x, num_y, 1)), yy.reshape((num_x, num_y, 1))), axis=2)

    return grid_positions, xx, yy


def __create_clean_data(n):
    mu1 = [0.0, 0.0]
    mu2 = [0.0, 1.5]
    mu3 = [1.5, 1.5]
    sigma = np.full((2, 2), [[0.075, 0], [0, 0.075]])

    Y1 = np.random.multivariate_normal(mu1, sigma, n)
    Y2 = np.random.multivariate_normal(mu2, sigma, n)
    Y3 = np.random.multivariate_normal(mu3, sigma, n)

    return np.vstack((Y1, Y2, Y3))


def logsumexp_stable(x):
    xmax = np.max(x)
    return xmax + np.log(np.sum(np.exp(x-xmax)))


def __calculate_kde(y, Y, h):
    kde = np.ndarray((y.shape[0], y.shape[1]))

    for i in np.arange(y.shape[0]):
        for j in np.arange(y.shape[1]):
            kde[i, j] = 1 / Y.shape[0] * np.sum(__calculate_likelihood_of_2d_gaussian(y[i, j], Y, h))  # use KDE based on Gaussian kernel to
            # create PDF

    return kde


def __calculate_likelihood_of_2d_gaussian(y, Y, h):
    return 1 / (2 * np.pi * h ** 2) * np.exp(- np.linalg.norm(y - Y, axis=1, ord=2) ** 2 / (2 * h ** 2))


def task3():
    """ Bayesian Image Denoising

        Requirements for the plots:
        - the first row should show your results for \sigma^2=0.1
        - the second row should show your results for \sigma^2=1.
        - arange your K images as a grid
    """

    fig, ax = plt.subplots(2, 4, figsize=(15,8))
    fig.suptitle('Task 3 - Bayesian Image Denoising', fontsize=16)

    ax[0,0].title.set_text(r'$\mathbf{y}^*$')
    ax[0,1].title.set_text(r'$\mathbf{x}$')
    ax[0,2].title.set_text(r'$\mathbf{\hat y}_{\operatorname{CM}}(\mathbf{x})$')
    ax[0,3].title.set_text(r'$\mathbf{\hat y}_{\operatorname{MAP}}(\mathbf{x})$')
    ax[0,0].set_ylabel(r'$\sigma^2=0.1$')
    ax[1,0].set_ylabel(r'$\sigma^2=1.$')

    for a in ax.reshape(-1):
        a.set_xticks([])
        a.set_yticks([])

    """ Start of your code
    """
    
    """ End of your code
    """
    return fig


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    tasks = [task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
