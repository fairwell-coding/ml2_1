""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from medmnist import ChestMNIST


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
    bandwidths = [0, 0.1, 0.25, 0.5]  # [0.1, 0.2, 0.5] [0.1, 0.25, 0.5]
    grid_positions, xx, yy = __create_2d_grid()
    kde_results = []
    x = __select_out_of_distribution_test_dataset(Y)

    for i, h in enumerate(bandwidths):
        ax[i].plot(x[0], x[1], "X", color="black", markersize="7", label=r'$X_{outlier}$')
        ax[i].plot(Y[:n, 0], Y[:n, 1], 'o', markersize="2", color="firebrick")
        ax[i].plot(Y[n:2*n, 0], Y[n:2*n, 1], 'o', markersize="2", color="lightgreen")
        ax[i].plot(Y[2*n:, 0], Y[2*n:, 1], 'o', markersize="2", color="blue")

        Y_expected = np.mean(Y, axis=0)
        ax[i].plot(Y_expected[0], Y_expected[1], 'H', markersize="7", color="purple", label=r'$\bar{Y}_{global}$')
        if i == 0:
            continue
        kde_results.append(__calculate_kde(grid_positions, Y, h))
        ax[i].contourf(xx, yy, kde_results[i-1], levels=10, cmap='terrain')
        ax[i].contour(xx, yy, kde_results[i-1], levels=10, colors='gold')
        ax[i].set_title(r'KDE Prior $h=$' + str(h))

    prior = np.zeros_like(Y)
    for i in [0, 1, 2, 3]:
        h = 1
        likelihood = __calculate_likelihood_of_2d_gaussian(x, Y, h).reshape((Y.shape[0], -1))
        if i == 0:
            prior = [1/N, 1/N]  # dirac measure
        else: 
            prior = __calculate_kde(Y, Y, bandwidths[i])
    
        # conditional mean
        cond_mean = np.sum(Y * prior * likelihood, axis=0) / np.sum(prior * likelihood, axis=0)
        ax[i].plot(cond_mean[0], cond_mean[1], "*", color="black", markersize="7", label=r'$\hat{y}_{CM}$')
        print('* conditional mean: ', cond_mean[0], cond_mean[1])

        # map
        map = np.argmax(prior * likelihood)
        ax[i].plot(Y[map][0], Y[map][1], "P", color="black", markersize="7", label=r'$\hat{y}_{MAP}$')
        print('+ map: ', Y[map][0], Y[map][1])
        ax[i].legend()
    plt.show()

    """ End of your code
    """

    return fig


def __select_out_of_distribution_test_dataset(Y):

    #Test points
    #x = [1, 0.57]
    #x = [1, 0.77]
    #x = [1, 0.37]
    #x = [1, 0]
    #x = [0.9, 0.87]
    #x = [1, 0.87]
    #x = [0.8, -0.1]
    #x = [1, 2.5]
    #x = [1.5, 0]

    return [0.74, 0.74]

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
    """ Calculate likelihood of 2D Gaussian needed for task 2.
    """
    return 1 / (2 * np.pi * h ** 2) * np.exp(- np.linalg.norm(y - Y, axis=1, ord=2) ** 2 / (2 * h ** 2))  # equation (7)


def __calculate_log_kde(Y, h):
    kde = np.ndarray((Y.shape[0], Y.shape[1], Y.shape[2]))

    for image in np.arange(Y.shape[0]):
        for i in np.arange(Y.shape[1]):
            for j in np.arange(Y.shape[2]):
                kde[image, i, j] = __kde_using_log(Y[image, i, j], Y[image], h)  # use log KDE based on Gaussian kernel to create PDF

    return kde


def __kde_using_log(y, Y, h):
    """ Calculate KDE prior for 2D Gaussian using log of prior and likelihood needed for task 3 (see equation (10) from assignment sheet). Since D = 2 in the assignment the implemented formula here
    already omits D and uses the resulting slightly simplified formula.
    """
    N = Y.shape[0]
    D = Y.shape[0] * Y.shape[1]

    return logsumexp_stable(- np.log(N) - D / 2 * np.log(2 * np.pi) - D * np.log(h) - np.linalg.norm(y - Y) / (2 * h ** 2))


def __calculate_log_likelihood_for_all_test_images(X, Y, deviation):
    log_likelihoods = np.ndarray((X.shape[0], Y.shape[1], Y.shape[2]))

    for image in np.arange(X.shape[0]):
        log_likelihoods[image, :, :] = __calculate_log_likelihood_of_gaussian(X[image], Y, deviation)  # use log likelihoods

    return log_likelihoods


def __calculate_log_likelihood_of_gaussian(y, Y, deviation):
    """ Calculate likelihood of 2D Gaussian needed for task 3.
    """

    D = Y.shape[0] * Y.shape[1]

    return - D / 2 * np.log(2 * np.pi * deviation) - np.linalg.norm(Y - y, axis=0) ** 2 / (2 * deviation)


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

    # Load ChestMNIST dataset
    train_data = ChestMNIST('train', download=True).imgs
    test_data = ChestMNIST('test', download=True).imgs

    # Normalize images
    train_data = train_data / 255
    test_data = test_data / 255

    # Randomly select N = 1000 training images for Y (uniform sampling)
    train_indices = np.random.choice(len(train_data), 1000)
    Y = train_data[train_indices]

    # Randomly select M = 25 test images for X (uniform sampling)
    test_indices = np.random.choice(len(test_data), 25)
    X_clean = test_data[test_indices]

    deviation_1 = 0.1  # used for both Gaussian noise and KDE prior
    deviation_2 = 1  # used for both Gaussian noise and KDE prior
    X_1 = __create_noisy_test_data(X_clean, 0.0, deviation_1)  # noisy test data with deviation = 0.1
    X_2 = __create_noisy_test_data(X_clean, 0.0, deviation_2)  # noisy test data with deviation = 1

    # KDE
    prior_1 = __calculate_log_kde(Y, deviation_1)
    prior_2 = __calculate_log_kde(Y, deviation_2)

    # Plot clean data, i.e. y*
    ax[0, 0].imshow(__reshape_containing_all_subimages(X_clean))  # (25, 28, 28) -->> (5, 5, 28, 28) -->> (5 * 28, 5 * 28)
    ax[1, 0].imshow(__reshape_containing_all_subimages(X_clean))  # (25, 28, 28) -->> (5, 5, 28, 28) -->> (5 * 28, 5 * 28)

    # Plot noisy data, i.e. X
    ax[0, 1].imshow(__reshape_containing_all_subimages(X_1))
    ax[1, 1].imshow(__reshape_containing_all_subimages(X_2))

    # Choose suitable deviation for test samples
    test_deviation_1 = 0.2
    test_deviation_2 = 0.1

    # calculate log likelihood
    likelihood_1 = __calculate_log_likelihood_for_all_test_images(X_1, Y, test_deviation_1)
    likelihood_2 = __calculate_log_likelihood_for_all_test_images(X_2, Y, test_deviation_2)

    # prior_1 * likelihood_1[0]
    # prior_1.reshape((1000, 1, 28, 28)) * likelihood_1.reshape((1, 25, 28, 28)) -->> 1000x25x28x28

    # conditional mean
    Y_ = Y.reshape(1000, 1, 28, 28)
    prior_1_ = prior_1.reshape((1000, 1, 28, 28))
    likelihood_1_ = likelihood_1.reshape((1, 25, 28, 28))
    prior_2_ = prior_2.reshape((1000, 1, 28, 28))
    likelihood_2_ = likelihood_2.reshape((1, 25, 28, 28))

    cond_mean_1 = np.sum(Y_ * prior_1_ * likelihood_1_, axis=0) / np.sum(prior_1_ * likelihood_1_, axis=0)
    cond_mean_2 = np.sum(Y_ * prior_2_ * likelihood_2_, axis=0) / np.sum(prior_2_ * likelihood_2_, axis=0)

    ax[0, 2].imshow(__reshape_containing_all_subimages(cond_mean_1))
    ax[1, 2].imshow(__reshape_containing_all_subimages(cond_mean_2))

    # MAP
    map1 = np.argmax(prior_1_ * likelihood_1_, axis=0)
    map2 = np.argmax(prior_2_ * likelihood_2_, axis=0)

    ax[0, 3].imshow(__reshape_containing_all_subimages(__get_argmax_pixel_values_from_training_samples(X_clean, Y, map1)))
    ax[1, 3].imshow(__reshape_containing_all_subimages(__get_argmax_pixel_values_from_training_samples(X_clean, Y, map2)))

    plt.show()

    """ End of your code
    """
    return fig


def __get_argmax_pixel_values_from_training_samples(X, Y, map):
    return Y.flatten()[map.flatten()].reshape((X.shape[0], X.shape[1], X.shape[2]))


def __create_noisy_test_data(X_clean, mu, deviation):
    noise_x_flattened = np.random.normal(mu, deviation, X_clean.shape[0] * X_clean.shape[1] * X_clean.shape[2])  # flattened noise for all images and both dimensions
    test_noise = noise_x_flattened.reshape((X_clean.shape[0], X_clean.shape[1], X_clean.shape[2]))
    return X_clean + test_noise


def __reshape_containing_all_subimages(x):
    img_matrix = np.zeros((5 * 28, 5 * 28))

    img_size = 28
    row = 0
    last_index = 0
    for i in range(5):
        if i == 0:
            img_matrix[0:img_size, :] = np.hstack(x[0:5, :])  # X_clean[i:(1+i)]
            row = img_size
            last_index = 5
        else:
            img_matrix[row:row + img_size, :] = np.hstack(x[last_index:last_index + 5, :])  # X_clean[i:(1+i)]
            last_index = last_index + 5
            row = row + img_size

    return img_matrix


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    # tasks = [task2, task3]
    tasks = [task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
