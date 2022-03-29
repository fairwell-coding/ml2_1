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

    mu1 = [0.0, 0.0]
    mu2 = [0.0, 1.5]
    mu3 = [1.5, 1.5]
    sigma = np.full((2, 2), [[0.075, 0], [0, 0.075]])

    Y1 = np.random.multivariate_normal(mu1, sigma, n)
    Y2 = np.random.multivariate_normal(mu2, sigma, n)
    Y3 = np.random.multivariate_normal(mu3, sigma, n)

    Y = np.vstack((Y1, Y2, Y3))

    h1 = 1e-2
    h2 = 1e-1
    h3 = 4e-1

    # y = np.linspace(-2., 2., N)

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(x, y)

    # plot with different bandwidths h
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    for i, h in enumerate([h1, h2, h3]):
        ax[i].plot(Y[:, 0], Y[:, 1], 'o')
        ax[i].contourf(xx, yy, __calculate_kde(yy, Y, h), cmap='gist_rainbow')
        ax[i].set_title('bandwidth h=%.2f' % h)
    plt.show()

    # ax[1].contourf(x, y, , cmap='gist_rainbow')

    """ End of your code
    """

    # ax[1].set_title(r'KDE Prior $h=$'+str(h1))
    # ax[2].set_title(r'KDE Prior $h=$'+str(h2))
    # ax[3].set_title(r'KDE Prior $h=$'+str(h3))
    # for a in ax.reshape(-1):
    #     a.legend()

    return fig


def __calculate_kde(y, Y, h):
    kde = np.zeros_like(y)

    for i in np.arange(y.shape[0]):
        for j in np.arange(y.shape[1]):
            kde[i, j] = 1 / Y.shape[0] * np.sum(1 / (2 * np.pi * h ** 2) * np.exp(- (y[i, j] - Y) ** 2 / (2 * h ** 2)))  # use KDE based on Gaussian kernel to create PDF

    return kde


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
    tasks = [task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
