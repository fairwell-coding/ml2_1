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

    mu1 = [0.0, 0.0]
    mu2 = [0.0, 1.5]
    mu3 = [1.5, 1.5]
    sigma = np.full((2, 2), [[0.075, 0], [0, 0.075]])

    y1 = np.random.multivariate_normal(mu1, sigma, int(N / 3))
    y2 = np.random.multivariate_normal(mu2, sigma, int(N / 3))
    y3 = np.random.multivariate_normal(mu3, sigma, int(N / 3))

    h1 = 1
    h2 = 1
    h3 = 1

    kde1 = __calculate_kde(N, h1, 0, y1)
    kde2 = __calculate_kde(N, h2, 0, y2)
    kde3 = __calculate_kde(N, h3, 0, y3)


    """ End of your code
    """

    ax[1].set_title(r'KDE Prior $h=$'+str(h1))
    ax[2].set_title(r'KDE Prior $h=$'+str(h2))
    ax[3].set_title(r'KDE Prior $h=$'+str(h3))
    for a in ax.reshape(-1):
        a.legend()

    return fig


def __calculate_kde(N, h, mu, y):
    # kd = 1 / N * np.sum(1 / (2 * np.pi * h ** 2) * np.exp(- (y - mu) ** 2 / (2 * h ** 2)))
    kde = 1 / (2 * np.pi * h ** 2) * np.exp(- (y - mu) ** 2 / (2 * h ** 2))
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
