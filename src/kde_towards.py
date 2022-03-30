import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.datasets import make_blobs


if __name__ == '__main__':
    # Create 2d data from 3 distributions
    n_components = 3
    X, truth = make_blobs(n_samples=300, centers=n_components,
                          cluster_std=[2, 1.5, 1],
                          random_state=42)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=truth)
    plt.title(f"Example of a mixture of {n_components} distributions")
    plt.xlabel("x")
    plt.ylabel("y")

    # Extract x and y
    x = X[:, 0]
    y = X[:, 1]

    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    print(xmin, xmax, ymin, ymax)

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # Fit Gaussian kernel using scipy
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    # Plot KDE in contourf plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    # ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Gaussian Kernel density estimation')

    plt.figure(figsize=(8, 8))
    for j in range(len(cset.allsegs)):
        for ii, seg in enumerate(cset.allsegs[j]):
            plt.plot(seg[:, 0], seg[:, 1], '.-', label=f'Cluster{j}, level{ii}')
    plt.legend()

    plt.show()

    print('x')
