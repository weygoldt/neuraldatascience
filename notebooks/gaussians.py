import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from IPython import embed
from sklearn.cluster import KMeans

# Generate data
N = 1000  # total number of samples

weights = np.array([0.3, 0.5, 0.2])  # percentage of each cluster
means = np.array([[0.0, 0.0], [5.0, 1.0], [0.0, 4.0]])  # means

S1 = np.array([[1.0, 0.0], [0.0, 1.0]])
S2 = np.array([[2.0, 1.0], [1.0, 2.0]])
S3 = np.array([[1.0, -0.5], [-0.5, 1.0]])
covs = np.stack([S1, S2, S3])  # cov


# Define the pdf of the mixture model
def pdf(x):
    return np.sum(
        [
            weights[k] * sp.stats.multivariate_normal.pdf(x, means[k], covs[k])
            for k in range(len(means))
        ],
        axis=0,
    )


# Generate random samples from the mixture model
values = np.zeros((N, 2))
labels = np.zeros(N)
for i in range(N):
    # Select from which gaussian to sample
    k = np.random.choice(len(means), p=weights)
    # sample from the selected gaussian
    values[i] = np.random.multivariate_normal(means[k], covs[k], size=1)
    # append cluster label
    labels[i] = k

# plot random samples
for label in labels:
    plt.scatter(
        values[labels == label, 0], values[labels == label, 1], label=label
    )
plt.show()

# plot the pdf
x = np.linspace(np.min(values[:, 0]), np.max(values[:, 0]), 100)
y = np.linspace(np.min(values[:, 1]), np.max(values[:, 1]), 100)
xx, yy = np.meshgrid(x, y)
zz = pdf(np.c_[xx.ravel(), yy.ravel()])
plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=plt.cm.Blues)
plt.show()
embed()
exit()


# Generate random samples from the mixture model
X_new = np.zeros((n_samples, 1))
for i in range(n_samples):
    # Select a component of the mixture model
    k = np.random.choice(n_components, p=weights)
    # Generate a random value from the selected component
    X_new[i] = np.random.normal(means[k], np.sqrt(covs[k]), size=1)

# Plot results
x_plot = np.linspace(-6, 10, 1000).reshape(-1, 1)
plt.plot(x_plot, pdf(x_plot), color="black", linewidth=2, label="True pdf")
plt.hist(
    X, bins=50, density=True, color="navy", alpha=0.5, label="Original data"
)
plt.hist(
    X_new,
    bins=50,
    density=True,
    color="orange",
    alpha=0.5,
    label="Generated data",
)
plt.legend(loc="upper left")
plt.show()
