import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from IPython import embed

# Generate data
np.random.seed(0)
n_samples = 500
X = np.vstack([np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(0.6 * n_samples)),
               np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], int(0.4 * n_samples))])

# Initialize the parameters of the mixture model
n_components = 2
weights = np.ones(n_components) / n_components
means = np.random.randn(n_components, 2) * 2
covs = np.tile(np.eye(2), (n_components, 1, 1))

embed()
exit()

# Define the pdf of the mixture model
def pdf(x):
    return np.sum([weights[k] * multivariate_normal.pdf(x, means[k], covs[k]) for k in range(n_components)], axis=0)

# Generate random samples from the mixture model
X_new = np.zeros((n_samples, 2))
for i in range(n_samples):
    # Select a component of the mixture model
    k = np.random.choice(n_components, p=weights)
    # Generate a random value from the selected component
    X_new[i] = np.random.multivariate_normal(means[k], covs[k], size=1)

# Plot results
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = pdf(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues)
plt.scatter(X[:, 0], X[:, 1], s=30, color='navy', label='Original data')
plt.scatter(X_new[:, 0], X_new[:, 1], s=30, color='orange', label='Generated data')
plt.legend(loc='upper left')
plt.show()
