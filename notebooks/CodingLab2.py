#!/usr/bin/env python
# coding: utf-8

# _Neural Data Science_
# 
# Lecturer: Prof. Dr. Philipp Berens
# 
# Tutors: Jonas Beck, Ziwei Huang, Rita González Márquez
# 
# Summer term 2023
# 
# Student names: *FILL IN YOUR NAMES HERE*
# 
# # Coding Lab 2
# 
# - __Data__: Use the saved data `nds_cl_1_*.npy` from Coding Lab 1. Or, if needed, download the data files ```nds_cl_1_*.npy``` from ILIAS and save it in the subfolder ```../data/```.
# - __Dependencies__: You don't have to use the exact versions of all the dependencies in this notebook, as long as they are new enough. But if you run "Run All" in Jupyter and the boilerplate code breaks, you probably need to upgrade them.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from IPython import embed
from sklearn.cluster import KMeans

# from __future__ import annotations

# get_ipython().run_line_magic('load_ext', 'jupyter_black')

# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '--time --date --timezone --updated --python --iversions --watermark -p sklearn')


# In[2]:


#plt.style.use("../matplotlib_style.txt")


# ## Load data

# In[3]:


# replace by path to your solutions
b = np.load("../data/nds_cl_1_features.npy")
s = np.load("../data/nds_cl_1_spiketimes_s.npy")
w = np.load("../data/nds_cl_1_waveforms.npy")
np.random.seed(0)


# ## Task 1: Generate toy data
# 
# Sample 1000 data points from a two dimensional mixture of Gaussian model with three clusters  and the following parameters:
# 
# $\mu_1 = \begin{bmatrix}0\\0\end{bmatrix}, \Sigma_1 = \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}, \pi_1=0.3$
# 
# $\mu_2 = \begin{bmatrix}5\\1\end{bmatrix}, \Sigma_2 = \begin{bmatrix}2 & 1\\1 & 2\end{bmatrix}, \pi_2=0.5$
# 
# $\mu_3 = \begin{bmatrix}0\\4\end{bmatrix}, \Sigma_3 = \begin{bmatrix}1 & -0.5\\-0.5 & 1\end{bmatrix}, \pi_3=0.2$
# 
# Plot the sampled data points and indicate in color the cluster each point came from. Plot the cluster means as well.
# 
# *Grading: 1 pts*
# 

# In[4]:


def sample_data(
    N: int, m: np.ndarray, S: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Generate N samples from a Mixture of Gaussian distribution with
    means m, covariances S and priors p.

    Parameters
    ----------

    N: int
        Number of samples

    m: np.ndarray, (n_clusters, n_dims)
        Means

    S: np.ndarray, (n_clusters, n_dims, n_dims)
        Covariances

    p: np.ndarray, (n_clusters, )
        Cluster weights / probablities

    Returns
    -------

    labels: np.array, (n_samples, )
        Grund truth labels.

    x: np.array, (n_samples, n_dims)
        Data points
    """

    # insert your code here
    n_labels = np.arange(0, len(m[:, 0]))

    x = np.zeros((N, len(m[0])))
    labels = np.zeros(N)
    for i in range(N):
        # create for one sample three cluster with the probalility
        k = np.random.choice(n_labels, p=p)
        # sample from a multivarian distribution 
        x[i, :] =  np.random.multivariate_normal(m[k], S[k])
        labels[i] = k

    return labels, x


# In[5]:


N = 1000  # total number of samples

p = np.array([0.3, 0.5, 0.2])  # percentage of each cluster
m = np.array([[0.0, 0.0], [5.0, 1.0], [0.0, 4.0]])  # means

S1 = np.array([[1.0, 0.0], [0.0, 1.0]])
S2 = np.array([[2.0, 1.0], [1.0, 2.0]])
S3 = np.array([[1.0, -0.5], [-0.5, 1.0]])
S = np.stack([S1, S2, S3])  # cov

labels, x = sample_data(N, m, S, p)


# In[ ]:


# ----------------------------------------------
# plot points from mixture of Gaussians (0.5 pt)
# ----------------------------------------------

fig, ax = plt.subplots(figsize=(5, 5))


# ## Task 2: Implement a Gaussian mixture model
# 
# Implement the EM algorithm to fit a Gaussian mixture model in `fit_mog()`.  Sort the data points by inferring their class labels from your mixture model (by using maximum a-posteriori classification). Fix the seed of the random number generator to ensure deterministic and reproducible behavior. Test it on the toy dataset specifying the correct number of clusters and make sure the code works correctly. Plot the data points from the toy dataset and indicate in color the cluster each point was assigned to by your model. How does the assignment compare to ground truth? If you run the algorithm multiple times, you will notice that some solutions provide suboptimal clustering solutions - depending on your initialization strategy.  
# 
# *Grading: 4 pts*
# 

# In[7]:


def fit_mog(
    x: np.ndarray, n_clusters: int, niters: int = 10, random_seed: int = 2046
) -> tuple[np.ndarray]:
    """Fit Mixture of Gaussian model using EM algo.

    Parameters
    ----------

    x: np.array, (n_samples, n_dims)
        Input data

    n_clusters: int
        Number of clusters

    niters: int
        Maximal number of iterations.

    random_seed: int
        Random Seed


    Returns
    -------

    labels: np.array, (n_samples)
        Cluster labels

    m: list or np.array, (n_clusters, n_dims)
        Means

    S: list or np.array, (n_clusters, n_dims, n_dims)
        Covariances

    p: list or np.array, (n_clusters, )
        Cluster weights / probablities
    """

    # fill in your code here

    np.random.seed(random_seed)
    n, _ = np.shape(x)

    ### Initialize the gausian mixture model with kmeans as first guess
    kmeans = KMeans(n_clusters=n_clusters).fit(x)
    # Intial gues from the kmeans for the centers or means! 
    kmean_centers = kmeans.cluster_centers_
    # Inital mixing coefficiant of the all cluster in n_clusters
    pi = np.ones(n_clusters) / n_clusters
    # Initial covariance matrix, identity marix with n_clusters rows 
    cov = [np.eye(x.shape[1]) for _ in range(n_clusters)]

    for step in range(niters):

        # E step calculation of gamma
        # creating empty list with size of n_samples and n_clusters. ex (1000, 3) 1000 samples and 3 clusters
        gamma_cluster = np.zeros((n, n_clusters))
        # go through n_cluster to calulate the upper term of the gamma:
        for cluster in range(n_clusters):
            gamma_cluster[:, cluster]  = pi[cluster] * sp.stats.multivariate_normal.pdf(x, mean=kmean_centers[cluster], cov=cov[cluster])
        # creating the sum of gamma over all n_clusters
        gamma_all_clusters = np.sum(gamma_cluster, axis=1)
        
        gamma = gamma_cluster / gamma_all_clusters[:, np.newaxis]
        
        # M step updating the mean, pi, cov with gamma 




    ## create and initialize the cluster centers and the weight paramters
    #weights = np.ones(k) / k
    #means = np.random.choice(x.flatten(), (k, d))
    #
    ## covariance matix guess
    #cov = [np.eye(d) for _ in range(k)]
    ## -------------------------
    ## EM maximisation (2.5 pts)
    ## -------------------------
    #
    #for step in range(niters):
    #    probs = np.zeros((n, k))
    #    for cluster in range(k):
    #        probs[:, cluster] = weights[cluster] * sp.stats.multivariate_normal.pdf(x, mean=means[cluster], cov=cov[cluster])
    #    #probs /= probs.sum(axis=1, keepdims=True)
    #    
    #    for cluster in range(k):  
    #        yamma = probs[cluster]* weights[cluster]  / np.sum([probs[i] * weights[i] for i in range(k)], axis=0)

    embed()
    exit()
    #continue
        # E step
        # Evaluate the posterior probablibities `r`
        # using the current values of `m` and `S`

        # M step
        # Estimate new `m`, `S` and `p`

    pass


# Run Mixture of Gaussian on toy data

# In[8]:


mog_labels, m, S, p = fit_mog(x, 3, random_seed=0)


# Plot toy data with cluster assignments and compare to original labels

# In[ ]:


mosaic = [["True", "MoG"]]
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(8, 4), layout="constrained")

# -----------------
# Add plot (0.5 pts)
# -----------------


# ## Task 3: Model complexity
# A priori we do not know how many neurons we recorded. Extend your algorithm with an automatic procedure to select the appropriate number of mixture components (clusters). Base your decision on the Bayesian Information Criterion:
# 
# $BIC = -2L+P \log N,$
# 
# where $L$ is the log-likelihood of the data under the best model, $P$ is the number of parameters of the model and $N$ is the number of data points. You want to minimize the quantity. Plot the BIC as a function of mixture components. What is the optimal number of clusters on the toy dataset?
# 
# You can also use the BIC to make your algorithm robust against suboptimal solutions due to local minima. Start the algorithm multiple times and pick the best solutions for extra points. You will notice that this depends a lot on which initialization strategy you use.
# 
# *Grading: 3 pts*
# 
# 

# In[10]:


def mog_bic(
    x: np.ndarray, m: np.ndarray, S: np.ndarray, p: np.ndarray
) -> tuple[float, float]:
    """Compute the BIC for a fitted Mixture of Gaussian model

    Parameters
    ----------

    x: np.array, (n_samples, n_dims)
        Input data

    m: np.array, (n_clusters, n_dims)
        Means

    S: np.array, (n_clusters, n_dims, n_dims)
        Covariances

    p: np.array, (n_clusters, )
        Cluster weights / probablities

    Return
    ------

    bic: float
        BIC

    LL: float
        Log Likelihood
    """

    # insert your code here

    # -------------------------
    # implement the BIC (1.5 pts)
    # -------------------------

    pass


# In[11]:


# ---------------------------------------------------------------------------------------------------
# Compute and plot the BIC for mixture models with different numbers of clusters (e.g., 2 - 6). (0.5 pts)
# Make your algorithm robust against local minima. (0.5 pts) and plot the result (0.5 pts)
# ---------------------------------------------------------------------------------------------------

K = range(2, 7)
num_seeds = 10

BIC = np.zeros((num_seeds, len(K)))
LL = np.zeros((num_seeds, len(K)))

# run mog and BIC multiple times here


# In[ ]:


fig, ax = plt.subplots(figsize=(4, 4))
# plot BIC


# ## Task 4: Spike sorting using Mixture of Gaussian 
# Run the full algorithm on your set of extracted features (including model complexity selection). Plot the BIC as a function of the number of mixture components on the real data. For the best model, make scatter plots of the first PCs on all four channels (6 plots). Color-code each data point according to its class label in the model with the optimal number of clusters. In addition, indicate the position (mean) of the clusters in your plot. 
# 
# *Grading: 3 pts*
# 

# In[13]:


# ------------------------------------------------------------------------------------------
# Select the model that best represents the data according to the BIC (include plot) (1 pt)
# ------------------------------------------------------------------------------------------

K = np.arange(2, 16)
num_seeds = 5

BIC = np.zeros((num_seeds, len(K)))
LL = np.zeros((num_seeds, len(K)))


# In[ ]:


fig, ax = plt.subplots(figsize=(4, 4))

# plot BIC


# Refit model with lowest BIC and plot data points

# In[15]:


random_seed, kk = np.where(BIC == BIC.min())
random_seed = random_seed[0]
kk = kk[0]
print(f"lowest BIC: # cluster = {K[kk]}")
# a, m, S, p = fit_mog(b, K[kk], random_seed=random_seed)


# In[ ]:


# -------------------------------------------------------------------------------------
# Create scatterplots of the first PCs under the best model for all 4 channels. (2 pts)
# -------------------------------------------------------------------------------------


mosaic = [
    ["Ch2 vs Ch1", ".", "."],
    ["Ch3 vs Ch1", "Ch3 vs Ch2", "."],
    ["Ch4 vs Ch1", "Ch4 vs Ch2", "Ch4 vs Ch3"],
]
fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(8, 8), layout="constrained")


# ### Task 5: Cluster separation
# 
# Implement linear discriminant analysis to visualize how well each cluster is separated from its neighbors in the high-dimensional space in the function `separation()`. Project the spikes of each pair of clusters onto the axis that optimally separates those two clusters. 
# 
# Plot a matrix with pairwise separation plots, showing the histogram of the points in both clusters projected on the axis best separating the clusters (as shown in the lecture). *Hint:* Since Python 3.5+, matrix multiplications can be compactely written as `x@y`.
# 
# *Grading: 4 pts*
# 

# In[17]:


def separation(
    b: np.ndarray,
    m: np.ndarray,
    S: np.ndarray,
    p: np.ndarray,
    assignment: np.ndarray,
    nbins: int = 50,
):
    """Calculate cluster separation by LDA.

    proj, bins = separation(b, m, S, p, assignment)
    projects the data on the LDA axis for all pairs of clusters. The result
    is normalized such that the left (i.e. first) cluster has
    zero mean and unit variances. The LDA axis is estimated from the model.
    ---

    Parameters
    ----------
    b: np.array, (n_spikes, n_features)
        Features.

    m: np.array, (n_clusters, n_features)
        Means.

    S: np.array, (n_clusters, n_features, n_features)
        Covariance.

    p: np.array, (n_clusters, )
        Cluster weight.

    assignment: np.array, (n_spikes, )
        Cluster assignments / labels for each spike

    nbins: int
        Number of bins in a lda histogram.


    Returns
    -------

    proj: np.array, (n_bins, n_clusters, n_clusters)
        computed lda histo# Comparing the cells in particular

    bins: np.array, (n_bins)
        bin times relative to center    #bins x 1
    """

    # insert your code here

    # ---------------------------------------------------------------------
    # compute the optimal separating axes for each pair of clusters (2 pts)
    # ---------------------------------------------------------------------

    # -------------------------------------------
    # normalise according to first cluster (1 pt)
    # -------------------------------------------

    # --------------------------------------
    # plot histograms on optimal axis (1 pt)
    # --------------------------------------

    return proj, bins


# In[18]:


# proj, bins = separation(b, m, S, p, a)

