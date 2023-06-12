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
# Names: FILL IN YOUR NAMES HERE
#
# # Coding Lab 6
#
# In this exercise we are going to fit a latent variable model (Poisson GPFA) to both toy data and real data from monkey primary visual cortex.

# ## Preliminaries
#
# ### 1. Code
#
# The toolbox we are going to use contains an implementation of the EM algorithm to fit the poisson-gpfa.
#
# Assuming you `git clone https://github.com/mackelab/poisson-gpfa` to the notebooks/ directory and have the following directory structure:
#
#
# ```
# ├── data/
# │   └── nds_cl_6_data.mat
# ├── notebooks
# │   ├── poisson-gpfa/
# │   └── CodingLab6.ipynb
# ├── matplotlib_style.txt
# ├── requirements.txt
# ```
#
# then you can import the related functions via:
#
# ```
# import sys
# sys.path.append('./poisson-gpfa/')
# sys.path.append('./poisson-gpfa/funs')
#
# import funs.util as util
# import funs.engine as engine
# ```
#
# Change the paths if you have different directory structure. For the details of the algorithm, please refer to the thesis `hooram_thesis.pdf` from ILIAS.
#
# ### 2. Data
#
# Download the data file ```nds_cl_6_data.mat``` from ILIAS and save it in a ```data/``` folder.

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt  #
from IPython import embed

# style
import seaborn as sns

# poisson-gpfa
import sys

sys.path.append("./poisson-gpfa/")
sys.path.append("./poisson-gpfa/funs")

import funs.util as util
import funs.engine as engine

# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('load_ext', 'jupyter_black')

# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '--time --date --timezone --updated --python --iversions --watermark -p sklearn')


# In[2]:


plt.style.use("../matplotlib_style.txt")


# ## Task 1. Generate some toy data to test the poisson-GPFA code
#
# We start by verifying our code on toy data. The cell below contains code to generate data for 30 neurons, 100 trials (1000 ms each) and 50ms bin size. The neurons' firing rate $\lambda_k$ is assumed to be a constant $d_k$ modulated by a one-dimensional latent state $x$, which is drawn from a Gaussian process:
#
# $\lambda_k = \exp(c_kx + d_k)$
#
# Each neuron's weight $c_k$ is drawn randomly from a normal distribution and spike counts are sampled form a Poisson distribution with rate $\lambda_k$.
#
# Your task is to fit a Poisson GPFA model with one latent variable to this data (see `engine.PPGPFAfit`).
#
# Hint: You can use `util.dataset?`, `engine.PPGPFAfit?` or `util.initializeParams?` to find out more about the provided package.
#
# *Grading: 3 pts*

# In[3]:


# get_ipython().run_line_magic('pinfo', 'util.dataset')


# In[18]:


# ---------------------------------
# simulate a training set (0.5 pts)
# --------------------------------
# Initialize random number generator
seed = 42
# Specify dataset & fitting parameters
neurons = 30
trials = 100
dt = 1000  # ms for each trial
bin_size = 50  # ms
latent_variabel = 1
trainigs_dataset = util.dataset(
    trialDur=dt,
    binSize=bin_size,
    numTrials=trials,
    ydim=neurons,
    xdim=latent_variabel,
    seed=seed,
)
trainigs_dataset.plotTrajectory()


# In[5]:


# get_ipython().run_line_magic('pinfo', 'util.initializeParams')


# # In[6]:


# get_ipython().run_line_magic('pinfo', 'engine.PPGPFAfit')


# ### Fit the model

# In[7]:


# -----------------------
# fit the model (0.5 pts)
# -----------------------

# Initialize parameters using Poisson-PCA
params = util.initializeParams(
    xdim=latent_variabel, ydim=neurons, experiment=trainigs_dataset
)
print(params.keys())

# choose sensible parameters and run fit
fitToy = engine.PPGPFAfit(trainigs_dataset, params, batchSize=100)


# In[8]:


# some useful functions
def allTrialsState(fit, p):
    """Reshape the latent signal and the spike counts"""
    x = np.zeros([p, 0])
    for i in range(len(fit.infRes["post_mean"])):
        x = np.concatenate((x, fit.infRes["post_mean"][i]), axis=1)
    return x


def allTrialsX(training_set):
    """Reshape the ground truth
    latent signal and the spike counts"""
    x_gt = np.array([])
    for i in range(len(training_set.data)):
        x_gt = np.concatenate((x_gt, training_set.data[i]["X"][0]), axis=0)
    return x_gt


# ### Plot the ground truth vs. inferred model
# Verify your fit by plotting both ground truth and inferred parameters for:
# 1. weights C
# 2. biases d
# 3. latent state x
#
# Note that the sign of fitted latent state and its weights are ambiguous (you can flip both without changing the model). Make sure you correct the sign for the plot if it does not match the ground truth.

# In[9]:


# All trials latent state vector
x_est = allTrialsState(fitToy, 1)
x_true = allTrialsX(trainigs_dataset)
print(x_est.shape, x_true.shape)


# In[21]:


# ----------------------------------------------------
# Plot ground truth and inferred weights `C` (0.5 pts)
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
label = "Inferred over iterations"
# colormap of the inffered weights
colors = plt.cm.Blues(np.linspace(0.25, 1, len(fitToy.paramSeq[:-1])))

for iter in range(len(fitToy.paramSeq[:-1])):
    ax.scatter(
        np.arange(0, neurons),
        -fitToy.paramSeq[iter]["C"],
        color=colors[iter],
    )

ax.plot(
    np.arange(0, neurons),
    trainigs_dataset.params["C"],
    color="red",
    marker="o",
    linestyle="dashed",
    label="Ground truth",
)

ax.plot(
    np.arange(0, neurons),
    -fitToy.paramSeq[-1]["C"],
    color=colors[-1],
    marker="o",
    linestyle="dashed",
    label="Inferred",
)
# plot the colormap of the inferred weights
# change the arange to the number of iterations

ticks = np.arange(0, len(fitToy.paramSeq[:-1]) + 1, 10)


import matplotlib.colors as mcolors

cmap = mcolors.ListedColormap(colors)

# Create the colorbar
colorbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(colors))),
    ticks=np.arange(0, len(fitToy.paramSeq[:-1]) + 1, 10),
    label="Iterations",
)

colorbar.set_ticks(ticks)
ax.set_xlabel("Neurons")
ax.set_ylabel("Weights")
ax.legend(loc="upper right")


# add plot
# consider also plotting the optimal weights as a dotted line for reference


# In[22]:


# ---------------------------------------------------
# Plot ground truth and inferred baises `d` (0.5 pts)
# ---------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))
label = "Inferred over iterations"
# colormap of the inffered weights
colors = plt.cm.Blues(np.linspace(0.25, 1, len(fitToy.paramSeq[:-1])))

for iter in range(len(fitToy.paramSeq[:-1])):
    ax.scatter(
        np.arange(0, neurons),
        fitToy.paramSeq[iter]["d"],
        color=colors[iter],
    )

ax.plot(
    np.arange(0, neurons),
    trainigs_dataset.params["d"],
    color="red",
    marker="o",
    linestyle="dashed",
    label="Ground truth",
)

ax.plot(
    np.arange(0, neurons),
    fitToy.paramSeq[-1]["d"],
    color=colors[-1],
    marker="o",
    linestyle="dashed",
    label="Inferred",
)
# plot the colormap of the inferred weights
# change the arange to the number of iterations

ticks = np.arange(0, len(fitToy.paramSeq[:-1]) + 1, 10)


import matplotlib.colors as mcolors

cmap = mcolors.ListedColormap(colors)

# Create the colorbar
colorbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(colors))),
    ticks=np.arange(0, len(fitToy.paramSeq[:-1]) + 1, 10),
    label="Iterations",
)

colorbar.set_ticks(ticks)
ax.set_xlabel("Neurons")
ax.set_ylabel("Biases")
ax.legend(loc=(0.8, 0.99))


# add plot
# consider also plotting the optimal weights as a dotted line for reference


# In[12]:


# ------------------------------------------------------
# Plot ground truth and inferred latent states `x` (1pt)
# ------------------------------------------------------

trials = 100
first_trial = 40
last_trial = 50
plot_trials = np.arange(first_trial, last_trial)
print(np.shape(fitToy.infRes["post_mean"]))

fig, ax = plt.subplots(figsize=(12, 3))

start = first_trial * 20
end = start + 20
padding = 5
start_and_stops = []

for i in plot_trials:
    if i == plot_trials[0]:
        start_and_stops.extend([start, end - 1])
        ax.plot(
            np.arange(start, end),
            trainigs_dataset.data[i]["X"][0],
            color="red",
            linewidth=3,
            label="Ground truth",
        )
        ax.plot(
            np.arange(start, end),
            fitToy.infRes["post_mean"][i][0],
            color="blue",
            linewidth=3,
            label="Inferred",
        )
        ax.axvspan(end - 1, end + padding, color="gray", alpha=0.8)
        # trial number in the middle of the plot
        ax.text(
            (start + end) / 2 - 10,
            3,
            f"Trial {i}",
            verticalalignment="center",
            fontsize=12,
        )
        start += 20
        end += 20
    elif i == 48:
        start_and_stops.extend([start + padding, end - 1 + padding])
        ax.plot(
            np.arange(start + padding, end + padding),
            trainigs_dataset.data[i]["X"][0],
            color="red",
            linewidth=3,
        )
        ax.plot(
            np.arange(start + padding, end + padding),
            fitToy.infRes["post_mean"][i][0],
            color="blue",
            linewidth=3,
        )
        ax.axvspan(end - 1 + padding, end + padding + padding, color="gray", alpha=0.8)
        # trial number in on the top of the plot
        ax.text(
            (start + padding + end + padding) / 2 - 7,
            3,
            f"Trial {i}",
            verticalalignment="center",
            fontsize=12,
        )
        start = end + padding
        end += 20 + padding
    else:
        start_and_stops.extend([start + padding, end - 1 + padding])
        ax.plot(
            np.arange(start + padding, end + padding),
            trainigs_dataset.data[i]["X"][0],
            color="red",
            linewidth=3,
        )
        ax.plot(
            np.arange(start + padding, end + padding),
            -fitToy.infRes["post_mean"][i][0],
            color="blue",
            linewidth=3,
        )
        ax.axvspan(end - 1 + padding, end + padding + padding, color="gray", alpha=0.8)
        # trial number in on the top of the plot
        ax.text(
            (start + padding + end + padding) / 2 - 7,
            3,
            f"Trial {i}",
            verticalalignment="center",
            fontsize=12,
        )
        start = end + padding
        end += 20 + padding

ax.set_xlabel("Time [ms]")
ax.set_ylabel("Latent state")
ax.legend(loc=(0.9, 1.01))
ax.set_title("Latent states", pad=20)
ax.set_xticks(
    start_and_stops,
)
ax.set_xticklabels(start_and_stops, rotation=45)


# add plot
# plot only for a subset of trials
# consider seperating each trial by a vertical line


# ## Task 2: Fit GPFA model to real data.
#
# We now fit the model to real data and cross-validate over the dimensionality of the latent variable.
#
# *Grading: 2 pts*
#
#

# ### Load data
#
# The cell below implements loading the data and encapsulates it into a class that matches the interface of the Poisson GPFA engine. You don't need to do anything here.

# In[13]:


class EckerDataset:
    """Loosy class"""

    def __init__(
        self,
        path,
        subject_id=0,
        ydim=55,
        trialDur=2000,
        binSize=100,
        numTrials=100,
        ydimData=False,
        numTrData=True,
    ):
        T = int(trialDur / binSize)
        matdat = sio.loadmat(path)
        self.matdat = matdat
        data = []
        trial_durs = []
        for trial_id in range(numTrials):
            trial_time = matdat["spikeTimes"][:, trial_id][0]
            trial_big_time = np.min(trial_time)
            trial_end_time = np.max(trial_time)
            trial_durs.append(trial_end_time - trial_big_time)
        for trial_id in range(numTrials):
            Y = []
            spike_time = []
            data.append(
                {
                    "Y": matdat["spikeCounts"][:, :, trial_id],
                    "spike_time": matdat["spikeTimes"][:, trial_id],
                }
            )
        self.T = T
        self.trial_durs = trial_durs
        self.data = data
        self.trialDur = trialDur
        self.binSize = binSize
        self.numTrials = numTrials
        self.ydim = ydim


# In[14]:


path = "../data/nds_cl_6_data.mat"
data = EckerDataset(path)


# ### Fit Poisson GPFA models and perform model comparison
#
# Split the data into 80 trials used for training and 20 trials held out for performing model comparison. On the training set, fit models using one to five latent variables. Compute the performance of each model on the held-out test set.
#
# Hint: You can use the `crossValidation` function in the Poisson GPFA package.
#
# Optional: The `crossValidation` function computes the mean-squared error on the test set, which is not ideal. The predictive log-likelihood under the Poisson model would be a better measure, which you are welcome to compute instead.

# ### Derivation for log-likelihood
#
# _You can add your calculations in_ $\LaTeX$ _here_.
#
# $p_\lambda(x_t) = \ldots$
#
# $L(\lambda_k; x_1, ..., x_N) = \ldots$
#
# $log(L) = l(\lambda_k; x_1, ..., x_N) = \ldots$

# In[15]:


# ------------------------------
# Perfom cross validation (1 pt)
# ------------------------------

# fit the model to the data
xdim = 1  # number of modulators
initParams = util.initializeParams(xdim, data.ydim, data)
fitBatch = engine.PPGPFAfit(data, initParams, batchSize=data.numTrials)


# In[30]:


# do the actual cross validation
maxXdim = 5
xval = util.crossValidation(
    data,
    numTrainingTrials=80,
    numTestTrials=20,
    maxXdim=maxXdim,
)


# ### Plot the test error
#
# Make a plot of the test error for the five different models. As a baseline, please also include the test error of a model without a latent variable. This is essentially the mean-squared error of a constant rate model (or Poisson likelihood if you did the optional part above).

# In[27]:


# -----------------------------------------------------------------------------------------
# Compute and plot the test errors for the different latent variable models (0.5 + 0.5 pts)
# -----------------------------------------------------------------------------------------

train_set, test_set = util.splitTrainingTestDataset(
    data, numTrainingTrials=80, numTestTrials=20
)

# compute baseline error
# baseline_error = np.mean((test_set - np.mean(train_set)) ** 2)


# In[31]:


# Your plot here
fig, ax = plt.subplots(figsize=(4, 3))

# plot model error
plt.plot(np.arange(1, maxXdim + 1), xval.errs, "b.-", markersize=5, linewidth=2)
# plt.legend([xv.method],fontsize=9,framealpha=0.2)
plt.xlabel("Latent Dimensionality")
plt.ylabel("Error")
plt.title("Latent Dimension vs. Prediction Error")

# plot baseline
# ax.axhline(baseline_error, linestyle="--")


# ## Task 3. Visualization: population rasters and latent state. Use the model with a single latent state.
#
# Create a raster plot where you show for each trial the spikes of all neurons as well as the trajectory of the latent state `x` (take care of the correct time axis). Sort the neurons by their weights `c_k`.
#
#
# Plot only the first 20 trials.
#
# *Grading: 2 pts*

# In[95]:


print(np.shape(fitBatch.infRes["post_mean"]))
print(np.shape(data.data[3]["spike_time"][0]))
c = fitBatch.paramSeq[-1]["C"][:, 0]
# print(c)

# # sort c by x
# neurons = data.data[3]["spike_time"][30]
# print(neurons)
# # sorted_neurons = data.data[3]["spike_time"][x][3]
# print(sorted_neurons)


# In[111]:


from numpy import matlib

trialduration = 2000
binSize = 100
# Your plot here
fig, axs = plt.subplots(10, 2, figsize=(14, 14))
ts = np.linspace(0, trialduration, binSize)

#### stimulus
# amplitude
xa = 0.15
# sinusoidal stimulus
xs = 0.7 * xa * np.sin(ts / 1000 * 3.4 * 2 * np.pi) + xa

# sorted 'c'
index_sorted_c = np.argsort(fitBatch.paramSeq[-1]["C"][:, 0])
def squeezing(trial_data):
    return [np.squeeze(trial_data[i]) for i in range(len(trial_data))]

with sns.axes_style("ticks"):
    for ntrial, ax in enumerate(axs.flat):
        x = range(50, 2000, 100)  # assume binsize of 100ms

        # ------------------------
        # plot latent state (1 pt)
        # ------------------------
        ax.plot(x, fitBatch.infRes["post_mean"][ntrial][0])

        # hint: can be plotted on top of the corresponding raster

        # sort neurons by firing rate
        neurons_trial = data.data[ntrial]["spike_time"][index_sorted_c]
        neruons_trial_squeezed = squeezing(neurons_trial)
        # ----------------------------------
        # plot raster for each neuron (1 pt)
        # ----------------------------------
        for neruon in neruons_trial_squeezed:
            ax.scatter(neruon, np.ones(len(neruon)) * 0.5, s=1, color="black")

        if ntrial == 0:
            ax.legend()
        if ntrial == 1:
            ax.plot([1000, 2000], [-30, -30], color="green")
            ax.text(1300, -50, "1sec")
        if ntrial < 2:
            ax.plot(ts, (xs * 40) + data.ydim, "k", color="black")

        ax.set_yticks([])
        ax.set_xticks([])


# ## Task 4. Visualization of covariance matrix.
#
# Plot (a) the covariance matrix of the observed data as well as its approximation using (b) one and (c) five latent variable(s). Use the analytical solution for the covariance matrix of the approximation*. Note that the solution is essentially the mean and covariance of the [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution).
#
# $ \mu = \exp(\frac{1}{2} \text{ diag}(CC^T)+d)$
#
# $ \text{Cov}= \mu\mu^T \odot \exp(CC^T)+\text{ diag}(\mu) - \mu\mu^T$
#
# *[Krumin, M., and Shoham, S. (2009). Generation of Spike Trains with Controlled Auto- and Cross-Correlation Functions. Neural Computation 21, 1642–1664](http://www.mitpressjournals.org/doi/10.1162/neco.2009.08-08-847).
#
# *Grading: 3 pts*

# In[ ]:


# insert your code here

# --------------------------------------------------------------
# Complete the analytical solution for the covariance matrix of
# the approximation using the provide equations (2 pts)
# --------------------------------------------------------------


def cov(fit):
    # add your code here
    return c, mu


# --------------------------------------------------------------
# Plot the covariance matrix (1 pt) of
# (1) the observed data
# (2) its approximation using 1 latent variable
# (3) its approximation using 5 latent variable
# --------------------------------------------------------------

obs_corr = np.cov(data.all_raster)
opt_r1, mu1 = cov(xval.fits[0])
opt_r5, mu5 = cov(xval.fits[4])

vmin = -1
vmax = 1

fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
# add plot to visualize the differences in the covariance matrices
