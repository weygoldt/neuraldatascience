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
# Names: Kathrin Root, Alexander Wendt, Patrick Weygoldt
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

# In[3]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
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


# In[4]:


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

# In[5]:


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


# ### Fit the model

# In[6]:


# -----------------------
# fit the model (0.5 pts)
# -----------------------

# Initialize parameters using Poisson-PCA
params = util.initializeParams(
    xdim=latent_variabel, ydim=neurons, experiment=trainigs_dataset
)

# choose sensible parameters and run fit
fitToy = engine.PPGPFAfit(trainigs_dataset, params, batchSize=100)


# In[7]:


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


# In[17]:


# ------------------------------------------------------
# Plot ground truth and inferred latent states `x` (1pt)
# ------------------------------------------------------

trials = 100
first_trial = 45
last_trial = 47
dt = 1000  # ms for each trial
bin_size = 50  # ms
plot_trials = np.arange(first_trial, last_trial)
print(np.shape(fitToy.infRes["post_mean"]))

fig, ax = plt.subplots(figsize=(12, 3))
embed()
exit()
for i in plot_trials:
    time_trial = np.arange(i * dt, (i + 1) * dt, bin_size)
    # because the sign of the latend states are arbitrary we need to flip them for trial 48
    if i == 48:
        ax.plot(
            time_trial,
            trainigs_dataset.data[i]["X"][0],
            color="red",
            linewidth=3,
            label="Ground truth",
        )
        ax.plot(
            time_trial,
            fitToy.infRes["post_mean"][i][0],
            color="blue",
            linewidth=3,
        )
        ax.axvline(time_trial[-1], ymin=-2, ymax=3, color="gray", alpha=0.8)
        # trial number in on the top of the plot
        ax.text(
            time_trial / 2 - 7,
            3,
            f"Trial {i}",
            verticalalignment="center",
            fontsize=12,
            label="Inferred",
        )
    ax.plot(
        time_trial,
        trainigs_dataset.data[i]["X"][0],
        color="red",
        linewidth=3,
        label="Ground truth",
    )
    ax.plot(
        time_trial,
        -fitToy.infRes["post_mean"][i][0],
        color="blue",
        linewidth=3,
        label="Inferred",
    )
    ax.axvline(time_trial[-1], ymin=-2, ymax=3, color="gray", alpha=0.8)
    # trial number in on the top of the plot
    ax.text(
        time_trial / 2 - 7,
        3,
        f"Trial {i}",
        verticalalignment="center",
        fontsize=12,
        label="Inferred",
    )


    # if i == plot_trials[0]:
    #     ax.legend(loc=(0.9, 1.01))


ax.set_xlabel("Time [ms]")
ax.set_ylabel("Latent state")
ax.set_title("Latent states", pad=20)
# ax.set_xticks(np.arange(first_trial * dt, (last_trial + 1) * dt + 1, bin_size))

# add plot
# plot only for a subset of trials
# consider seperating each trial by a vertical line


# ## Task 2: Fit GPFA model to real data. 
# 
# We now fit the model to real data and cross-validate ov

# ### Load data
# 
# The cell below implements loading the data and encapsulates it into a class that matches the interface of the Poisson GPFA engine. You don't need to do anything here.

# In[140]:


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
        util.dataset.getMeanAndVariance(self)
        util.dataset.getAvgFiringRate(self)
        util.dataset.getAllRaster(self)


# In[146]:


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


# In[160]:


# do the actual cross validation
maxXdim = 5
xval = util.crossValidation(
    data,
    numTrainingTrials=80,
    numTestTrials=20,
    maxXdim=maxXdim,
    learningMethod="diag",
    batchSize=80,
)


# ### Plot the test error
# 
# Make a plot of the test error for the five different models. As a baseline, please also include the test error of a model without a latent variable. This is essentially the mean-squared error of a constant rate model (or Poisson likelihood if you did the optional part above).

# In[159]:


# -----------------------------------------------------------------------------------------
# Compute and plot the test errors for the different latent variable models (0.5 + 0.5 pts)
# -----------------------------------------------------------------------------------------

train_set, test_set = util.splitTrainingTestDataset(
    data, numTrainingTrials=80, numTestTrials=20
)

# compute baseline error
baseline_error = np.mean((test_set - np.mean(train_set)) ** 2)


# In[163]:


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

# In[164]:


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
        # plot latent state on different y axis
        ax2 = ax.twinx()
        ax2.plot(x, fitBatch.infRes["post_mean"][ntrial][0])
        ax2.set_yticks(np.arange(-5, 5))
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        # hint: can be plotted on top of the corresponding raster

        # sort neurons by weights
        neurons_trial = data.data[ntrial]["spike_time"][index_sorted_c]
        neruons_trial_squeezed = squeezing(neurons_trial)
        # ----------------------------------
        # plot raster for each neuron (1 pt)
        # ----------------------------------
        for neuron in range(len(neruons_trial_squeezed)):
            if neurons_trial[neuron].size == 1:
                continue
            if neurons_trial[neuron].size == 0:
                continue

            else:
                ax.eventplot(
                    neruons_trial_squeezed[neuron],
                    lineoffsets=neuron,
                    linelengths=0.6,
                    linewidths=0.6,
                    color="black",
                )
        ax.set_title("Trial " + str(ntrial + 1), loc="left")

        if ntrial == 0:
            ax.legend()
        if ntrial == 19:
            ax.plot([1000, 2000], [-30, -30], color="green")
            ax.text(1300, -50, "1sec")
        if ntrial < 2:
            ax.plot(ts, (xs * 40) + data.ydim, "k", color="black")

        ax.set_yticks([])
        ax.set_xticks([])
        fig.supxlabel("Time [ms]")


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

# In[188]:


# insert your code here

# --------------------------------------------------------------
# Complete the analytical solution for the covariance matrix of
# the approximation using the provide equations (2 pts)
# --------------------------------------------------------------


def cov(fit):
    # add your code here
    mu = np.exp(
        (
            np.diag(fit.paramSeq[-1]["C"] @ fit.paramSeq[-1]["C"].T)
            + fit.paramSeq[-1]["d"]
        )
        / 2
    )
    c = (
        (mu @ mu.T) * np.exp(fit.paramSeq[-1]["C"] @ fit.paramSeq[-1]["C"].T)
        + np.diag(mu)
        - (mu @ mu.T)
    )
    return c, mu


print(np.shape(xval.fits[0].infRes["post_cov"]))

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

axs[0].imshow(obs_corr, vmin=vmin, vmax=vmax, cmap="winter")
axs[0].set_title("Observed Data")
axs[1].imshow(opt_r1, vmin=vmin, vmax=vmax, cmap="winter")

axs[1].set_title("1 Latent Variable")
axs[2].imshow(opt_r5, vmin=vmin, vmax=vmax, cmap="winter")

axs[2].set_title("5 Latent Variables")
fig.supxlabel("Neuron Index")
fig.supylabel("Neuron Index")

