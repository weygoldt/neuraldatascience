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
# Name: FILL IN YOUR NAMES HERE
# 
# # Coding Lab 4
# 
# If needed, download the data files ```nds_cl_4_*.csv``` from ILIAS and save it in the subfolder ```../data/```. Use a subset of the data for testing and debugging, ideally focus on a single cell (e.g. cell number x). The spike times and stimulus conditions are read in as pandas data frames. You can solve the exercise by making heavy use of that, allowing for many quite compact computationis. If you need help on that, there is lots of [documentation](http://pandas.pydata.org/pandas-docs/stable/index.html) and several good [tutorials](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python#gs.L37i87A) are available online. Of course, converting the data into classical numpy arrays is also valid.

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.optimize as opt
from IPython import embed

from scipy import signal as signal

import itertools

# %matplotlib qt6
# `
# # get_ipython().run_line_magic('load_ext', 'jupyter_black')

# # get_ipython().run_line_magic('load_ext', 'watermark')
# # get_ipython().run_line_magic('watermark', '--time --date --timezone --updated --python --iversions --watermark -p sklearn')
# `

# In[8]:


plt.style.use("../matplotlib_style.txt")


# ## Load data

# In[9]:


spikes = pd.read_csv("../data/nds_cl_4_spiketimes.csv")  # neuron id, spike time
stims = pd.read_csv("../data/nds_cl_4_stimulus.csv")  # stimulus onset in ms, direction

stimDur = 2000.0  # in ms
nTrials = 11  # number of trials per condition
nDirs = 16  # number of conditions
deltaDir = 22.5  # difference between conditions

stims["StimOffset"] = stims["StimOnset"] + stimDur


# We require some more information about the spikes for the plots and analyses we intend to make later. With a solution based on dataframes, it is natural to compute this information here and add it as additional columns to the `spikes` dataframe by combining it with the `stims` dataframe. We later need to know which condition (`Dir`) and trial (`Trial`) a spike was recorded in, the relative spike times compared to stimulus onset of the stimulus it was recorded in (`relTime`) and whether a spike was during the stimulation period (`stimPeriod`). But there are many options how to solve this exercise and you are free to choose any of them.

# In[10]:


# you may add computations as specified above
spikes["Dir"] = np.nan
spikes["relTime"] = np.nan
spikes["Trial"] = np.nan
spikes["stimPeriod"] = np.nan

dirs = np.unique(stims["Dir"])
trialcounter = np.zeros_like(dirs)

for i, row in stims.iterrows():
    trialcounter[dirs == row["Dir"]] += 1

    i0 = spikes["SpikeTimes"] > row["StimOnset"]
    i1 = spikes["SpikeTimes"] < row["StimOffset"]

    select = i0.values & i1.values

    spikes.loc[select, "Dir"] = row["Dir"]
    spikes.loc[select, "Trial"] = trialcounter[dirs == row["Dir"]][0]
    spikes.loc[select, "relTime"] = spikes.loc[select, "SpikeTimes"] - row["StimOnset"]
    spikes.loc[select, "stimPeriod"] = True

spikes = spikes.dropna()


# In[11]:


spikes.head()


# ## Task 1: Plot spike rasters
# 
# In a raster plot, each spike is shown by a small tick at the time it occurs relative to stimulus onset. Implement a function `plotRaster()` that plots the spikes of one cell as one trial per row, sorted by conditions (similar to what you saw in the lecture). Why are there no spikes in some conditions and many in others?
# 
# If you opt for a solution without a dataframe, you need to change the interface of the function.
# 
# *Grading: 2 pts*
# 

# In[12]:


def plotRaster(spikes, neuron):
    """plot spike rasters for a single neuron sorted by condition

    Parameters
    ----------

    spikes: pd.DataFrame
        Pandas DataFrame with columns
            Neuron | SpikeTimes | Dir | relTime | Trial | stimPeriod

    neuron: int
        Neuron ID


    Note
    ----

    this function does not return anything, it just creates a plot!
    """

    spikes_neuron = spikes[spikes["Neuron"] == neuron]

    # sort the spikes by direction
    spikes_neuron = spikes_neuron.sort_values(by="Dir")
    # plot the spikes for each trial in a raster plot
    dirs = spikes_neuron["Dir"].unique()[::-1]

    fig, ax = plt.subplots(len(dirs), figsize=(10, 9), sharex=True)

    for d, directions in enumerate(dirs):
        spikes_neuron_dir = spikes_neuron[spikes_neuron["Dir"] == directions]
        ax[d].scatter(
            spikes_neuron_dir["relTime"],
            spikes_neuron_dir["Trial"],
            marker="|",
            color="k",
            s=20,
            linewidths=0.8,
        )
        ax[d].set_ylabel(f"{directions}", rotation=0, labelpad=20)
        ax[d].set_yticks([])
    ax[0].set_title(f"Neuron {neuron}")
    fig.supylabel("Directions [degree]")
    # create supylabel on the right side

    # fig.supylabel("Trials n")
    fig.supxlabel("Time [ms]")

    # insert your code here
    # stim direction should be on the y-axis and time on the x-axis
    # you can use plt.scatter or plt.plot to plot the responses to each stim


# Show examples of different neurons. Good candidates to check are 28, 29, 36 or 37. 

# In[13]:


# plotRaster(spikes, 28)
# plotRaster(spikes, 29)
# plotRaster(spikes, 36)
# plotRaster(spikes, 37)


# ## Task 2: Plot spike density functions
# 
# Compute an estimate of the spike rate against time relative to stimulus onset. There are two ways:
# * Discretize time: Decide on a bin size, count the spikes in each bin and average across trials. 
# * Directly estimate the probability of spiking using a density estimator with specified kernel width. 
# 
# Implement one of them in the function `plotPsth()`. If you use a dataframe you may need to change the interface of the function.
# 
# 
# *Grading: 2 pts*
# 

# In[14]:


def plotPSTH(spikes, neuron):
    """Plot PSTH for a single neuron sorted by condition

    Parameters
    ----------

    spikes: pd.DataFrame
        Pandas DataFrame with columns
            Neuron | SpikeTimes | Dir | relTime | Trial | stimPeriod

    neuron: int
        Neuron ID


    Note
    ----

    this function does not return anything, it just creates a plot!
    """

    def gaussian_pdf(x, loc, scale):
        pdf = np.exp(-((x - loc) ** 2) / (2.0 * scale**2)) / (
            np.sqrt(2 * np.pi) * scale
        )
        return pdf

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    max_trials = 11
    spikes_neuron = spikes[spikes["Neuron"] == neuron]
    # insert your code here
    sigma = 0.1
    tmax = 20 * sigma
    ktime = np.arange(-tmax, tmax, 0.001)
    kernel_o = gaussian_pdf(ktime, 0, sigma)
    spikes_neuron = spikes_neuron.sort_values(by="Dir")
    # plot the spikes for each trial in a raster plot
    dirs = spikes_neuron["Dir"].unique()
    time = np.arange(0, 2000.0, 0.1)
    fig, ax = plt.subplots(len(dirs), figsize=(10, 9), sharex=True, sharey=True)

    for d, directions in enumerate(dirs[::-1]):
        spikes_neuron_dir = spikes_neuron[spikes_neuron["Dir"] == directions]
        # create empty array for the rate
        rates = np.zeros((len(spikes_neuron_dir["Trial"].unique()), len(time)))
        for t, trial in enumerate(np.sort(spikes_neuron_dir["Trial"].unique())):
            spikes_neuron_dir_trial = spikes_neuron_dir[
                spikes_neuron_dir["Trial"] == trial
            ]
            spikes = spikes_neuron_dir_trial["relTime"].to_numpy()
            index = []
            for spike in spikes:
                idx = find_nearest(time, spike)
                index.append(idx)
            binaryrate = np.zeros(len(time))
            binaryrate[index] = 1.0
            rate = np.convolve(binaryrate, kernel_o, mode="same")
            rates[t, :] = rate
        mean_rates = np.sum(rates, axis=0) / max_trials
        ax[d].plot(time, mean_rates)
        ax[d].set_ylabel(f"{directions}", rotation=0, labelpad=20)
        ax[d].set_yticks([])
    ax[0].set_title(f"Neuron {neuron}")
    fig.supylabel("Directions [degree]")
    # create supylabel on the right side

    # fig.supylabel("Trials n")
    fig.supxlabel("Time [ms]")


# Show examples of different neurons. Good candidates to check are 28, 29, 36 or 37. 

# In[15]:


# plotPSTH(spikes, 28)
# plotPSTH(spikes, 29)
# plotPSTH(spikes, 36)
# plotPSTH(spikes, 37)


# # In[16]:


# plotRaster(spikes, 37)
# plotPSTH(spikes, 37)


# ## Task 3: Fit and plot tuning functions
# 
# The goal is to visualize the activity of each neuron as a function of stimulus direction. First, compute the spike counts of each neuron for each direction of motion and trial.  The result should be a matrix `x`, where $x_{jk}$ represents the spike count of the $j$-th response to the $k$-th direction of motion (i.e. each column contains the spike counts for all trials with one direction of motion).	If you used dataframes above, the `groupby()` function allows to implement this very compactely. Make sure you don't loose trials with zero spikes though. Again, other implementations are completely fine.
# 
# Fit the tuning curve, i.e. the average spike count per direction, using a von Mises model. To capture the non-linearity and direction selectivity of the neurons, we will fit a modified von Mises function:
# 
# $$ f(\theta) = \exp(\alpha + \kappa (\cos (2*(\theta-\phi))-1) + \nu (\cos (\theta-\phi)-1))$$
# 
# Here, $\theta$ is the stimulus direction. Implement the von Mises function in `vonMises()` and plot it to understand how to interpret its parameters $\phi$, $\kappa$, $\nu$, $\alpha$. Perform a non-linear least squares fit using a package/function of your choice. Implement the fitting in `tuningCurve()`. 
# 
# Plot the average number of spikes per direction, the spike counts from individual trials as well as your optimal fit.
# 
# Select two cells that show nice tuning to test you code.
# 
# *Grading: 3 pts*

# In[17]:


def vonMises(theta, alpha, kappa, nu, phi):
    """Evaluate the parametric von Mises tuning curve with parameters p at locations theta.

    Parameters
    ----------

    θ: np.array, shape=(N, )
        Locations. The input unit is degree.

    theta, alpha, kappa, nu, phi: float
        Function parameters

    Return
    ------
    f: np.array, shape=(N, )
        Tuning curve.
    """

    theta = np.radians(theta)
    phi = np.radians(phi)

    # insert your code here
    f = np.exp(
        alpha + kappa * (np.cos(2 * (theta - phi)) - 1) + nu * (np.cos(theta - phi) - 1)
    )

    # -----------------------------------
    # Implement the Mises model (0.5 pts)
    # -----------------------------------

    return f


# Plot the von Mises function while varying the parameters systematically.

# In[18]:


# --------------------------------------------------------------------------------
# plot von Mises curves with varying parameters and explain what they do (0.5 pts)
# --------------------------------------------------------------------------------
alpha = np.arange(-1, 1, 0.1)
kappa = np.arange(0, 10, 1)
nu = np.arange(0, 10, 1)
phi = np.arange(0, 2 * np.pi, np.pi / 2)
theta = np.arange(0, 2 * np.pi, 0.1)
x = np.linspace(
    0,
    360,
    len(theta),
)
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
for i, a in enumerate(alpha):
    f = vonMises(theta, a, 1, 1, 0)
    ax[0].plot(x, f, label=f"alpha = {a}")

    # ax[0].legend()
for i, k in enumerate(kappa):
    f = vonMises(theta, 0, k, 1, 0)
    ax[1].plot(x, f, label=f"kappa = {k}")
    # ax[1].legend()
for i, n in enumerate(nu):
    f = vonMises(theta, 0, 1, n, 0)
    ax[2].plot(x, f, label=f"nu = {n}")
    # ax[2].legend()
for i, p in enumerate(phi):
    f = vonMises(theta, 0, 1, 1, p)
    ax[3].plot(x, f, label=f"phi = {p}")
    # ax[3].legend()
for i, t in enumerate(theta):
    f = vonMises(t, 0, 1, 1, 0)
    ax[4].scatter(t, f, label=f"theta = {t}")
ax[0].set_title("alpha")
ax[1].set_title("kappa")
ax[2].set_title("nu")
ax[3].set_title("phi")
ax[4].set_title("theta")
plt.close()

# - alpha is scales up the peak of the function
# - kappa is the width of the peak
# - nu scales the peaks realtive to each other
# - phi shifts the function along the x-axis
# - theta is the stimulus direction

# In[64]:


def tuningCurve(counts, dirs, show=True):
    """Fit a von Mises tuning curve to the spike counts in count with direction dir using a least-squares fit.

    Parameters
    ----------

    counts: np.array, shape=(total_n_trials, )
        the spike count during the stimulation period

    dirs: np.array, shape=(total_n_trials, )
        the stimulus direction in degrees

    show: bool, default=True
        Plot or not.


    Return
    ------
    popt: np.array or list, (4,)
        parameter vector of tuning curve function
    """

    # insert your code here

    # get spike count
    matrix = {}
    for directions in np.unique(dirs):
        matrix[directions] = counts[dirs == directions]
    spike_counts_df = pd.DataFrame(matrix)
    

    # convert to radians
    
    popt, pcov = opt.curve_fit(vonMises, dirs, counts, maxfev=100000)
    
    x = np.arange(0, 360, 1)

    y = vonMises(x, *popt)

    if show == True:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(dirs, counts, "o", label="data")
        ax.plot(x, y, label="fit")
        ax.set_xlabel("direction (degree)")
        ax.set_ylabel("spike count")
        plt.legend()
        plt.show()
        plt.close()

        return
    else:
        return popt


# Plot tuning curve and fit for different neurons. Good candidates to check are 28, 29 or 37. 

# In[53]:


def get_data(spikes, neuron):
    spk_by_dir = (
        spikes[spikes["Neuron"] == neuron]
        .groupby(["Dir", "Trial"])["stimPeriod"]
        .sum()
        .astype(int)
        .reset_index()
    )

    dirs = spk_by_dir["Dir"].values
    counts = spk_by_dir["stimPeriod"].values

    # because we count spikes only when they are present, some zero entries in the count vector are missing
    for i, Dir in enumerate(np.unique(spikes["Dir"])):
        m = nTrials - np.sum(dirs == Dir)
        if m > 0:
            dirs = np.concatenate((dirs, np.ones(m) * Dir))
            counts = np.concatenate((counts, np.zeros(m)))

    idx = np.argsort(dirs)
    dirs_sorted = dirs[idx]  # sorted dirs
    counts_sorted = counts[idx]

    return dirs_sorted, counts_sorted


# In[65]:


# ---------------------------------------------------------
# plot tuning curve and fit for different neurons (0.5 pts)
# ---------------------------------------------------------

dirs, counts = get_data(spikes, 28)
tuningCurve(counts, dirs, show=True)


# In[22]:


dirs, counts = get_data(spikes, 29)
tuningCurve(counts, dirs, show=True)
# add plot


# In[23]:


dirs, counts = get_data(spikes, 36)
tuningCurve(counts, dirs, show=True)
# add plot


# In[24]:


dirs, counts = get_data(spikes, 37)
tuningCurve(counts, dirs, show=True)
# add plot


# In[25]:


dirs, counts = get_data(spikes, 32)
tuningCurve(counts, dirs, show=True)
# add plot
embed()
exit()

# ## Task 4: Permutation test for direction tuning
# 
# Implement a permutation test to quantitatively assess whether a neuron is direction/orientation selective. To do so, project the vector of average spike counts, $m_k=\frac{1}{N}\sum_j x_{jk}$ on a complex exponential with two cycles, $v_k = \exp(\psi i \theta_k)$, where $\theta_k$ is the $k$-th direction of motion in radians and $\psi \in 1,2$ is the fourier component to test (1: direction, 2: orientation). Denote the projection by $q=m^Tv$. The magnitude $|q|$ tells you how much power there is in the $\psi$-th fourier component. 
# 
# Estimate the distribution of |q| under the null hypothesis that the neuron fires randomly across directions by running 1000 iterations where you repeat the same calculation as above but on a random permutation of the trials (that is, randomly shuffle the entries in the spike count matrix x). The fraction of iterations for which you obtain a value more extreme than what you observed in the data is your p-value. Implement this procedure in the function ```testTuning()```. 
# 
# Illustrate the test procedure for one of the cells from above. Plot the sampling distribution of |q| and indicate the value observed in the real data in your plot. 
# 
# How many cells are tuned at p < 0.01?
# 
# *Grading: 3 pts*
# 

# In[26]:


def testTuning(counts, dirs, psi=1, niters=1000, show=False):
    """Plot the data if show is True, otherwise just return the fit.

    Parameters
    ----------

    counts: np.array, shape=(total_n_trials, )
        the spike count during the stimulation period

    dirs: np.array, shape=(total_n_trials, )
        the stimulus direction in degrees

    psi: int
        fourier component to test (1 = direction, 2 = orientation)

    niters: int
        Number of iterations / permutation

    show: bool
        Plot or not.

    Returns
    -------
    p: float
        p-value
    q: float
        magnitude of second Fourier component

    qdistr: np.array
        sampling distribution of |q| under the null hypothesis

    """

    # insert your code here

    # -------------------------------
    # calculate m, nu and q (0.5 pts)
    # -------------------------------

    # -------------------------------------------------------------------------
    # Estimate the distribution of q under the H0 and obtain the p value (1 pt)
    # -------------------------------------------------------------------------

    if show == True:
        # -------------------------------
        # plot the test results (0.5 pts)
        # -------------------------------
        fig, ax = plt.subplots(figsize=(7, 4))
        # you can use sns.histplot for the histogram
    else:
        return p, q, qdistr


# Show null distribution for the example cell:

# In[27]:


# ------------------------------------------------------------
# Plot null distributions for example cells 28 & 29. (0.5 pts)
# ------------------------------------------------------------

dirs, counts = get_data(spikes, 28)
# add plot


# In[28]:


dirs, counts = get_data(spikes, 29)
# add plot


# Test all cells for orientation and direction tuning

# In[29]:


# -------------------------------------------------------
# Test cells for orientation / direction tuning (0.5 pts)
# -------------------------------------------------------

# collect p values for orientation / direction selectivity


# Number of direction tuned neurons:

# In[30]:


# count cells with p > 0.01 (which ones are they?)


# Number of orientation tuned neurons:

# In[31]:


# count cells with p > 0.01 (which ones are they?)

