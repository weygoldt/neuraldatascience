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
# Student name: Kathrin Root, Alexander Wendt, Patrick Weygoldt 
# 
# # Coding Lab 3
# 
# - __Data__: Download the data file ```nds_cl_3_*.csv``` from ILIAS and save it in a subfolder ```../data/```.
# - __Dependencies__: You don't have to use the exact versions of all the dependencies in this notebook, as long as they are new enough. But if you run "Run All" in Jupyter and the boilerplate code breaks, you probably need to upgrade them.
# 
# Two-photon imaging is widely used to study computations in populations of neurons. In this exercise sheet we will study properties of different indicators and work on methods to infer spikes from calcium traces. All data is provided at a sampling rate of 100 Hz. For analysis, please resample it to 25 Hz using `scipy.signal.decimate`.

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from IPython import embed
# from __future__ import annotations

# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('load_ext', 'jupyter_black')

# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', '--time --date --timezone --updated --python --iversions --watermark -p sklearn')


# In[4]:


plt.style.use("../matplotlib_style.txt")


# ## Load data

# In[5]:


# ogb dataset from Theis et al. 2016 Neuron
ogb_calcium = pd.read_csv("../data/nds_cl_3_ogb_calcium.csv", header=0)
ogb_spikes = pd.read_csv("../data/nds_cl_3_ogb_spikes.csv", header=0)

# gcamp dataset from Chen et al. 2013 Nature
gcamp_calcium = pd.read_csv("../data/nds_cl_3_gcamp2_calcium.csv", header=0)
gcamp_spikes = pd.read_csv("../data/nds_cl_3_gcamp2_spikes.csv", header=0)


# In[6]:


ogb_calcium.shape, ogb_spikes.shape, gcamp_calcium.shape, gcamp_spikes.shape


# `<<<<<<< HEAD`

# In[7]:


ogb_spikes.head()
ogb_calcium.head()
# ogb_spikes["6"].to_numpy()


# `=======`

# In[8]:


ogb_spikes.head()


# ## Task 1: Visualization of calcium and spike recordings
# 
# We start again by plotting the raw data - calcium and spike traces in this case. One dataset has been recorded using the synthetic calcium indicator OGB-1 at population imaging zoom (~100 cells in a field of view) and the other one using the genetically encoded indicator GCamp6f zooming in on individual cells. Plot the traces of an example cell from each dataset to show how spikes and calcium signals are related. A good example cell for the OGB-dataset is cell 5. For the CGamp-dataset a good example is cell 6. Zoom in on a small segment of tens of seconds and offset the traces such that a valid comparison is possible.
# 
# *Grading: 2 pts*

# In[25]:


good_cell_ogb = 4
good_cell_gcamp = 5

samplerate = 25
downsaple_factor = 100 / samplerate
print(int(downsaple_factor))

# Calcium data
calcium_ogb_cell = ogb_calcium[f"{good_cell_ogb}"].to_numpy()
calcium_ogb_cell_dec = signal.decimate(calcium_ogb_cell, int(downsaple_factor))

calcium_gcamp_cell = gcamp_calcium[f"{good_cell_gcamp}"].to_numpy()
calcium_gcamp_cell_dec = signal.decimate(calcium_gcamp_cell, int(downsaple_factor))


# Spike data
spike_ogb_cell = ogb_spikes[f"{good_cell_ogb}"].to_numpy()
spike_ogb_cell_dec = signal.decimate(spike_ogb_cell, int(downsaple_factor))

spike_gcamp_cell = gcamp_spikes[f"{good_cell_gcamp}"].to_numpy()
spike_gcamp_cell_dec = signal.decimate(spike_gcamp_cell, int(downsaple_factor))


# --------------------------
# make new time axis

time_ogb = np.arange(0, len(calcium_ogb_cell)) / 100
time_ogb_dec = np.arange(0, len(calcium_ogb_cell_dec)) / samplerate
time_gcamp = np.arange(0, len(calcium_gcamp_cell)) / 100
time_gcamp_dec = np.arange(0, len(calcium_gcamp_cell_dec)) / samplerate


fig, axs = plt.subplots(
    2, 2, figsize=(9, 5), height_ratios=[3, 1], layout="constrained", sharex=True
)
xlims_ogb = [100, 150]
xlims_gcamp = [100, 150]
axs[0, 0].plot(time_ogb, calcium_ogb_cell, label="ogb")
axs[0, 0].plot(time_ogb_dec, calcium_ogb_cell_dec, label="ogb dec")
axs[0, 0].set_title("Calcium ogb")
axs[0, 0].set_xlim(xlims_ogb)
axs[0, 0].set_ylabel("Delta F/F")
axs[0, 0].legend()


axs[0, 1].plot(time_gcamp, calcium_gcamp_cell, label="gcamp")
axs[0, 1].plot(time_gcamp_dec, calcium_gcamp_cell_dec, label="gcamp dec")
axs[0, 1].set_title("Calcium gcamp")
axs[0, 1].set_ylabel("Delta F/F")
axs[0, 1].set_xlim(xlims_gcamp)
axs[0, 1].legend()

axs[1, 0].plot(time_ogb, spike_ogb_cell)
axs[1, 0].plot(time_ogb_dec, spike_ogb_cell_dec)
axs[1, 0].set_title("Spike ogb")
axs[1, 0].set_xlim(xlims_ogb)
axs[1, 0].scatter(time_ogb, spike_ogb_cell)
axs[1, 0].set_ylabel("Spikes")
axs[1, 0].set_xlabel("time [s]")

axs[1, 1].plot(time_gcamp, spike_gcamp_cell)
axs[1, 1].plot(time_gcamp_dec, spike_gcamp_cell_dec)
axs[1, 1].scatter(time_gcamp, spike_gcamp_cell)
axs[1, 1].set_title("Spike gcamp")
axs[1, 1].set_xlim(xlims_gcamp)
axs[1, 1].set_ylabel("Spikes")
axs[1, 1].set_xlabel("time [s]")


# plot raw gcamp data


# ## Task 2: Simple deconvolution
# 
# It is clear from the above plots that the calcium events happen in relationship to the spikes. As a first simple algorithm implement a deconvolution approach like presented in the lecture in the function `deconv_ca`. Assume an exponential kernel where the decay constant depends on the indicator ($\tau_{OGB}= 0.5 s$, $\tau_{GCaMP}= 0.1 s$). As we know that there can be no negative rates, apply a heavyside function to the output. Plot the kernel as well as an example cell with true and deconvolved spike rates. Scale the signals such as to facilitate comparisons. You can use functions from `scipy` for this.
# 
# *Grading: 3 pts*
# 

# In[16]:


def deconv_ca(ca, tau, dt):
    """Compute the deconvolution of the calcium signal.

    Parameters
    ----------

    ca: np.array, (n_points,)
        Calcium trace

    tau: float
        decay constant of conv kernel

    dt: float
        sampling interval. 1/samplerate

    Return
    ------

    sp_hat: np.array
    kernel: np.array
    """

    # get the filter coefficients
    coeffs = signal.butter(4, 0.3, "lowpass", fs=dt, output="sos")
    # pass the filter coefficients  and the data to the filter function
    filtered_data = signal.sosfiltfilt(sos=coeffs, x=ca)


    n = 0
    while n < 5000:
        all_peaks = signal.find_peaks(filtered_data)[0]
        diff_peaks = np.abs(np.diff(filtered_data[all_peaks]))

        p_min = np.argmin(diff_peaks)


        datapoints = 25
        mean_segment = np.mean(
            filtered_data[
                np.arange(all_peaks[p_min] - datapoints, all_peaks[p_min] + datapoints)
            ]
        )
        filtered_data[
            np.arange(all_peaks[p_min] - datapoints, all_peaks[p_min] + datapoints)
        ] = mean_segment

        p_min = diff_peaks[p_min]
        if p_min > 1:
            break
        n += 1

    inv_kernel = np.exp(-np.arange(0, 3 * tau, 1 / dt) / tau)

    sp_hat, _ = signal.deconvolve(filtered_data, inv_kernel)
    sp_hat[sp_hat < 0] = 0
    sp_hat = np.pad(sp_hat, (0, len(filtered_data) - len(sp_hat)), "constant")
    return sp_hat, inv_kernel


deconv_calcium_ogb_cell, ogb_kernel = deconv_ca(calcium_ogb_cell_dec, 0.5, samplerate)
deconv_calcium_gcamp_cell, gcamp_kernel = deconv_ca(
    calcium_gcamp_cell_dec, 0.1, samplerate
)


# In[17]:


fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")

ax.plot(ogb_kernel, label="ogb kernel")
ax.plot(gcamp_kernel, label="gcamp kernel")
ax.set_xlabel("time [s]")
ax.set_ylabel("exponetial kernel")
ax.legend()


# In[26]:


# OGB
fig, axs = plt.subplots(
    3,
    1,
    figsize=(6, 4),
    height_ratios=[1, 1, 1],
    gridspec_kw=dict(hspace=0),
    sharex=True,
)

axs[0].plot(time_ogb_dec, calcium_ogb_cell_dec, label="ogb")
axs[1].plot(time_ogb_dec, deconv_calcium_ogb_cell, label="ogb dec")
axs[2].plot(time_ogb_dec, spike_ogb_cell_dec, label="ogb spike")
axs[2].set_xlabel("time [s]")
axs[0].set_ylabel("delta F/F")
axs[1].set_ylabel("deconvolved delta F/F")
axs[2].set_ylabel("Rate [Hz]")
fig.align_labels()


# In[28]:


# GCAMP
fig, axs = plt.subplots(
    3,
    1,
    figsize=(6, 4),
    height_ratios=[1, 1, 1],
    gridspec_kw=dict(hspace=0),
    sharex=True,
)
axs[0].plot(time_gcamp_dec, calcium_gcamp_cell_dec, label="ogb")
axs[1].plot(time_gcamp_dec, deconv_calcium_gcamp_cell, label="ogb dec")
axs[2].plot(time_gcamp_dec, spike_gcamp_cell_dec, label="ogb spike")
axs[2].set_xlabel("time [s]")
axs[0].set_ylabel("delta F/F")
axs[1].set_ylabel("deconvolved delta F/F")
axs[2].set_ylabel("Rate [Hz]")
fig.align_labels()


# ## Task 3: Run more complex algorithm
# 
# As reviewed in the lecture, a number of more complex algorithms for inferring spikes from calcium traces have been developed. Run an implemented algorithm on the data and plot the result. There is a choice of algorithms available, for example:
# 
# * Vogelstein: [oopsi](https://github.com/liubenyuan/py-oopsi)
# * Theis: [c2s](https://github.com/lucastheis/c2s)
# * Friedrich: [OASIS](https://github.com/j-friedrich/OASIS)
# 
# *Grading: 2 pts*
# 
# 

# In[30]:


import oasis

c, s, b, g, lam = oasis.functions.deconvolve(calcium_ogb_cell_dec)


# In[36]:


fig, axs = plt.subplots(
    3,
    1,
    figsize=(6, 5),
    height_ratios=[1, 1, 1],
    gridspec_kw=dict(hspace=0),
    sharex=True,
)

axs[0].plot(time_ogb_dec, c, label="ogb_oasis")
axs[1].plot(time_ogb_dec, s, label="spike_oasis")
axs[2].plot(time_ogb_dec, spike_ogb_cell_dec, label="ogb spike")
axs[2].set_xlabel("time [s]")
axs[0].set_ylabel("denoised delta F/F")
axs[1].set_ylabel("Rate [Hz]")
axs[2].set_ylabel("Rate [Hz]")
axs[2].set_xlim(0, 100)
fig.align_labels()
# OGB Cell


# In[40]:


c, s, b, g, lam = oasis.functions.deconvolve(calcium_gcamp_cell_dec)


# In[42]:


fig, axs = plt.subplots(
    3,
    1,
    figsize=(6, 5),
    height_ratios=[1, 1, 1],
    gridspec_kw=dict(hspace=0),
    sharex=True,
)

axs[0].plot(time_gcamp_dec, c, label="ogb_oasis")
axs[1].plot(time_gcamp_dec, s, label="spike_oasis")
axs[2].plot(time_gcamp_dec, spike_gcamp_cell_dec, label="ogb spike")
axs[2].set_xlabel("time [s]")
axs[0].set_ylabel("denoised delta F/F")
axs[1].set_ylabel("Rate [Hz]")
axs[2].set_ylabel("Rate [Hz]")
axs[2].set_xlim(0, 100)
fig.align_labels()


# ## Task 4: Evaluation of algorithms
# 
# To formally evaluate the algorithms on the two datasets run the deconvolution algorithm and the more complex one on all cells and compute the correlation between true and inferred spike trains. `DataFrames` from the `pandas` package are a useful tool for aggregating data and later plotting it. Create a dataframe with columns
# 
# * algorithm
# * correlation
# * indicator
# 
# and enter each cell. Plot the results using `stripplot` and/or `boxplot` in the `seaborn` package.
# 
# *Grading: 3 pts*
# 

# Evaluate on OGB data

# In[ ]:

algo = []
c = []
indicator = []

for cell in range(ogb_calcium.shape[1]):
    # get the current cell out of the dataframe
    cell_calcium_data = ogb_calcium[f"{cell}"].to_numpy()[~np.isnan(ogb_calcium[f"{cell}"].to_numpy())]
    cell_spike_data = ogb_spikes[f"{cell}"].to_numpy()[~np.isnan(ogb_spikes[f"{cell}"].to_numpy())]
    # decimate the data
    ogb_calcium_dec = signal.decimate(cell_calcium_data, int(downsaple_factor))
    spike_ogb_cell_dec = signal.decimate(cell_spike_data, int(downsaple_factor))
    # caluclate the time array
    time_ogb_dec = np.arange(0, len(ogb_calcium_dec)) / samplerate

    # deconvolve the data with our function
    deconv_calcium_ogb_cell, ogb_kernel = deconv_ca(ogb_calcium_dec, 0.5, samplerate)
    # deconvolve the data with oasis
    ca, s, b, g, lam = oasis.functions.deconvolve(ogb_calcium_dec)

    # calulate the correlation between spike_ogb_cell_dec and deconv_calcium_ogb_cell
    c.append(np.corrcoef(spike_ogb_cell_dec, deconv_calcium_ogb_cell)[0, 1])
    algo.append("Our_Algorithm")
    indicator.append("OGB")

    # calulate the correlation between spike_ogb_cell_dec and oasis
    c.append(np.corrcoef(spike_ogb_cell_dec, s)[0, 1])
    algo.append("Oasis")
    indicator.append("OGB")
embed()
exit()
# -------------------------------------------------
# Create dataframe for OGB Cell as described (1 pt)
# -------------------------------------------------


# Create OGB dataframe

# In[ ]:


df_ogb = pd.DataFrame({"algorithm": algo, "correlation": c, "indicator": indicator})
df_ogb.head()


# Evaluate on GCamp data

# In[ ]:


# ---------------------------------------------------
# Create dataframe for GCamP Cell as described (1 pt)
# ---------------------------------------------------


# Create GCamp dataframe

# In[ ]:


df_gcamp = pd.DataFrame({"algorithm": algo, "correlation": c, "indicator": indicator})
df_gcamp.head()


# Combine both dataframes and plot

# In[ ]:


# ---------------------------------------------------------------------------
# Create Strip/Boxplot for both cells and algorithms Cell as described (1 pt)
# hint: you can seperate the algorithms by color
# ---------------------------------------------------------------------------

