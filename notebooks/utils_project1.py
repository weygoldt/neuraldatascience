import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import seaborn as sns


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

    f = np.exp(
        alpha + kappa * (np.cos(2 * (theta - phi)) - 1) + nu * (np.cos(theta - phi) - 1)
    )

    return f


def tuningCurve(counts, dirs, show=True, tile_name=""):
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
    x: np.array, shape=(360, )
        x-axis for plotting
    y: np.array, shape=(360, )
        y-axis for plotting
    """

    # insert your code here
    upper_bounds = (np.inf, np.inf, np.inf, 360)
    lower_bounds = (0, 0, 0, 0)
    bounds = (lower_bounds, upper_bounds)

    try:
        popt, pcov = opt.curve_fit(vonMises, dirs, counts, maxfev=1000)
    except RuntimeError:
        popt, pcov = opt.curve_fit(
            vonMises, dirs, counts, maxfev=1000000, bounds=bounds, method="trf"
        )

    x = np.arange(0, 360, 1)

    y = vonMises(x, *popt)
    if y.max() > 100:
        return None, None, None

    if show is True:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(dirs, counts, "o", label="data")
        ax.plot(x, y, label="fit")
        ax.set_xlabel("Direction (degree)")
        ax.set_ylabel("Spike Count")
        ax.set_title(tile_name)
        plt.legend()

    else:
        return popt, x, y


def get_spike_counts_per_orientation(data, spike_data, roi):
    """
    Compute the spike count for a given region of interest (ROI) for each orientation in the stimulus table.

    Parameters
    ----------
    data : dict
        A dictionary containing the stimulus table with columns "start", "end", and "orientation".
    spike_data : numpy.ndarray
        A array of spike data with shape (num_rois, num_samples).
    roi : int
        The index of the ROI for which to compute the spike counts.

    Returns
    -------
    dirs : numpy.ndarray
        A array of orientation values in degrees, sorted in ascending order.
    counts : numpy.ndarray
        A array of spike counts for the given ROI, corresponding to each orientation in `dirs`.
    """

    dirs = []
    counts = []
    for i, row in enumerate(data["stim_table"].iterrows()):
        ori = row[1]["orientation"]
        if np.isnan(ori):
            continue
        dirs.append(ori)
        start_times = row[1]["start"].astype(int)
        end_times = row[1]["end"].astype(int)
        counts.append(spike_data[roi, start_times:end_times].sum())
    idx = np.argsort(dirs)
    dirs = np.array(dirs)[idx]
    counts = np.array(counts)[idx]

    return dirs, counts


def get_spike_counts_per_orientation_temporalfreq(data, spike_data, roi, temporal_freq):
    """
    Compute the spike count for a given region of interest (ROI) and each orientation of a visual stimulus
    with a given temporal frequency.

    Parameters
    ----------
    data : dict
        A dictionary containing the stimulus table and metadata.
    spike_data : numpy.ndarray
        A array of spike counts, where the first dimension corresponds to the ROI and the second dimension
        corresponds to time bins.
    roi : int
        The index of the ROI for which to compute the spike counts.
    temporal_freq : float
        The temporal frequency of the visual stimulus for which to compute the spike counts.

    Returns
    -------
    dirs : numpy.ndarray
        An array of orientation values in degrees, sorted in ascending order.
    counts : numpy.ndarray
        An array of spike counts, where each element corresponds to the spike count for the corresponding
        orientation value in `dirs`.
    """

    dirs = []
    counts = []
    for i, row in enumerate(
        data["stim_table"][
            data["stim_table"]["temporal_frequency"] == temporal_freq
        ].iterrows()
    ):
        ori = row[1]["orientation"]
        if np.isnan(ori):
            continue
        dirs.append(ori)
        start_times = row[1]["start"].astype(int)
        end_times = row[1]["end"].astype(int)
        counts.append(spike_data[roi, start_times:end_times].sum())
    idx = np.argsort(dirs)
    dirs = np.array(dirs)[idx]
    counts = np.array(counts)[idx]

    return dirs, counts


def testTuning(counts, dirs, psi=1, niters=1000, show=False, title_name=""):
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
    np.random.seed(42)
    dirs_unique = np.unique(dirs)
    means = np.zeros(len(dirs_unique))
    for d, directions in enumerate(np.unique(dirs)):
        means[d] = np.mean(counts[dirs == directions])

    # imaginary exponential function mu
    dirs_unique_rad = np.deg2rad(dirs_unique)

    nu = np.exp((1j * psi * dirs_unique_rad) * (2 * np.pi))

    q = means @ nu

    abs_q = np.absolute(q)
    if abs_q == 0:
        return 1, abs_q, np.array([0.0] * niters)
    counts_shuffle = np.array(counts)
    qs_shuffle = np.zeros(niters)
    valid_counter = 0

    for i in range(niters):
        np.random.shuffle(counts_shuffle)
        means = np.zeros(len(dirs_unique))
        for d, directions in enumerate(np.unique(dirs)):
            means[d] = np.mean(counts_shuffle[dirs == directions])

        q_shuffle = means @ nu
        abs_q_shuffle = np.absolute(q_shuffle)
        qs_shuffle[i] = abs_q_shuffle

        if abs_q_shuffle > abs_q:
            valid_counter += 1

    p = valid_counter / niters

    if show is True:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(qs_shuffle, bins=100, stat="proportion", ax=ax, label="null")
        ax.axvline(abs_q, color="red", label="observed")
        ax.set_xlabel("|q|")
        ax.set_ylabel("Fraction of Runs")
        ax.legend()
        ax.set_title(f"{title_name}")

        # you can use sns.histplot for the histogram
    else:
        return p, abs_q, qs_shuffle


def dff_orientation(data: dict):
    """Calculates the average respones in all trials and takes the mean/std dff for each orientation for each roi.

    Parameters
    ----------
    data : dict
        The data dictionary.

    Returns
    -------
    mean_calcium_orientation : np.ndarray (n_rois, n_orientations)
        The mean dff for each orientation for each roi.
    std_calcium_orientation : np.ndarray (n_rois, n_orientations)
        The standard deviation of the dff for each orientation for each roi.
    """

    orientations = data["stim_table"]["orientation"].unique()
    orientations = orientations[~np.isnan(orientations)]
    mean_calcium_orientation = np.zeros((data["dff"].shape[0], len(orientations)))
    std_calcium_orientation = np.zeros((data["dff"].shape[0], len(orientations)))

    for i, orientation in enumerate(np.sort(orientations)):
        # define the start and end times for each orientation
        start_times = (
            data["stim_table"]["start"][
                data["stim_table"]["orientation"] == orientation
            ]
            .to_numpy()
            .astype(int)
        )
        end_times = (
            data["stim_table"]["end"][data["stim_table"]["orientation"] == orientation]
            .to_numpy()
            .astype(int)
        )

        for roi in range(data["dff"].shape[0]):
            # calculate the mean dff for each orientation for each roi
            mean_calcium_orientation[roi, i] = np.mean(
                [
                    np.mean(data["dff"][roi, s:e])
                    for s, e in zip(start_times, end_times)
                ],
                axis=0,
            )
            # calculate the std dff for each orientation for each roi
            std_calcium_orientation[roi, i] = np.std(
                [
                    np.mean(data["dff"][roi, s:e])
                    for s, e in zip(start_times, end_times)
                ],
                axis=0,
            )
    return mean_calcium_orientation, std_calcium_orientation


def dff_orientation_temporal_frequency(data: dict):
    """Calculate the mean/std dff for each orientation for each roi with respect to
    the temporal frequency.

    Parameters
    ----------
    data : dict
        The data dictionary. See `load_data` for details.

    Returns
    -------
    mean_calcium_orientation : np.ndarray (n_rois, n_orientations, n_temporal_frequencies)
        The mean dff for each orientation and temporal frequency for each roi.
    std_calcium_orientation : np.ndarray (n_rois, n_orientations, n_temporal_frequencies)
        The standard deviation of the dff for each orientation and temporal frequency
            for each roi.
    """

    # get the unique orientations and temporal frequencies
    orientations = data["stim_table"]["orientation"].unique()
    orientations = np.sort(orientations[~np.isnan(orientations)])
    temporal_frequencies = data["stim_table"]["temporal_frequency"].unique()
    temporal_frequencies = np.sort(
        temporal_frequencies[~np.isnan(temporal_frequencies)]
    )

    mean_calcium_orientation = np.zeros(
        (data["dff"].shape[0], len(orientations), len(temporal_frequencies))
    )
    std_calcium_orientation = np.zeros(
        (data["dff"].shape[0], len(orientations), len(temporal_frequencies))
    )

    for i, orientation in enumerate(orientations):
        for j, freq in enumerate(temporal_frequencies):
            start_times = (
                data["stim_table"]["start"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            end_times = (
                data["stim_table"]["end"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            for roi in range(data["dff"].shape[0]):
                mean_calcium_orientation[roi, i, j] = np.mean(
                    [
                        np.mean(data["dff"][roi, s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                )
                std_calcium_orientation[roi, i, j] = np.std(
                    [
                        np.mean(data["dff"][roi, s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                )
    return mean_calcium_orientation, std_calcium_orientation


def spike_orientation_mean(data: dict, spike_data):
    """
    Computes the mean and standard deviation of the spike counts for each ROI and orientation in the stimulus table.

    Parameters:
    -----------
    data : dict
        A dictionary containing the following keys:
        - "stim_table": a pandas DataFrame with columns "start", "end", and "orientation", representing the stimulus presentation times and orientations.
        - "dff": a numpy array of shape (n_rois, n_frames), representing the fluorescence signals of the ROIs.
    spike_data : list of numpy arrays
        A list of numpy arrays of shape (n_frames,), representing the spike counts of each ROI.

    Returns:
    --------
    mean_spike_orientation : numpy array
        A numpy array of shape (n_rois, n_orientations), representing the mean spike counts for each ROI and orientation.
    std_spike_orientation : numpy array
        A numpy array of shape (n_rois, n_orientations), representing the standard deviation of the spike counts for each ROI and orientation.
    """

    orientations = data["stim_table"]["orientation"].unique()
    orientations = orientations[~np.isnan(orientations)]
    mean_spike_orientation = np.zeros((data["dff"].shape[0], len(orientations)))
    std_spike_orientation = np.zeros((data["dff"].shape[0], len(orientations)))

    for i, orientation in enumerate(np.sort(orientations)):
        start_times = (
            data["stim_table"]["start"][
                data["stim_table"]["orientation"] == orientation
            ]
            .to_numpy()
            .astype(int)
        )
        end_times = (
            data["stim_table"]["end"][data["stim_table"]["orientation"] == orientation]
            .to_numpy()
            .astype(int)
        )
        for roi in range(data["dff"].shape[0]):
            mean_spike_orientation[roi, i] = np.mean(
                [np.sum(spike_data[roi][s:e]) for s, e in zip(start_times, end_times)],
                axis=0,
            )
            std_spike_orientation[roi, i] = np.std(
                [np.sum(spike_data[roi][s:e]) for s, e in zip(start_times, end_times)],
                axis=0,
            )
    return mean_spike_orientation, std_spike_orientation


def spike_orientation_median(data: dict, spike_data: np.ndarray, q: int) -> tuple:
    """
    Calculates the median and percentile spike orientation for each ROI in the given spike data.

    Parameters
    ----------
    data : dict
        A dictionary containing the stimulus table and dff data.
    spike_data : numpy.ndarray
        A numpy array containing spike data for each ROI.
    q : int
        The percentile value to calculate.

    Returns
    -------
    tuple
        A tuple containing three numpy arrays:
            - median_spike_orientation: The median spike orientation for each ROI and orientation.
            - spike_orientation_percentile_5: The 5th percentile spike orientation for each ROI and orientation.
            - spike_orientation_percentile_95: The 95th percentile spike orientation for each ROI and orientation.

    """

    orientations = data["stim_table"]["orientation"].unique()
    orientations = orientations[~np.isnan(orientations)]
    median_spike_orientation = np.zeros((data["dff"].shape[0], len(orientations)))
    spike_orientation_percentile_95 = np.zeros(
        (data["dff"].shape[0], len(orientations))
    )
    spike_orientation_percentile_5 = np.zeros((data["dff"].shape[0], len(orientations)))

    for i, orientation in enumerate(np.sort(orientations)):
        start_times = (
            data["stim_table"]["start"][
                data["stim_table"]["orientation"] == orientation
            ]
            .to_numpy()
            .astype(int)
        )
        end_times = (
            data["stim_table"]["end"][data["stim_table"]["orientation"] == orientation]
            .to_numpy()
            .astype(int)
        )
        for roi in range(data["dff"].shape[0]):
            median_spike_orientation[roi, i] = np.median(
                [np.sum(spike_data[roi][s:e]) for s, e in zip(start_times, end_times)],
                axis=0,
            )
            percentiles = np.percentile(
                [np.sum(spike_data[roi][s:e]) for s, e in zip(start_times, end_times)],
                axis=0,
                q=q,
            )
            spike_orientation_percentile_5[roi, i] = percentiles[0]
            spike_orientation_percentile_95[roi, i] = percentiles[1]
    return (
        median_spike_orientation,
        spike_orientation_percentile_5,
        spike_orientation_percentile_95,
    )


def spike_orientation_mean_temporal(data: dict, spike_data):
    """
    Computes the mean and standard deviation of spike counts for each orientation and temporal frequency.

    Parameters
    ----------
    data : dict
        A dictionary containing the stimulus table and dff data.
    spike_data : numpy.ndarray
        A numpy array containing spike counts for each ROI.

    Returns
    -------
    tuple
    """

    orientations = data["stim_table"]["orientation"].unique()
    orientations = np.sort(orientations[~np.isnan(orientations)])
    temporal_frequencies = data["stim_table"]["temporal_frequency"].unique()
    temporal_frequencies = np.sort(
        temporal_frequencies[~np.isnan(temporal_frequencies)]
    )

    mean_spike_orientation_freq = np.zeros(
        (data["dff"].shape[0], len(orientations), len(temporal_frequencies))
    )
    std_spike_orientation_freq = np.zeros(
        (data["dff"].shape[0], len(orientations), len(temporal_frequencies))
    )
    for i, orientation in enumerate(orientations):
        for j, freq in enumerate(temporal_frequencies):
            start_times = (
                data["stim_table"]["start"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            end_times = (
                data["stim_table"]["end"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            for roi in range(data["dff"].shape[0]):
                mean_spike_orientation_freq[roi, i, j] = np.mean(
                    [
                        np.sum(spike_data[roi][s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                )
                std_spike_orientation_freq[roi, i, j] = np.std(
                    [
                        np.sum(spike_data[roi][s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                )
    return mean_spike_orientation_freq, std_spike_orientation_freq


def spike_orientation_temporal_median(data: dict, spike_data, q):
    """
    Computes the median and percentile spike counts for each orientation and temporal frequency.
    
    Parameters
    ----------
    data : dict
        A dictionary containing the stimulus table and dff data.
    spike_data : numpy.ndarray
        A numpy array containing spike counts for each ROI.
    q : tuple
        The percentile values to calculate.

    Returns
    -------
    tuple
    """

    orientations = data["stim_table"]["orientation"].unique()
    orientations = np.sort(orientations[~np.isnan(orientations)])
    temporal_frequencies = data["stim_table"]["temporal_frequency"].unique()
    temporal_frequencies = np.sort(
        temporal_frequencies[~np.isnan(temporal_frequencies)]
    )

    median_spike_orientation_freq = np.zeros(
        (data["dff"].shape[0], len(orientations), len(temporal_frequencies))
    )
    q95 = np.zeros((data["dff"].shape[0], len(orientations), len(temporal_frequencies)))
    q5 = np.zeros((data["dff"].shape[0], len(orientations), len(temporal_frequencies)))

    for i, orientation in enumerate(orientations):
        for j, freq in enumerate(temporal_frequencies):
            start_times = (
                data["stim_table"]["start"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            end_times = (
                data["stim_table"]["end"][
                    (data["stim_table"]["orientation"] == orientation)
                    & (data["stim_table"]["temporal_frequency"] == freq)
                ]
                .to_numpy()
                .astype(int)
            )
            for roi in range(data["dff"].shape[0]):
                median_spike_orientation_freq[roi, i, j] = np.mean(
                    [
                        np.sum(spike_data[roi][s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                )
                percentiles = np.percentile(
                    [
                        np.sum(spike_data[roi][s:e])
                        for s, e in zip(start_times, end_times)
                    ],
                    axis=0,
                    q=q,
                )
                q5[roi, i, j] = percentiles[0]
                q95[roi, i, j] = percentiles[1]

    return median_spike_orientation_freq, q5, q95


def smooth_rate(data, instant_firing_rate, window):
    """
    Smooth the firing rate by taking the average of the firing rate over a window.

    Parameters
    ----------
    data : dict
        The data dictionary. See `load_data` for details.
    instant_firing_rate : np.ndarray (n_orientations, n_timepoints)
        The instantaneous firing rate for each orientation for each roi.
    window : int
        The size of the window to smooth over.

    Returns
    -------
    smoothed_rate : np.ndarray (n_orientations, n_timepoints)
        The smoothed firing rate for each orientation for each roi.
    """
    # create matrix filled with nans
    smoothed_rate = np.empty((len(instant_firing_rate), len(instant_firing_rate[0])))
    smoothed_rate.fill(np.nan)
    # define the step size
    step_size = int(np.floor(window / 2))
    # define the length of the window to smooth over
    length = np.arange(
        (window - step_size), len(instant_firing_rate[0]) - step_size + 1
    )
    # get the unique orientations sorted
    orientations = data["stim_table"]["orientation"].unique()
    orientations = np.sort(orientations[~np.isnan(orientations)])

    # calculate the mean of the smoothing window, save it to the matrix and
    # then shift the window by step_size
    for i in range(len(instant_firing_rate)):
        for j in range(len(length)):
            smoothed_rate[i, j] = np.nanmean(
                instant_firing_rate[orientations[i]][(j - step_size) : (j + step_size)]
            )
    return smoothed_rate
