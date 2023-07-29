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

    # insert your code here
    f = np.exp(
        alpha + kappa * (np.cos(2 * (theta - phi)) - 1) + nu * (np.cos(theta - phi) - 1)
    )

    return f


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
    upper_bounds = (np.inf, np.inf, np.inf, 360)
    lower_bounds = (0, 0, 0, 0)
    bounds = (lower_bounds, upper_bounds)

    # try:
    #     popt, pcov = opt.curve_fit(vonMises, dirs, counts, maxfev=1000)
    # except RuntimeError:
    popt, pcov = opt.curve_fit(
        vonMises, dirs, counts, maxfev=1000000, bounds=bounds, method="trf"
    )

    x = np.arange(0, 360, 1)

    y = vonMises(x, *popt)

    if show is True:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(dirs, counts, "o", label="data")
        ax.plot(x, y, label="fit")
        ax.set_xlabel("Direction (degree)")
        ax.set_ylabel("Spike Count")
        plt.legend()

        return
    else:
        return popt


def get_spike_counts_per_orientation(data, spike_data, roi):
    # spike count for one roi for each orientation
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
    # spike count for one roi for each orientation
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
    # print(p)

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
    """Calculate the mean/std dff for each orientation for each roi.

    Parameters
    ----------
    data : dict
        The data dictionary. See `load_data` for details.

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


def dff_orientation_temporal_frequency(data:dict):
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
    temporal_frequencies = np.sort(temporal_frequencies[~np.isnan(temporal_frequencies)])

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
