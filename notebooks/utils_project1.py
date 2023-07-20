import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
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

    if show == True:
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
    np.random.seed(42)
    dirs_unique = np.unique(dirs)
    means = np.zeros(len(dirs_unique))
    for d, directions in enumerate(np.unique(dirs)):
        means[d] = np.mean(counts[dirs == directions])

    # imaginary exponential function mu
    dirs_unique_rad = np.deg2rad(dirs_unique)

    nu = np.exp((1j * psi * dirs_unique_rad) * (2 * np.pi))

    q = means @ nu
    if q.imag != 0:
        print("Warning: q is not real")
        

    abs_q = np.absolute(q)
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

    if show == True:
        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(qs_shuffle, bins=100, stat="proportion", ax=ax, label="null")
        ax.axvline(abs_q, color="red", label="observed")
        ax.set_xlabel("|q|")
        ax.set_ylabel("Fraction of Runs")

        # you can use sns.histplot for the histogram
    else:
        return p, abs_q, qs_shuffle
