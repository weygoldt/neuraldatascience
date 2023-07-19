import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt


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
