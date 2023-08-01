# q = (2.5, 97.5)
# (
#     median_spikes,
#     spike_orientation_percentile_lower,
#     spike_orientation_percentile_upper,
# ) = utils.spike_orientation_median(data, spike_data, q)

# # plot a polar plot for for each orientation
# roi = 54
# fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})

# ax.plot(
#     orientations_polar_plot,
#     median_spikes[roi, :],
# )


# ax.fill_between(
#     orientations_polar_plot,
#     spike_orientation_percentile_lower[roi, :],
#     spike_orientation_percentile_upper[roi, :],
#     alpha=0.5,
#     zorder=-1,
# )

# print(orientations_polar_plot)
# print(
#     spike_orientation_percentile_lower[roi, :],
#     spike_orientation_percentile_upper[roi, :],
# )
# ax.set_title(f"ROI {roi}")

# (
#     median_spikes_temp,
#     lower_spikes_temp,
#     upper_spikes_temp,
# ) = utils.spike_orientation_temporal_median(data, spike_data, q)

# fig, axs = plt.subplots(
#     1, 5, subplot_kw={"projection": "polar"}, figsize=(20, 10), sharey=True
# )
# roi = 67
# plt.subplots_adjust(wspace=0.3)
# for i, ax in enumerate(axs.flat):
#     ax.plot(
#         orientations_polar_plot,
#         median_spikes_temp[roi, :, i],
#     )
#     # add std to the plot
#     ax.fill_between(
#         orientations_polar_plot,
#         lower_spikes_temp[roi, :, i],
#         upper_spikes_temp[roi, :, i],
#         alpha=0.6,
#     )
#     ax.set_title(f"{temporal_frequencies[i]} Hz")


# orientations = np.sort(data["stim_table"]["orientation"].unique())
# colors_orentation = np.zeros(len(data["stim_table"]["orientation"]), dtype=int)
# for i, ori in enumerate(orientations):
#     if i == 8:
#         colors_orentation[data["stim_table"]["blank_sweep"] == 1] = i
#     colors_orentation[data["stim_table"]["orientation"] == ori] = i

# rois_plotting = 3

# fig, axs = plt.subplots(len(rois[:rois_plotting]), figsize=(20, 10), sharex=True)
# plt.subplots_adjust(hspace=0.3)
# limits = [0, 300]
# for i, roi in enumerate(rois[:rois_plotting]):
#     axs[i].plot(data["t"], data["dff"][roi], c="k")
#     for indx, (s, e) in enumerate(
#         zip(
#             data["t"][np.array(data["stim_table"]["start"], dtype=int)],
#             data["t"][np.array(data["stim_table"]["end"], dtype=int)],
#         )
#     ):
#         axs[i].axvspan(
#             s,
#             e,
#             alpha=0.6,
#             color=colors_list[colors_orentation[indx]],
#         )
#     orientation_patches = [
#         mp.Rectangle(
#             (0, 0),
#             1,
#             1,
#             fc=colors_list[i],
#             alpha=0.6,
#             angle=orientations[i],
#         )
#         for i in range(9)
#     ]

#     axs[i].set_xlim(limits)
#     axs[i].set_title(f"ROI {roi}")


# axs[0].legend(
#     orientation_patches,
#     orientations,
#     bbox_to_anchor=(0.2, 1.32),
#     loc="center left",
#     borderaxespad=-3,
#     fontsize=12,
#     title="Orientation (degrees)",
#     title_fontsize=12,
#     ncol=9,
# )

# # plot a polar plot for for each orientation
# for roi in rois[:rois_plotting]:
#     fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
#     ax.plot(
#         orientations_polar_plot,
#         median_spikes[roi, :],
#     )
#     print(median_spikes[roi, :])
#     print(spike_orientation_percentile_lower[roi, :])
#     print(spike_orientation_percentile_upper[roi, :])
#     # add std to the plot
#     ax.fill_between(
#         orientations_polar_plot,
#         spike_orientation_percentile_lower[roi, :],
#         spike_orientation_percentile_upper[roi, :],
#         alpha=1,
#     )
#     ax.set_yticks(np.arange(0, 5, 1))
#     ax.set_title(f"ROI {roi}")