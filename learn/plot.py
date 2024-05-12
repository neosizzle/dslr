import matplotlib.pyplot as plt
import sys

import model as ft_model

houses_len = len(ft_model.HOUSES)
features_len = len(ft_model.SELECTED_FEATURES)

def init_mpl():
	plt.style.use('_mpl-gallery')
	fig, ax = plt.subplots(houses_len, features_len, figsize=(houses_len * 7, features_len * 3))
	plt.tight_layout(pad=3)

	# set metadata
	for i in range(houses_len):
		for j in range(features_len):
			ax[i][j].set_title(f"{ft_model.HOUSES[i]}_{ft_model.SELECTED_FEATURES[j]}")

	return (fig, ax)

def plot_index(index, ax, plot_data, x_regress, y_regress):
	house_idx = index // features_len
	feat_idx = index % features_len
	ax[house_idx][feat_idx].scatter(plot_data[0], plot_data[1], alpha=0.6)
	ax[house_idx][feat_idx].plot(x_regress, y_regress, alpha=0.6)

def save_fig():
	plt.savefig(f"{sys.argv[0]}.png")