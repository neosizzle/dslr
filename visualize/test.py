import matplotlib.pyplot as plt
import numpy as np
import sys

# https://www.dummies.com/article/academics-the-arts/math/statistics/how-to-calculate-percentiles-in-statistics-169783/
def percentile(data, percent):
	sorted_data = sorted(data)
	total_len = len(data)
	percentile_idx = percent * total_len
	
	# determine if number is whole number
	if percentile_idx == int(percentile_idx) :
		data_0 = sorted_data[int(percentile_idx)]
		data_1 = sorted_data[int(percentile_idx) + 1]
		return (data_0 + data_1) / 2
	else:
		return sorted_data[int(percentile_idx)]

def generate_historgram(data) :
	# populate bins -  hardcoded as length 10
	bin_len = 10
	bins = []
	sorted_data = sorted(data)
	step = (sorted_data[-1] - sorted_data[0]) / bin_len
	curr_bin = sorted_data[0]

	for i in range(bin_len + 1) :
		bins.append((curr_bin + (i * step)))

	hist = []
	for i in range(len(bins) - 1):
		min = bins[i]
		max = bins[i + 1]
		num_elements = len(list(filter(lambda x: x >= min and x < max, sorted_data)))
		hist.append(num_elements)
	
	# fix for last element
	max = bins[-1]
	min = bins[-2]
	hist[-1] = len(list(filter(lambda x: x == max or x > min, sorted_data)))

	return (hist, bins)

def get_centers(data) :
	res = []
	for i in range(len(data) - 1):
		min = data[i]
		max = data[i + 1]
		res.append((min + max) / 2)
	return res

np.random.seed(0)
x = 10 * np.random.randn(200)

# x = [4, 5, 3, 5, 5, 5, 5, 4.9]
# for i in range(-2, 2):
# 	x.append(i)

hist, bins = np.histogram(x, bins=10)
my_hist, my_bins = generate_historgram(x)
# print(x)
# print(f"their bin: {bins}, their histo {hist}")
# print(f"my bin: {my_bins}, my histo {my_hist}")
# print(hist)
width = 1 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
my_center = get_centers(my_bins)
# print(center)
# print(my_center)
plt.bar(center, hist, width=width)
plt.bar(my_center, my_hist, width=width, color='red', alpha=0.5)
plt.savefig(f"test.png")