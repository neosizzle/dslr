import sys
import csv
import matplotlib.pyplot as plt
import numpy as np

DATA_MODEL = [
    {"name": "Index", "idx": 0, "type": "int"},
    {"name": "Hogwarts House", "idx": 1, "type": "string"},
    {"name": "First Name", "idx": 2, "type": "string"},
    {"name": "Last Name", "idx": 3, "type": "string"},
    {"name": "Birthday", "idx": 4, "type": "string"},
    {"name": "Best Hand", "idx": 5, "type": "string"},
    {"name": "Arithmancy", "idx": 6, "type": "float"},
    {"name": "Astronomy", "idx": 7, "type": "float"},
    {"name": "Herbology", "idx": 8, "type": "float"},
    {"name": "Defense Against the Dark Arts", "idx": 9, "type": "float"},
    {"name": "Divination", "idx": 10, "type": "float"},
    {"name": "Muggle Studies", "idx": 11, "type": "float"},
    {"name": "Ancient Runes", "idx": 12, "type": "float"},
    {"name": "History of Magic", "idx": 13, "type": "float"},
    {"name": "Transfiguration", "idx": 14, "type": "float"},
    {"name": "Potions", "idx": 15, "type": "float"},
    {"name": "Care of Magical Creatures", "idx": 16, "type": "float"},
    {"name": "Charms", "idx": 17, "type": "float"},
    {"name": "Flying", "idx": 18, "type": "float"}
]

HOUSES = [
	"Ravenclaw",
	"Slytherin",
	"Gryffindor",
	"Hufflepuff",
]

def sum(data) :
	total = 0
	for val in data:
		total += val
	return total

def mean(values) :
	return sum(values) / len(values)

def count(values) :
	return len(values)

def stddev(data, ddof=0):
    mean_data = mean(data)

    squared_diffs = [(x - mean_data) ** 2 for x in data]

    # Calculate the average of the squared differences
    variance = sum(squared_diffs) / (len(squared_diffs) - ddof)

    stddev = variance ** 0.5
    return stddev

def min(data):
	min = sys.maxsize
	for val in data :
		if val < min :
			min = val
	return min

def max(data):
	max = -sys.maxsize - 1
	for val in data :
		if val > max :
			max = val
	return max

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

# match models and return appropriate type
def match_types(data, type) :
	if type == "float":
		try:
			return float(data)
		except:
			return 0.0
	if type == "int":
		try:
			return int(data)
		except:
			return 0
	return data

def get_house_color(house) :
	if house == "Ravenclaw":
		return "blue"
	if house == "Slytherin":
		return "green"
	if house == "Gryffindor":
		return "orange"
	if house == "Hufflepuff":
		return "red"

# read and parse csv
# [[indices], [houses], [name] ... ]
def read_csv(filepath) :
	res = []
	for type in DATA_MODEL :
		res.append([])
	file=open(filepath, "r")
	reader = csv.reader(file)
	skip_first_line = True
	for line in reader:
		t=line

		if skip_first_line:
			skip_first_line = False
			continue

		for model in DATA_MODEL:
			idx = model["idx"]
			type = model["type"]
			data = match_types(line[idx], type)
			res[idx].append(data)

	return res

def plot_histogram(data, ax):
	feature_idx_offset = 6
	histo_matrix = []
	for house in HOUSES:
		histo_matrix.append([])

	for idx, house in enumerate(histo_matrix):
		for i in range(13):
			histo_matrix[idx].append([])

	# iterate through all rows of data via idx, this will populate histo_matrix
	for row_idx in data[0]:
		# if row_idx == 5:
		# 	break

		# get house
		house = data[1][row_idx]

		# iterate through all enumerable features
		for model in DATA_MODEL :
			if model["type"] != "float" :
				continue
			feature_value = data[model["idx"]][row_idx]
			house_idx = HOUSES.index(house)
			feature_idx = model["idx"] - feature_idx_offset
			histo_matrix[house_idx][feature_idx].append(feature_value)
			# print(f"name: {feature_name}, value: {feature_value}")
	
	for house_idx, house_data in enumerate(histo_matrix) :
		house_name = HOUSES[house_idx]
		house_color = get_house_color(house_name)	
		
		for feature_idx, feature_data in enumerate(histo_matrix[house_idx]):
			model_idx = feature_idx + feature_idx_offset
			feature_name = DATA_MODEL[model_idx]['name']
			title = f"{house_name}: {feature_name}"
			ax[feature_idx].set_title(feature_name)

			x_values = np.arange(len(feature_data))
			y_values = feature_data
			hist, bins = generate_historgram(y_values)
			center = get_centers(bins)
			width = 1 * (bins[1] - bins[0])
			ax[feature_idx].bar(center, hist, width=width, color=house_color, alpha=0.5)

			print(f"{house_name}: {DATA_MODEL[model_idx]['name']}")	 

def main():
	if len(sys.argv) != 2:
		print("Usage: python describe_csv.py [filename]")
		return
	# init matplotlib
	plt.style.use('_mpl-gallery')
	fig, ax = plt.subplots(1, 13, figsize=(205, 25))
	# ax.set_title("Histogram of all features")
	plt.tight_layout(pad=2)


	data = read_csv(sys.argv[1])
	plot_histogram(data, ax)
	# print(data)

	# export matplotlib as png
	plt.savefig(f"{sys.argv[0]}.png")
main()