import sys
import csv

from termcolor import cprint, colored

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

# https://www.youtube.com/watch?v=_vDRKlTz7yo
# tail on right = positive skew
# tail on left = negative skew
def skew(data):
	_mean = mean(data)
	median = percentile(data, 0.5)
	_stddev = stddev(data)
	return (3 * (_mean - median)) / _stddev

# match models and return appropriate type
def match_types(data, type) :
	if type == "float":
		return float(data)
	if type == "int":
		return int(data)
	return data

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

		data_row = []
		for model in DATA_MODEL:
			idx = model["idx"]
			type = model["type"]
			try:
				data = match_types(line[idx], type)
				data_row.append(data)
			except:
				continue

		if len(data_row) == len(DATA_MODEL):
			for idx, col in enumerate(data_row):
				res[idx].append(col)				
	return res


def describe_data(data) :
	for model in DATA_MODEL:
		idx = model["idx"]
		type = model["type"]
		name = model["name"]
		if type == "float":
			cprint(f"{name}", 'red')
			cprint(f"|_ {colored('Count:', 'green')} {count(data[idx])}", "red")
			cprint(f"|_ {colored('Mean:', 'green')} {mean(data[idx])}", "red")
			cprint(f"|_ {colored('Std:', 'green')} {stddev(data[idx])}", "red")
			cprint(f"|_ {colored('Min:', 'green')} {min(data[idx])}", "red")
			cprint(f"|_ {colored('25%:', 'green')} {percentile(data[idx], 0.25)}", "red")
			cprint(f"|_ {colored('50%:', 'green')} {percentile(data[idx], 0.5)}", "red")
			cprint(f"|_ {colored('75%:', 'green')} {percentile(data[idx], 0.75)}", "red")
			cprint(f"|_ {colored('Max:', 'green')} {max(data[idx])}", "red")
			cprint(f"|_ {colored('Skew:', 'green')} {skew(data[idx])}", "red")

			print()


def main():
	if len(sys.argv) != 2:
		print("Usage: python describe_csv.py [filename]")
		return
	data = read_csv(sys.argv[1])
	describe_data(data)
main()