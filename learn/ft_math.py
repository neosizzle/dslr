
import sys
import math

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

def sigmoid(z) :
	# print(z)
	return 1 / (1 + math.exp(z * -1))

def sigmoid_reverse(z) :
	# print(z)
	return math.exp(z) / (1 + math.exp(z))

def linear(m, x, c):
	return (x * m) + c