import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import random

categories = ["shortking", "mid", "lbj"]

class Data:
	def __init__(self, category, height, id, midscore):
		self.category = category
		self.height = height,
		self.id = id
		self.midscore = midscore
	
	def get_feature(self, feature):
		if feature == "height":
			return self.height[0]
		if feature == "midscore":
			return self.midscore
		return None

def sigmoid(z) :
	# print(z)
	return 1 / (1 + math.exp(z * -1))

def sigmoid_reverse(z) :
	# print(z)
	return math.exp(z) / (1 + math.exp(z))

def linear(m, x, c):
	return (x * m) + c

def derivative_m(y_binary, x, curr_m, curr_c):
	# print(f"linear: {linear(curr_m, x, curr_c)}")
	return (sigmoid(linear(curr_m, x, curr_c)) - y_binary) * x

def derivative_c(y_binary, x, curr_m, curr_c):
	return (sigmoid(linear(curr_m, x, curr_c)) - y_binary)

def step_size_out_of_range(number, range) :
	return number < -range or number > range


def gradient_descent_range(init_m, init_c, learning_rate, x_values, y_values):
	step_size_m = 6969
	step_size_c = 6969
	step_size_range = 0.0001
	steps = 0

	while step_size_out_of_range(step_size_m, step_size_range) or step_size_out_of_range(step_size_c, step_size_range) :
		should_change_m = False
		should_change_c = False
		total_deriv_m = 0
		total_deriv_c = 0
		steps += 1

		if step_size_out_of_range(step_size_m, step_size_range) :
			for (index, x) in enumerate(x_values):
				total_deriv_m += derivative_m(y_values[index], x, init_m, init_c)
			total_deriv_m /= len(x_values)
			step_size_m = learning_rate * total_deriv_m
			should_change_m = True


		if step_size_out_of_range(step_size_c, step_size_range) :
			for (index, x) in enumerate(x_values):
				total_deriv_c += derivative_c(y_values[index], x, init_m, init_c)
			total_deriv_c /= len(x_values)
			step_size_c = learning_rate * total_deriv_c
			should_change_c = True
	
		
		# update b0 and b1 babsed on stepsize
		if should_change_m:
			init_m = init_m - step_size_m
		if should_change_c:
			init_c = init_c - step_size_c

	# print(f"total_deriv_m: {total_deriv_m}, total_deriv_c: {total_deriv_c}")
	# print(f"step_size_m: {step_size_m}, step_size_c: {step_size_c}")
	print(f"GD m {init_m} c {init_c}, steps {steps}")
	return (init_m, init_c)

def gradient_descent_batch(init_m, init_c, learning_rate, _x_values, _y_values):
	step_size_m = 6969
	step_size_c = 6969
	step_size_range = 0.0001
	steps = 0

	sample_indices = random.sample(range(0, len(_x_values)), round(len(_x_values) / 2))
	x_values = []
	y_values = []
	for index in sample_indices:
		x_values.append(_x_values[index])
		y_values.append(_y_values[index])

	while step_size_out_of_range(step_size_m, step_size_range) or step_size_out_of_range(step_size_c, step_size_range) :
		should_change_m = False
		should_change_c = False
		total_deriv_m = 0
		total_deriv_c = 0
		steps += 1

		if step_size_out_of_range(step_size_m, step_size_range) :
			for (index, x) in enumerate(x_values):
				total_deriv_m += derivative_m(y_values[index], x, init_m, init_c)
			total_deriv_m /= len(x_values)
			step_size_m = learning_rate * total_deriv_m
			should_change_m = True


		if step_size_out_of_range(step_size_c, step_size_range) :
			for (index, x) in enumerate(x_values):
				total_deriv_c += derivative_c(y_values[index], x, init_m, init_c)
			total_deriv_c /= len(x_values)
			step_size_c = learning_rate * total_deriv_c
			should_change_c = True
	
		
		# update b0 and b1 babsed on stepsize
		if should_change_m:
			init_m = init_m - step_size_m
		if should_change_c:
			init_c = init_c - step_size_c

	# print(f"total_deriv_m: {total_deriv_m}, total_deriv_c: {total_deriv_c}")
	# print(f"step_size_m: {step_size_m}, step_size_c: {step_size_c}")
	print(f"GD m {init_m} c {init_c}, steps {steps}")
	return (init_m, init_c)

def gradient_descent(init_m, init_c, steps, learning_rate, x_values, y_values):
	for i in range(steps):
		total_deriv_m = 0
		total_deriv_c = 0
		
		for (index, x) in enumerate(x_values):
			total_deriv_m += derivative_m(y_values[index], x, init_m, init_c)
			# print(f"y_values[index]: {y_values[index]:}, x: {x}, init_m: {init_m}, init_c: {init_c}, deriv_m: {derivative_m(y_values[index], x, init_m, init_c)}")
			total_deriv_c += derivative_c(y_values[index], x, init_m, init_c)
		
		# print(f"total_deriv_m: {total_deriv_m}, total_deriv_c: {total_deriv_c}")
		total_deriv_m /= len(x_values)
		total_deriv_c /= len(x_values)
		step_size_m = learning_rate * total_deriv_m
		step_size_c = learning_rate * total_deriv_c

		init_m -= step_size_m
		init_c -= step_size_c
		print(f"step_size_m: {step_size_m}, step_size_c: {step_size_c}")
	return (init_m, init_c)
	

def normalize_value(data, min, max):
	 return ((data - min) / (max - min))

def normalize_data(data) :
	min_x = min(data)
	max_x = max(data)
	return list(map(lambda x : (x - min_x) / (max_x - min_x) , data))

def denormalize_data(data, min, max) :
	return list(map(lambda x: x * (max - min) + min, data))

def generate_data() :
	res = []
	
	res.append(Data("lbj", 180, 0, 11))
	res.append(Data("shortking", 125, 1, 4))
	res.append(Data("shortking", 120, 2, 7))
	res.append(Data("lbj", 185, 3, 14))
	res.append(Data("lbj", 190, 4, 1))
	res.append(Data("lbj", 195, 5, 5))
	res.append(Data("shortking", 110, 6, 10))
	res.append(Data("shortking", 115, 7, 12))
	res.append(Data("mid", 123, 8, 99))
	res.append(Data("mid", 122, 9, 101))
	# res.append(Data("mid", 190, 10))

	return res

# {
# 	"[category]_[featurename]": [x_values, y_values, og_x_min, og_x_max],
# 	"[category2]_[featurename2]": [x_values, y_values, og_x_min, og_x_max],
# }

def generate_data_matrix(categories, features, data):
	res = {}
	for category in categories:
		for feature in features:
			key = f"{category}_{feature}"
			feature_values = list(map(lambda x: x.get_feature(feature), data))
			# todo check for none here
			min_features = min(feature_values)
			max_features = max(feature_values)
			x_values = normalize_data(feature_values)
			y_values = list(map(lambda x: 1 if x.category == category else 0, data))
			res[key] = [x_values, y_values, min_features, max_features]
	return res

def generate_regression_line(slope, intercept, min, max, steps) :
	res = ([], [])

	step_size = max / steps
	curr_x = min
	for i in range(steps):
		res[0].append(curr_x)
		res[1].append((sigmoid_reverse(linear(slope, curr_x, intercept))))
		curr_x += step_size

	return res


def main():
	# init matplotlib
	plt.style.use('_mpl-gallery')
	fig, ax = plt.subplots(1, 6, figsize=(30, 5))
	plt.tight_layout(pad=5)

	data = generate_data()
	# x_values = normalize_data(list(map(lambda x: x.height[0], data)))
	# y_values = normalize_data(list(map(lambda x: 1 if x.is_cool else 0, data)))
	# ax[0].scatter(x_values, y_values, alpha=0.6)
	data_matrix = generate_data_matrix(categories, ['height', 'midscore'], data)

	index = 0
	weights = {}
	for key, plot_data in data_matrix.items():
		# logistic regression
		(slope, intercept) = gradient_descent_range(1, 0, 0.5, plot_data[0], plot_data[1])
		print(f"({slope}, {intercept})")

		x_regres, y_regres = generate_regression_line(slope, intercept, 0, 1, 50)
		ax[index].scatter(plot_data[0], plot_data[1], alpha=0.6)
		ax[index].plot(x_regres, y_regres, 'c', alpha=0.5)
		ax[index].set_title(key)
		
		weights[key] = [slope, intercept, plot_data[2], plot_data[3]]
		index += 1

	to_predict = Data(None, 13, 69, 200)
	
	probabilities = {}
	for category in categories :
		probabilities[category] = 1

	for key, weight_data in weights.items():
		slope = weight_data[0]
		intercept = weight_data[1]
		feature = key.split("_")[1]
		category = key.split("_")[0]
		
		normalied_x_value = normalize_value(to_predict.get_feature(feature), weight_data[2], weight_data[3])
		probability = (sigmoid_reverse(linear(slope, normalied_x_value, intercept)))
		probabilities[category] *= probability

		# print(f"{key}: {probability}")


	higest_likelihood = 0
	res = "HUH"
	for category in categories :
		if higest_likelihood < probabilities[category]:
			higest_likelihood = probabilities[category]
			res = category
	print(f"Predicted : {res}")

	# export matplotlib as png
	plt.savefig(f"{sys.argv[0]}.png")

main()