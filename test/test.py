import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

class Data:
	def __init__(self, is_cool, height, id):
		self.is_cool = is_cool
		self.height = height,
		self.id = id

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

def normalize_data(data) :
	min_x = min(data)
	max_x = max(data)
	return list(map(lambda x : (x - min_x) / (max_x - min_x) , data))

def generate_data() :
	res = []
	
	res.append(Data(True, 180, 0))
	res.append(Data(False, 125, 1))
	res.append(Data(False, 120, 2))
	res.append(Data(True, 185, 3))
	res.append(Data(True, 190, 4))
	res.append(Data(True, 195, 5))
	res.append(Data(False, 110, 6))
	res.append(Data(False, 115, 7))
	res.append(Data(False, 169, 8))

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
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	plt.tight_layout(pad=5)

	data = generate_data()
	x_values = normalize_data(list(map(lambda x: x.height[0], data)))
	y_values = normalize_data(list(map(lambda x: 1 if x.is_cool else 0, data)))
	ax[0].scatter(x_values, y_values, alpha=0.6)
	
	# logistic regression
	(slope, intercept) = gradient_descent_range(1, 0, 0.5, x_values, y_values)
	# print(f"({slope}, {intercept})")
	# print(sigmoid_reverse(linear(slope, 0, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.1, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.2, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.3, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.4, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.5, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.6, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.7, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.8, intercept)))
	# print(sigmoid_reverse(linear(slope, 0.9, intercept)))
	# print(sigmoid_reverse(linear(slope, 1, intercept)))
	x_regres, y_regres = generate_regression_line(slope, intercept, 0, 1, 50)
	ax[0].plot(x_regres, y_regres, 'c', alpha=0.5)


	# export matplotlib as png
	plt.savefig(f"{sys.argv[0]}.png")

main()