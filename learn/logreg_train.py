import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import json

import preprocess as ft_preprocess
import plot as ft_plot
import model as ft_model
import ft_math

def derivative_m(y_binary, x, curr_m, curr_c):
	# print(f"linear: {linear(curr_m, x, curr_c)}")
	return (ft_math.sigmoid(ft_math.linear(curr_m, x, curr_c)) - y_binary) * x

def derivative_c(y_binary, x, curr_m, curr_c):
	return (ft_math.sigmoid(ft_math.linear(curr_m, x, curr_c)) - y_binary)

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
	# print(f"GD m {init_m} c {init_c}, steps {steps}")
	return (init_m, init_c)

def gradient_descent_batch(init_m, init_c, learning_rate, _x_values, _y_values):
	step_size_m = 6969
	step_size_c = 6969
	step_size_range = 0.00005
	steps = 0

	sample_indices = random.sample(range(0, len(_x_values)), len(_x_values) - 200)
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
	# print(f"GD m {init_m} c {init_c}, steps {steps}")
	return (init_m, init_c)

def generate_regression_line(slope, intercept, min, max, steps) :
	res = ([], [])

	step_size = max / steps
	curr_x = min
	for i in range(steps):
		res[0].append(curr_x)
		res[1].append((ft_math.sigmoid_reverse(ft_math.linear(slope, curr_x, intercept))))
		curr_x += step_size

	return res

def write_to_json(weights):
	filename = "weights.json"

	# writing to csv file
	with open(filename, 'w') as jsonfile:
		r = json.dumps(weights)
		jsonfile.write(r)

def main():
	if len(sys.argv) != 2:
		print("Usage: python logreg_train.py [filename]")
		return

	(fig, ax) = ft_plot.init_mpl()
	raw_data = ft_preprocess.read_csv(sys.argv[1])

	ft_preprocess.preprocess_data(raw_data)
	data_matrix = ft_preprocess.generate_data_matrix(ft_model.HOUSES, ft_model.SELECTED_FEATURES, raw_data)

	weights = {}

	for index, (key, plot_data) in enumerate(data_matrix.items()):
		# logistic regression
		(slope, intercept) = gradient_descent_batch(1, 0, 0.5, plot_data[0], plot_data[1])
		print(f"{key}: ({slope}, {intercept})")
		
		x_regres, y_regres = generate_regression_line(slope, intercept, 0, 1, 50)
		ft_plot.plot_index(index, ax, plot_data, x_regres, y_regres)
		weights[key] = [slope, intercept, plot_data[2], plot_data[3]]
	# print(data_matrix)
	write_to_json(weights)
	ft_plot.save_fig()
main()