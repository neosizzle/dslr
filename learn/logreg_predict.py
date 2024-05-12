import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import csv

import preprocess as ft_preprocess
import plot as ft_plot
import model as ft_model
import ft_math

def normalize_value(data, min, max):
	return ((data - min) / (max - min))

def read_json():
	with open('weights.json', mode='r') as infile:
		return json.load(infile)

def write_to_csv(predictions):
	filename = "houses.csv"

	keys = predictions[0].keys()
	# writing to csv file
	with open(filename, 'w') as output_file:
		dict_writer = csv.DictWriter(output_file, keys)
		dict_writer.writeheader()
		dict_writer.writerows(predictions)

def main():
	if len(sys.argv) != 2:
		print("Usage: python logreg_predict.py [filename]")
		return

	to_predict_data = ft_preprocess.read_csv(sys.argv[1])
	weights = read_json()

	ft_preprocess.preprocess_data(to_predict_data)

	predictions = []
	for idx, to_predict in enumerate(to_predict_data):
		probabilities = {}
		for category in ft_model.HOUSES :
			probabilities[category] = 1
				
		for key, weight_data in weights.items():
			slope = weight_data[0]
			intercept = weight_data[1]
			feature = key.split("_")[1]
			category = key.split("_")[0]
			
			normalied_x_value = normalize_value(to_predict.get_feature(feature), weight_data[2], weight_data[3])
			probability = (ft_math.sigmoid_reverse(ft_math.linear(slope, normalied_x_value, intercept)))
			if category == "Hufflepuff":
				rc = probabilities[ft_model.HOUSES[0]]
				sl = probabilities[ft_model.HOUSES[1]]
				gd = probabilities[ft_model.HOUSES[2]]
			probabilities[category] *= probability

		higest_likelihood = 0
		res = "HUH"
		for category in ft_model.HOUSES:
			if higest_likelihood < probabilities[category]:
				higest_likelihood = probabilities[category]
				res = category
		predictions.append({"Index": idx, "Hogwarts House": res})
	write_to_csv(predictions)
main()