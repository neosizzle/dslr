import sys
import csv

import model as ft_model

# read and parse csv
# [[indices], [houses], [name] ... ]
def read_csv(filepath) :
	res = []
	file=open(filepath, "r")
	reader = csv.reader(file)
	skip_first_line = True
	for line in reader:
		t=line

		if skip_first_line:
			skip_first_line = False
			continue

		entry = ft_model.Model()
		for column in ft_model.DATA_MODEL:
			index = column["idx"]
			feature = column["name"]
			data = ft_model.match_types(line[index], column["type"])
			entry.set_feature(feature, data)
		res.append(entry)
	return res

# get mean values of all features
def get_mean_values(data):
	res = [None] * len(ft_model.DATA_MODEL)
	feature_total = [None] * len(ft_model.DATA_MODEL)
	feature_count = [None] * len(ft_model.DATA_MODEL)

	# iterate through all data
	for row in data:
		# iterate through all features
		all_features = row.get_all_features()
		for feature in all_features :
			# determine if the feature is enumerable
			value = row.get_feature(feature)
			model = next(x for x in ft_model.DATA_MODEL if x["name"] == feature)
			model_idx = model["idx"]
			if model["type"] != "float" or value == None:
				continue

			if feature_total[model_idx] == None:
				feature_count[model_idx] = 0
				feature_total[model_idx] = 0

			feature_count[model_idx] += 1
			feature_total[model_idx] += value

	for i in range(len(feature_total)):
		if feature_count[i] == None:
			continue
		res[i] = feature_total[i] / feature_count[i]
	return res

# cleanup none data my replacing them with the mean
def preprocess_data(data):
	mean_values = get_mean_values(data)

	# iterate through all data
	for row in data:
		# iterate through all features
		all_features = row.get_all_features()
		for feature in all_features :
			model = next(x for x in ft_model.DATA_MODEL if x["name"] == feature)
			model_idx = model["idx"]
			if model["type"] != "float":
				continue

			if row.get_feature(feature) == None :
				row.set_feature(feature, mean_values[model_idx])

def normalize_data(data) :
	min_x = min(data)
	max_x = max(data)
	return list(map(lambda x : (x - min_x) / (max_x - min_x) , data))

# generate data matrix for multiple binary logistic regression
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
			y_values = list(map(lambda x: 1 if x.get_feature('Hogwarts House') == category else 0, data))
			res[key] = [x_values, y_values, min_features, max_features]
	return res