import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import accuracy_score

def read_csv(filepath) :
	res = []
	file=open(filepath, "r")
	reader = csv.reader(file)
	skip_first_line = True
	for line in reader:
		if skip_first_line:
			skip_first_line = False
			continue

		res.append(line[1])
	return res

def main():
	if len(sys.argv) != 3:
		print("Usage: python validate.py [filename_predicted] [filename_truth]")
		return
	predicted_data = read_csv(sys.argv[1])
	truth_data = read_csv(sys.argv[2])
	accuracy = accuracy_score(predicted_data, truth_data)
	print(f"accuracy: {accuracy * 100}%")
main()