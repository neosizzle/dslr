import matplotlib.pyplot as plt
import sys

def init_mpl():
	plt.style.use('_mpl-gallery')
	fig, ax = plt.subplots(2, 2, figsize=(10, 10))
	plt.tight_layout(pad=2)
	return (fig, ax)

def save_fig():
	plt.savefig(f"{sys.argv[0]}.png")