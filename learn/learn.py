import sys
import matplotlib.pyplot as plt
import numpy as np

import preprocess as ft_preprocess
import plot as ft_plot

def main():
    if len(sys.argv) != 2:
        print("Usage: python learn.py [filename]")
        return
    
    (fig, ax) = ft_plot.init_mpl()
    raw_data = ft_preprocess.read_csv(sys.argv[1])

    ft_preprocess.preprocess_data(raw_data)

    print(raw_data)
main()