import numpy as np
import os
import matplotlib.pyplot as plt
from Q1 import read_files

script_dir = os.path.dirname(__file__)
rel_path = "Q3/"
abs_file_path = os.path.join(script_dir, rel_path)
output_dir = "Q3_results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)