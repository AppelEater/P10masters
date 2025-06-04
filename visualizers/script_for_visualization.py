# %%
import Project_library as pl
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import jax.numpy as jnp
import numpy as np
import time

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'CMU Serif Roman',  # primary CMU serif face
    'Times New Roman',  # fallback
    'Georgia',
    'serif'
]
plt.rcParams['axes.grid'] = True
# 3. Set the global font size:
plt.rcParams['font.size'] = 14  # change this number to taste

# 4. (Optional) Adjust axes titles and labels separately if you like:
plt.rcParams['axes.titlesize'] = 14 
plt.rcParams['axes.labelsize'] = 14

# %%
results_string = "./results_sweep1748721538"


# The scripts labels for each iteration
labels = ["Initial demand", "Mode demand Variable", "Mode demand Sticky", "Expected demand Variable", "Expected demand Sticky"]


records = []

k = 0

with open(results_string, "rb") as f:
    try:
        config = pkl.load(f)["Config"]  # First entry with sweeping_parameters
        print(config)
    except EOFError:
        raise ValueError("File is empty or incorrectly formatted.")

    start = time.time()

    while True:

        try:
            sweep_result = pkl.load(f)
            print(sweep_result.keys())

        except:
            pass
