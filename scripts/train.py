import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
import pickle
import os
import subprocess

num_pmids = 50000

if not os.path.exists(f"../data/{num_pmids}_pmids_dataset.pkl"):
    try:
        subprocess.run(['python', 'generate_dataset.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating dataset: {e}")
        exit(1)

try:
    with open(f"{num_pmids}_pmids_dataset.pkl", "rb") as file:
        dataset = pickle.load(file)

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)



    















