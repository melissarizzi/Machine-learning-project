import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class preprocessing:
    def __init__(self, path,test_size=0.2, val_size=0.2):
        self.df = pd.read_csv(path)
        self.test_size = test_size
        self.val_size = val_size
        self.outliers = None

    def check_missing_value(self):
        self.n_missing = self.df.isnull().sum()
        self.df = self.df.dropna()

