import pandas as pd
import numpy as np

data = pd.read_csv("data_JP.csv")
print(len(pd.unique(data['userId'])))
print(len(pd.unique(data['venueid'])))
print(len(data))
