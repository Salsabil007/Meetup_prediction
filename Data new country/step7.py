import pandas as pd
import numpy as np

data = pd.read_csv("all_user_period.csv")
data = data.drop(data.columns[[9]],1)
data.to_csv("alluser_final_withtime&location.csv", index = False)