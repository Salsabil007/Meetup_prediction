import pandas as pd
import numpy as np

'''
chunks = pd.read_csv("data_with_CC_sorted.csv", chunksize=100000)
data = pd.concat(chunks)
'''


'''
###counting values of data points for each country to pick the highest one
value_counts = data['country_code'].value_counts()
counts = pd.DataFrame(value_counts)
counts = counts.reset_index()
counts.columns = ['country', 'counts']
counts.to_csv("country_count.csv", index = False)
'''


'''
data = data[data['country_code'] == 'JP']
data = data.sort_values(by=['userId','time_in_minute'])
data = data.drop(data.columns[[3,5]], 1) ##removing timezone offset and country code as offset is already added with time_in_minute
data.to_csv("data_JP.csv", index = False)
'''

data1 = pd.read_csv("data_JP.csv")
print(len(data1))

#the JP data is partitioned into parts and merged. check data partition folder.
