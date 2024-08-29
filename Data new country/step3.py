import pandas as pd
import numpy as np

'''
poi = pd.read_csv("POI.csv")
poi = poi[poi['country_code'] == "JP"]
poi.to_csv("POI_JP.csv", index = False)
'''

poi = pd.read_csv("POI_JP.csv")
poi = poi.drop(poi.columns[[4]],1)
data = pd.read_csv("data_JP_friend.csv")
data = data.drop(data.columns[[5]],1)
print(len(data))
result = pd.merge(data, poi, how = "inner", on = "venueid")
print(len(result))
result = result.sort_values(by=['userid','time_in_minute'])
result = result.drop_duplicates()
print(len(result))
result.to_csv("data_JP_total.csv", index = False)