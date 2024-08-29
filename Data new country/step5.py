import pandas as pd
import numpy as np

data = pd.read_csv("data_JP_final.csv")
data2 = data
data = data.drop(data.columns[[1,2,3,4,5,6,7,8,9,10,11,12]],1)
data['userid2'] = data2['userid2']
data['latitude'] = data2['latitude']
data['longitude'] = data2['longitude']
data['venue_catagory'] = data2['venue_catagory']
data['day'] = data2['day']
data['month'] = data2['month']
data['date'] = data2['date']
data['hour'] = data2['hour']
data['year'] = data2['year']
data['holiday'] = data2['holiday']
data['venue_cat'] = data2['venue_cat']


period = []
City = []
revisit = []
POI_total_check = []

for ind in data.index: 
    h = data['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
    City.append(1)
    revisit.append(1)
    POI_total_check.append(1)

data['City'] = City
data['revisit'] = revisit
data['POI_total_check'] = POI_total_check
data['period'] = period

print(data.head(10))
data.to_csv("prep1.csv", index = False)