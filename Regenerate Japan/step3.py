import pandas as pd
import numpy as np

data = pd.read_csv("all_user_period_temp.csv")
data = data.drop(data.columns[[2,12]],1) #removing utc time and period

dd = pd.DataFrame()
dd['userId'] = data['userId']
dd['latitude'] = data['latitude']
dd['longitude'] = data['longitude']
dd['day'] = data['day']
dd['month'] = data['month']
dd['date'] = data['date']
dd['hour'] = data['hour']
dd['year'] = data['year']
dd['time'] = data['time']
dd['venueid'] = data['venueid']
dd['venue catagory'] = data['venue catagory']
print(len(dd))
print(len(data))
print(data.head(5))
print(dd.head(5))

dd = dd.sort_values(by=['userId','time'])
dd.to_csv("all_user_poi.csv", index = False)
