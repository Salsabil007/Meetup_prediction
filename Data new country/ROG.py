import pandas as pd
import numpy as np
import math

def conv_to_cartesian(lat, lon):
    R = 6371 #km
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z
def conv_to_latlon(x,y,z):
    R = 6371
    lat = math.asin(z / R)
    lon = math.atan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)

def distance(lat1,lon1, lat2, lon2):
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    
    return d

data = pd.read_csv("alluser_final_withtime&location.csv")
un_data = data
un_data = un_data.drop(un_data.columns[[1,4,5,6,7,8,9,19,11,12,13,14,15]], 1)
print(un_data.dtypes)
un_data = un_data.drop_duplicates()
un_data = un_data
x = un_data['userId'].unique()
user = []
ROG = []
center_lat = []
center_lon = []
for i in x:
    temp = un_data[un_data['userId'] == i]
    x_t,y_t,z_t = 0.00,0.00,0.00
    for ind in temp.index:
        x,y,z = conv_to_cartesian(temp['latitude'][ind],temp['longitude'][ind])
        x_t += x
        y_t += y
        z_t += z
    x_t = x_t / len(temp)
    y_t = y_t / len(temp)
    z_t = z_t / len(temp)
    avg_lat,avg_lon = conv_to_latlon(x_t,y_t,z_t)
    for ind in temp.index:
        dist = distance(avg_lat,avg_lon,temp['latitude'][ind],temp['longitude'][ind])
        dist = dist * dist
        dist_sum += dist
    dist_sum = dist_sum / len(temp)
    dist_sum = math.sqrt(dist_sum)
    user.append(i)
    ROG.append(dist_sum)
    center_lat.append(avg_lat)
    center_lon.append(avg_lon)
list_of_tuples = list(zip(user, ROG, center_lat, center_lon))
df = pd.DataFrame(list_of_tuples,
                  columns = ['user', 'ROG', 'center_lat', 'center_lon'])

print(df.head(5))








