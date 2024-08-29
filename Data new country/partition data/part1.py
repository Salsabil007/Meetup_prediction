import pandas as pd
import numpy as np

#chunks = pd.read_csv("data_JP_sort_venue.csv", chunksize=100000)
#data = pd.concat(chunks)
#print(len(data))


#data = data[data['venueId'] <= '4b5853c6f964a520f65228e3']
#data.to_csv("data_p1_v1.csv", index = False)


'''
data1 = pd.read_csv("data_p11_v1.csv")
data2 = pd.read_csv("data_p11_v2.csv")
result = pd.merge(data1,data2,how='inner',on = 'venueId')
result = result[result.userId != result.userId2]
result.to_csv("result1.csv", index = False)'''

'''

data = pd.read_csv("data_p1_v1.csv")
data = data[data['venueId'] <= '4b47b936f964a520f53b26e3']
data.to_csv("data_p11_v1.csv", index = False) '''

'''
data = pd.read_csv("data_p1_v1.csv")
data = data[data['venueId'] > '4b47b936f964a520f53b26e3']
data.to_csv("data_p1_2.csv", index = False)
print(len(data))
'''
'''
chunks = pd.read_csv("data_JP_sort_venue.csv", chunksize=100000)
data = pd.concat(chunks)
data = data[data['venueId'] > '4b5853c6f964a520f65228e3']
data.to_csv("rest.csv", index = False)'''


'''
data = pd.read_csv("rest.csv")
data = data[data['venueId'] <= '4b600a59f964a520fed329e3']
data.to_csv("data_p2.csv", index = False)
print(len(data))
'''
'''
data = pd.read_csv("rest.csv")
data = data[data['venueId'] > '4b600a59f964a520fed329e3']
data.to_csv("rest.csv", index = False)
'''

'''
data = pd.read_csv("rest.csv")
data = data[data['venueId'] <= '4b77725df964a520f79a2ee3']
data.to_csv("data_p3.csv", index = False)
'''

'''
data = pd.read_csv("rest.csv")
data = data[data['venueId'] > '4b77725df964a520f79a2ee3']
data.to_csv("rest.csv", index = False)
'''


'''
data = pd.read_csv("rest.csv")
data = data[data['venueId'] <= '4baea3ecf964a520cfc93be3']
data.to_csv("data_p4.csv", index = False)
'''

data = pd.read_csv("rest.csv")
print(len(data))
data = data[data['venueId'] > '4baea3ecf964a520cfc93be3']
print(len(data))
data.to_csv("rest.csv", index = False)