import pandas as pd
import numpy as np

'''
data = pd.read_csv("data_JP_Atheena.csv")
print(len(data))

data = data[abs(data['time_in_minute'] - data['time_in_minute2']) <= 90]
print(len(data))
data.to_csv("data_JP_90min.csv", index = False)
'''
'''
data = pd.read_csv("data_JP_90min.csv")
data = data.drop(data.columns[[5,6]],1)
data.to_csv("data_JP_90min.csv", index = False) '''

is_friend = []
data = pd.read_csv("friendship2.csv")
with open('data_JP_90min.csv', 'r') as file:
        i = -1
        for line in file:
            i += 1
            if i == 0:
                continue
            p = 0
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if p == 1:
                    id1 = int(val)
                elif p == 5:
                    id2 = int(val)
                    data1 = data[(data.userid1 == id1) & (data.userid2 == id2)]
                    data2 = data[(data.userid1 == id2) & (data.userid2 == id1)]
                    if len(data1.index) == 0 and len(data2.index) == 0:
                        is_friend.append("NO")
                    else:
                        is_friend.append("YES")

                else:
                    continue

data = pd.read_csv("data_JP_90min.csv")
data['is_friend'] = is_friend
print(len(data))
data.to_csv("temp.csv", index = False)
data = data[data['is_friend'] == "YES"]
print(len(data))
data.to_csv("data_JP_friend.csv", index = False)

