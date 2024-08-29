import pandas as pd 
import numpy as np 

def str_to_date(str):
    str = str.strip() #removing any heading or tailing extra space
    x = str.split() #spliting from space #Format:Fri May 04 01:18:03 +0000 2012
    day,month,date,year = x[0],x[1],x[2],x[5]
    hour = x[3][0:2]
    return day,month,date,hour,year

def csv_to_df():
    day = []
    month = []
    date = []
    hour = []
    year = []
    holiday = [] # sat,sun - 1, fri - 0, rest - 2
    with open('all_user_period.csv', 'r') as file:
        i = -1
        for line in file:
            i += 1
            if i == 0:
                continue
            p = 0
            lat = 0.00
            long = 0.00
            for cnt,val in enumerate(line.split(',')):
                p += 1
                if p == 3:
                    d,m,dt,h,y = str_to_date(val)
                    #Sun - 0, Mon - 1, Tue - 2, Wed - 3, Thu - 4, Fri - 5, Sat - 6
                    if d == 'Sun':
                        day.append("0")
                    elif d == 'Mon':
                        day.append("1")
                    elif d == 'Tue':
                        day.append("2")
                    elif d == 'Wed':
                        day.append("3")
                    elif d == 'Thu':
                        day.append("4")
                    elif d == 'Fri':
                        day.append("5")
                    else:
                        day.append("6")
                    
                    if m == 'Jan':
                        month.append("0")
                    elif m == 'Feb':
                        month.append("1")
                    elif m == 'Mar':
                        month.append("2")
                    elif m == 'Apr':
                        month.append("3")
                    elif m == 'May':
                        month.append("4")
                    elif m == 'Jun':
                        month.append("5")
                    elif m == 'Jul':
                        month.append("6")
                    elif m == 'Aug':
                        month.append("7")
                    elif m == 'Sep':
                        month.append("8")
                    elif m == 'Oct':
                        month.append("9")
                    elif m == 'Nov':
                        month.append("10")
                    else:
                        month.append("11")

                    date.append(dt)
                    hour.append(h)
                    
                    if y == '2012':
                        year.append("0")
                    elif y == '2013':
                        year.append("1")
                    else:
                        year.append("2")
                    
                    if d == 'Sat' or d == 'Sun':
                        holiday.append("1")
                    elif d == 'Fri':
                        holiday.append("0")
                    else:
                        holiday.append("2")
                    break
                    
    dataframe = pd.read_csv("all_user_period.csv")
  
    dataframe['day'] = day
    dataframe['day'] = dataframe['day'].astype(int)
    dataframe['month'] = month
    dataframe['month'] = dataframe['month'].astype(int)
    dataframe['date'] = date
    dataframe['date'] = dataframe['date'].astype(int)
    dataframe['hour'] = hour
    dataframe['hour'] = dataframe['hour'].astype(int)
    dataframe['year'] = year
    dataframe['year'] = dataframe['year'].astype(int)
    #dataframe.to_csv("inter.csv", index=False)
    return dataframe




data = pd.read_csv("data_JP.csv")
data = data.drop(data.columns[[3]],1)

poi = pd.read_csv("POI_JP.csv")
poi = poi.drop(poi.columns[[4]],1)

result = pd.merge(data,poi,how='inner',on='venueid')
print(len(data))
print(len(result))

result.to_csv("all_user_period.csv", index = False)


data = csv_to_df()
time = []
period = []
for ind in data.index:
    mon = data['month'][ind]
    dat = data['date'][ind]
    year = data['year'][ind]
    ho = data['hour'][ind] 
    t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
    time.append(t)
    h = data['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
data['time'] = time 
data['period'] = period
print(data.head(5))
print(data.dtypes)
data.to_csv("all_user_period_temp.csv", index = False)
data = data.drop(data.columns[[1,2,5]],1)
data.to_csv("all_user_period.csv", index = False)


'''data = pd.read_csv("all_user_period.csv")
data2 = pd.read_csv("all_user_period_temp.csv")
print("len of data ",len(data))
print("len of data2 ",len(data2))
'''