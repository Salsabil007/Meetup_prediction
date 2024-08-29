import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    with open('data_JP_total.csv', 'r') as file:
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
                    
    dataframe = pd.read_csv("data_JP_total.csv")
  
    dataframe = dataframe.drop(dataframe.columns[[3]], 1)
    dataframe['day'] = day
    dataframe['month'] = month
    dataframe['date'] = date
    dataframe['hour'] = hour
    dataframe['year'] = year
    dataframe['holiday'] = holiday
    #dataframe.to_csv("inter.csv", index=False)
    return dataframe

data = csv_to_df()
data.to_csv("data_JP_final.csv", index = False)
