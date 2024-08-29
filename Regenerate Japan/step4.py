import pandas as pd 
import numpy as np

'''
data = pd.read_csv("alluser_final_withtime&location.csv")

venue = ['Laundry Service','Gas Station / Garage', 'Airport Terminal','Electronics Store','Road','Building',
'Drugstore / Pharmacy', 'Salon / Barbershop', 'Hardware Store', 'Sporting Goods Shop', 'Hospital', 'Bank','Training Salon',
'Car Wash', 'Courthouse','Temple','Military Base','Capitol Building','Airport Gate',"Doctor's Office",'Embassy / Consulate',
'Airport Lounge','Cemetery','Mountain','Train','Fire Station','Funeral Home','Emergency Room','Plane','Voting Booth',
'Taxi', 'Optical Shop', 'Airport Tram','Volcanoes','Bike Rental / Bike Share','Police Station','Radio Station',
'Butcher','Motorcycle Shop','Post Office','Laboratory']
print('before ',len(data))
for i in venue:
    data = data[data['venue_catagory'] != i]
print('after ',len(data))
data = data.sort_values(by=['userid'])
data.to_csv("user_nobadpoi.csv", index = False) '''

data = pd.read_csv("all_user_poi.csv")
'''data2 = pd.read_csv("POI_JP.csv")
data2 = data2.drop(data2.columns[[4]],1)
data = pd.merge(data1,data2,how='inner',on=['latitude','longitude'])'''

venue = ['Laundry Service','Gas Station / Garage', 'Airport Terminal','Electronics Store','Road','Building',
'Drugstore / Pharmacy', 'Salon / Barbershop', 'Hardware Store', 'Sporting Goods Shop', 'Hospital', 'Bank','Training Salon',
'Car Wash', 'Courthouse','Temple','Military Base','Capitol Building','Airport Gate',"Doctor's Office",'Embassy / Consulate',
'Airport Lounge','Cemetery','Mountain','Train','Fire Station','Funeral Home','Emergency Room','Plane','Voting Booth',
'Taxi', 'Optical Shop', 'Airport Tram','Volcanoes','Bike Rental / Bike Share','Police Station','Radio Station',
'Butcher','Motorcycle Shop','Post Office','Laboratory']
print('before ',len(data))
for i in venue:
    data = data[data['venue catagory'] != i]
print('after ',len(data))
data = data.sort_values(by=['userId'])


data.to_csv("user_poi.csv", index = False)