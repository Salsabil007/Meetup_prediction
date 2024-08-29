import pandas as pd 
import numpy as np

'''
data = pd.read_csv("prep1.csv")
value_counts = data['venue_catagory'].value_counts()
counts = pd.DataFrame(value_counts)
counts = counts.reset_index()
counts.columns = ['venue', 'counts']
counts.to_csv("venue_cat_count.csv", index = False)'''

data = pd.read_csv("prep1.csv")
data = data.drop_duplicates()
venue = ['Laundry Service','Gas Station / Garage', 'Airport Terminal','Electronics Store','Road','Building',
'Drugstore / Pharmacy', 'Salon / Barbershop', 'Hardware Store', 'Sporting Goods Shop', 'Hospital', 'Bank','Training Salon',
'Car Wash', 'Courthouse','Temple','Military Base','Capitol Building','Airport Gate',"Doctor's Office",'Embassy / Consulate',
'Airport Lounge','Cemetery','Mountain','Train','Fire Station','Funeral Home','Emergency Room','Plane','Voting Booth',
'Taxi', 'Optical Shop', 'Airport Tram','Volcanoes','Bike Rental / Bike Share','Police Station','Radio Station',
'Butcher','Motorcycle Shop','Post Office','Laboratory','Car Dealership','Pet Store','Camera Store','Paper / Office Supplies Store',
'Ferry']
print('before ',len(data))
for i in venue:
    data = data[data['venue_catagory'] != i]
print('after ',len(data))
data.to_csv("no_bad_POI.csv", index = False) 