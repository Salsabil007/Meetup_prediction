import pandas as pd
import numpy as np 

#removing quotation of every data
data1 = pd.read_csv("output_p1.csv",quotechar='"',skipinitialspace=True)
data2 = pd.read_csv("output_p2.csv",quotechar='"',skipinitialspace=True)
data3 = pd.read_csv("output_p3.csv",quotechar='"',skipinitialspace=True)
data4 = pd.read_csv("output_p4.csv",quotechar='"',skipinitialspace=True)
data5 = pd.read_csv("output_p5.csv",quotechar='"',skipinitialspace=True)
data6 = pd.read_csv("output_p6.csv",quotechar='"',skipinitialspace=True)
print(len(data1))
print(len(data2))
print(len(data3))
print(len(data4))
print(len(data5))
print(len(data6))
print("total ",len(data1)+len(data2)+len(data3)+len(data4)+len(data5)+len(data6))

data1 = data1.append(data2, ignore_index = True)
data1 = data1.append(data3, ignore_index = True)
data1 = data1.append(data4, ignore_index = True)
data1 = data1.append(data5, ignore_index = True)
data1 = data1.append(data6, ignore_index = True)

data1.to_csv("data_JP_Atheena.csv", index = False)
print(len(data1))