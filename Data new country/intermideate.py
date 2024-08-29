import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import tensorflow as tf
import math

def top_clstr_day_period2(data,day,user,period): 
    df = data[data['day'] == day]
    df1 = df[df['period'] == period]
    value_counts = df1['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    #counts = counts.head(1)
    #print("counts %%%%%%%%% ",counts)
    if len(counts) > 0:
        counts = counts.to_numpy()
        return counts[0][0]
    else:
        value_counts = df['cluster_grp'].value_counts()
        counts = pd.DataFrame(value_counts)
        counts = counts.reset_index()
        counts.columns = ['unique_id', 'counts']
        if len(counts) > 0:
            counts = counts.to_numpy()
            return counts[0][0]
            
    value_counts = data['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    counts = counts.to_numpy()
    return counts[0][0]

def week_cluster(user,backup2,period,time):
    #data = backup2[backup2['userid'] == user] ### old
    
    data = pd.read_csv("all_user_cluster.csv") ###newline
    data = data[data['userId'] ==user] ###new line
    data = data[data['time'] < time] ### new
    df = np.empty(7)
    a0,a1,a2,a3,a4,a5,a6 = 0,0,0,0,0,0,0
    for i in range(7):
        d = top_clstr_day_period2(data,i,user,period)
        df[i] = d
        if i == 0:
            a0 = d
        elif i == 1:
            a1 = d
        elif i == 2:
            a2 = d
        elif i == 3:
            a3 = d
        elif i == 4:
            a4 = d
        elif i == 5:
            a5 = d
        elif i == 6:
            a6 = d
    df = np.array(df)
    return a0,a1,a2,a3,a4,a5,a6

def training_model(df,nn,data_no_scale,backup2,data_no_scale_2,train_2):
    x = df['userid']
    x = x.unique()
    X,y = [],[]
    m,d,p,yr,grp,dt,near_cls_grp,clsr_day,clstr_period,pp,clstr_frnd=[],[],[],[],[],[],[],[],[],[],[]
    lat,long=[],[]
    t_diff,d_diff = [],[]
    wcd,wcn = [],[]
    nc = []
    cof = []
    y = np.empty(0)
    feature = []
    for i in x:
        instance = df[df.userid == i]
        sample = instance
        
        sample = sample.head(5)
        
        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        
        a = np.empty(15)
        a[0] = i
        a[1],a[2],a[3],a[4],a[5],a[6],a[7] = week_cluster(i,backup2,0,time)
        a[8],a[9],a[10],a[11],a[12],a[13],a[14] = week_cluster(i,backup2,1,time)
        feature.append(a)
    feature = np.array(feature)
    feature = pd.DataFrame(feature, columns = ['userid','wd1','wd2','wd3','wd4','wd5','wd6','wd7','wn1','wn2','wn3','wn4','wn5','wn6','wn7'])
    feature['userid'] = feature['userid'].astype(int)
    feature.to_csv("week_first.csv", index = False)

    feature = []
    df = train_2
    x = df['userid']
    x = x.unique()
    for i in x:
        instance = df[df.userid == i]
        sample = instance
    
        sample = sample.head(5)
        
        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        
        a = np.empty(15)
        a[0] = i
        a[1],a[2],a[3],a[4],a[5],a[6],a[7] = week_cluster(i,backup2,0,time)
        a[8],a[9],a[10],a[11],a[12],a[13],a[14] = week_cluster(i,backup2,1,time)
        feature.append(a)
    feature = np.array(feature)
    feature = pd.DataFrame(feature, columns = ['userid','wd1','wd2','wd3','wd4','wd5','wd6','wd7','wn1','wn2','wn3','wn4','wn5','wn6','wn7'])
    feature['userid'] = feature['userid'].astype(int)
    feature.to_csv("week_last.csv", index = False)





cluster_centers = pd.read_csv("cluster_center_0.002.csv")
nn = len(cluster_centers)
data = pd.read_csv("data_final_first6.csv")
data_2 = pd.read_csv("data_final_last6.csv")
cluster_centers = cluster_centers.to_numpy()
data_no_scale = pd.read_csv("data_before_scale_first6.csv")
data_no_scale_2 = pd.read_csv("data_before_scale_last6.csv")
backup2 = pd.read_csv("backup2_first6.csv")

time = []
for ind in backup2.index:
    mon = backup2['month'][ind]
    dat = backup2['date'][ind]
    year = backup2['year'][ind]
    ho = backup2['hour'][ind] 
    t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
    time.append(t)
backup2['time'] = time 

backup = pd.read_csv("backup.csv")
backup = backup.drop(backup.columns[[15]],1)
period = []
for ind in backup.index:
    h = backup['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
backup['period'] = period

train = data

train_2 = data_2
training_model(train,nn,data_no_scale,backup2,data_no_scale_2,train_2)
