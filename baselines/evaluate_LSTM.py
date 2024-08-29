##this code is for clustering with 0.002 quantile with lstm and attention and cluster prediction
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import tensorflow as tf
import math
import sys 

from sklearn.preprocessing import StandardScaler, LabelEncoder
from utility_LSTM import clustering, distance, clstr_find, find_near_cluster, cutlen, save_index_for_frnd, top_clstr_day,top_clstr_period, prob_of_presense, cluster_friend
from utility_LSTM import week_cluster,nearest_cluster_func, avgdist_incluster, numberOfcluster,score_poi,is_in_topk,cutlen2,score_poi_test,cluster_of_friend,score_poi_last
from utility_LSTM import score_poi22,min_dist_find
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Embedding, Input, Flatten
from keras.layers import Dropout
from sklearn.utils import resample
from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth
from keras.optimizers import SGD
from keras.layers import Bidirectional
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Lambda, dot, Activation, concatenate, TimeDistributed, RepeatVector, Concatenate
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization


class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector


'''
#part one
data = pd.read_csv("prep1.csv") #userid,userid2,latitude,longitude,venue_catagory,day,month,date,hour,year,holiday,venue_cat,City,revisit,POI_total_check,period
##remove data with length less than 6
length = 6
#mydict = save_index_for_frnd(data, length)
backup = data
data = data.drop(data.columns[[1,13,14]],1) #dropping userid2,revisit, POI_total_check 
data = data.drop_duplicates()
backup2 = data
value_counts = data['userid'].value_counts()
counts = pd.DataFrame(value_counts)
counts = counts.reset_index()
counts.columns = ['unique_id', 'counts']
counts = counts[counts.counts < length]
counts = counts.drop(counts.columns[[1]],1)
counts = counts.to_numpy()
counts = counts.flatten()
for i in counts: #removing all the users who has data less than len 
    data = data[data.userid != i]
ff = data
data = cutlen(data,length)

### for end data
data_2 = cutlen2(ff,length)

data = data.drop(data.columns[[11,12]],1)
data['userid'] = data['userid'].astype(int)
data['day'] = data['day'].astype(int)
data['month'] = data['month'].astype(int)
data['date'] = data['date'].astype(int)
data['hour'] = data['hour'].astype(int)
data['year'] = data['year'].astype(int)
data['holiday'] = data['holiday'].astype(int)
#data['venue_cat'] = data['venue_cat'].astype(int) ###################################################################################

########## for end data
data_2 = data_2.drop(data_2.columns[[11,12]],1)
data_2['userid'] = data_2['userid'].astype(int)
data_2['day'] = data_2['day'].astype(int)
data_2['month'] = data_2['month'].astype(int)
data_2['date'] = data_2['date'].astype(int)
data_2['hour'] = data_2['hour'].astype(int)
data_2['year'] = data_2['year'].astype(int)
data_2['holiday'] = data_2['holiday'].astype(int)
#data_2['venue_cat'] = data_2['venue_cat'].astype(int) ##############################################################################

time = []
for ind in data.index:
    mon = data['month'][ind]
    dat = data['date'][ind]
    year = data['year'][ind]
    ho = data['hour'][ind] 
    t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
    time.append(t)
data['time'] = time 

#### for end data 
time = []
for ind in data_2.index:
    mon = data_2['month'][ind]
    dat = data_2['date'][ind]
    year = data_2['year'][ind]
    ho = data_2['hour'][ind] 
    t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
    time.append(t)
data_2['time'] = time

data = data.sort_values(by=['userid','time'])
data_2 = data_2.sort_values(by=['userid','time'])

data = data.append(data_2, ignore_index = True)
data,nn, cluster_centers = clustering(data)
data['cluster_grp'] = data['cluster_grp'].astype(int)
print("number of clusters ", nn)
#cluster_centers = pd.read_csv("cluster_center_0.002.csv")
#nn = len(cluster_centers)
#cluster_centers = cluster_centers.to_numpy()
#print("number of clusters ", nn)
### need to separate data_2 from data
data_2 = data.tail(8652)
data = data.head(8652)

#clstr_assign = []
#for ind in data.index:
#    clstr = clstr_find(data['latitude'][ind], data['longitude'][ind],cluster_centers,nn)
#    clstr_assign.append(clstr)
#data['cluster_grp'] = clstr_assign



data.to_csv("step1_with_cluster.csv", index = False) 
data_2.to_csv("step1_with_cluster_2.csv", index = False) 

clstr_assign = []
for ind in backup.index:
    clstr = clstr_find(backup['latitude'][ind], backup['longitude'][ind],cluster_centers,nn)
    clstr_assign.append(clstr)
backup['cluster_grp'] = clstr_assign
backup.to_csv("backup.csv", index = False)


cc = pd.DataFrame(cluster_centers, columns = ['lat','long']) ## no need, we already have the file
cc.to_csv("cluster_center_0.002.csv", index = False)

clstr_assign = []
for ind in backup2.index:
    clstr = clstr_find(backup2['latitude'][ind], backup2['longitude'][ind],cluster_centers,nn)
    clstr_assign.append(clstr)
backup2['cluster_grp'] = clstr_assign
backup2.to_csv("backup2.csv", index = False)
'''

'''
#part2
data = pd.read_csv("step1_with_cluster.csv")
data_2 = pd.read_csv("step1_with_cluster_2.csv")
#userid,latitude,longitude,venue_catagory,day,month,date,hour,year,holiday,venue_cat,time,cluster_grp
#all are length 6. duplicate events with different people at same place and time are removed

period = []
for ind in data.index:
    h = data['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
data['period'] = period

### for end data
period = []
for ind in data_2.index:
    h = data_2['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
data_2['period'] = period
#print("data ", data.dtypes)


backup2 = pd.read_csv("backup2.csv")
#userid,latitude,longitude,venue_catagory,day,month,date,hour,year,holiday,venue_cat,City,period,cluster_grp
# no legth cutting. but removed duplicate events with same place and time and different people

backup2 = backup2.drop(backup2.columns[[12]],1) #removing previous period
period = []
for ind in backup2.index:
    h = backup2['hour'][ind]
    if h >= 7 and h <= 19: #day
        period.append(0)
    else:
        period.append(1) #night
backup2['period'] = period
backup2.to_csv("backup2_first6.csv", index = False) ######################## added this line

cluster_centers = pd.read_csv("cluster_center_0.002.csv")
nn = len(cluster_centers)
#print("clusters ",nn)
cluster_centers = cluster_centers.to_numpy()

all_user = pd.read_csv("alluser_final_withtime&location.csv")
#userId,latitude,longitude,day,month,date,hour,year,time


#################################################################################################

a_user = pd.read_csv("all_user_period.csv")
clstr = []
for ind in a_user.index:
    lat = a_user['latitude'][ind]
    lon = a_user['longitude'][ind]
    cl = clstr_find(lat,lon,cluster_centers,nn)
    clstr.append(cl)
a_user['cluster_grp'] = clstr
a_user.to_csv("all_user_cluster.csv", index = False)
a_user = pd.read_csv("all_user_cluster.csv")


near_cluster = find_near_cluster(data,all_user,cluster_centers,nn) #users' near cluster from all data
time_diff = []
dist_diff = []
prev = 0
prevlat = 0.00
prevlon = 0.00
for ind in data.index:
    #print(ind)
    if ind % 6 == 0:
        time_diff.append(5)
        prev = data['time'][ind]
        dist_diff.append(0)
        prevlat = data['latitude'][ind]
        prevlon = data['longitude'][ind]
    else:
        x = data['time'][ind] - prev
        time_diff.append(x)
        prev = data['time'][ind] 
        if x < 0:
            print("-1 got ", ind)
        dist = distance(prevlat,prevlon, data['latitude'][ind], data['longitude'][ind])
        prevlat = data['latitude'][ind]
        prevlon = data['longitude'][ind]
        dist_diff.append(abs(dist))

data['time_diff'] = time_diff
data['dist_diff'] = dist_diff 

### for end data
near_cluster2 = find_near_cluster(data_2,all_user,cluster_centers,nn) #users' near cluster from all data
time_diff = []
dist_diff = []
prev = 0
prevlat = 0.00
prevlon = 0.00
for ind in data_2.index:
    #print(ind)
    if ind % 6 == 0:
        time_diff.append(5)
        prev = data_2['time'][ind]
        dist_diff.append(0)
        prevlat = data_2['latitude'][ind]
        prevlon = data_2['longitude'][ind]
    else:
        x = data_2['time'][ind] - prev
        time_diff.append(x)
        prev = data_2['time'][ind] 
        if x < 0:
            print("-1 got ", ind)
        dist = distance(prevlat,prevlon, data_2['latitude'][ind], data_2['longitude'][ind])
        prevlat = data_2['latitude'][ind]
        prevlon = data_2['longitude'][ind]
        dist_diff.append(abs(dist))

data_2['time_diff'] = time_diff
data_2['dist_diff'] = dist_diff 



backup = pd.read_csv("backup.csv")
#userid,userid2,latitude,longitude,venue_catagory,day,month,date,hour,year,holiday,venue_cat,City,revisit,POI_total_check,period,cluster_grp
#no legth cutting. with duplicates events
#print("backup ", backup.dtypes)
length = 6
#mydict = save_index_for_frnd(backup, length)

bb = backup2
tim = []
for ind in bb.index:
    mon = bb['month'][ind]
    dat = bb['date'][ind]
    year = bb['year'][ind]
    ho = bb['hour'][ind] 
    t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
    tim.append(t)
bb['time'] = tim



clstr_ofthe_day = []
clstr_ofthe_period = []
prob_of_pres = []
clstr_of_friend = []
nearest_cluster = []
for ind in data.index:
    #################clstr = top_clstr_day(backup2,data['day'][ind],data['userid'][ind])
    clstr = top_clstr_day(bb,data['day'][ind],data['userid'][ind],a_user,data['time'][ind])
    clstr_ofthe_day.append(clstr)
    ################cp = top_clstr_period(backup2,data['period'][ind],data['userid'][ind])
    cp = top_clstr_period(bb,data['period'][ind],data['userid'][ind],a_user,data['time'][ind])
    clstr_ofthe_period.append(cp)
    pp = prob_of_presense(data['day'][ind],data['period'][ind],data['userid'][ind],backup2,data['cluster_grp'][ind])
    ###################pp = prob_of_presense(data['day'][ind],data['period'][ind],data['userid'][ind],a_user,data['cluster_grp'][ind])
    prob_of_pres.append(pp)
    cf = cluster_friend(data['userid'][ind],data['day'][ind],data['period'][ind],all_user,cluster_centers,nn,data['time'][ind], backup)
    clstr_of_friend.append(cf)
    n = nearest_cluster_func(cluster_centers,data['cluster_grp'][ind],nn)
    nearest_cluster.append(n)

### for end data
clstr_ofthe_day2 = []
clstr_ofthe_period2 = []
prob_of_pres2 = []
clstr_of_friend2 = []
nearest_cluster2 = []
for ind in data_2.index:
    ###############clstr = top_clstr_day(bb,data_2['day'][ind],data_2['userid'][ind])
    clstr = top_clstr_day(backup2,data_2['day'][ind],data_2['userid'][ind],a_user,data_2['time'][ind])
    clstr_ofthe_day2.append(clstr)
    ##############cp = top_clstr_period(bb,data_2['period'][ind],data_2['userid'][ind])
    cp = top_clstr_period(backup2,data_2['period'][ind],data_2['userid'][ind],a_user,data_2['time'][ind])
    clstr_ofthe_period2.append(cp)
    pp = prob_of_presense(data_2['day'][ind],data_2['period'][ind],data_2['userid'][ind],backup2,data_2['cluster_grp'][ind])
    ##############pp = prob_of_presense(data_2['day'][ind],data_2['period'][ind],data_2['userid'][ind],a_user,data_2['cluster_grp'][ind])
    prob_of_pres2.append(pp)
    cf = cluster_friend(data_2['userid'][ind],data_2['day'][ind],data_2['period'][ind],all_user,cluster_centers,nn,data_2['time'][ind], backup)
    clstr_of_friend2.append(cf)
    n = nearest_cluster_func(cluster_centers,data_2['cluster_grp'][ind],nn)
    nearest_cluster2.append(n)


data_save = data ###correction
data2_save = data_2 ###correction



dd = data
dd.to_csv("data_before_scale_first6.csv", index = False)
data_2.to_csv("data_before_scale_last6.csv", index = False)
data = data.head(6918) ###correction
data_2 = data_2.head(6918) ###correction
data = data.append(data_2, ignore_index = True)
dupp = data
dupp = dupp.drop(dupp.columns[[0,3,4,5,6,7,8,9,10,11,12,13]],1) 
scaler = StandardScaler().fit(dupp)
dupp = scaler.transform(dupp)
dupp = pd.DataFrame(dupp, columns = ['latitude','longitude','time_diff','dist_diff'])
dupp['userid'] = data['userid']
dupp['day'] = data['day']
dupp['month'] = data['month']
dupp['date'] = data['date']
dupp['hour'] = data['hour']
dupp['year'] = data['year']
#dupp['holiday'] = data['holiday']
dupp['period'] = data['period']
dupp['cluster_grp'] = data['cluster_grp']


###correction
new_data1 = data_save.tail(1734)
new_data2 = data2_save.tail(1734)
new_data = new_data1.append(new_data2, ignore_index = True)
dupp2 = new_data
dupp2 = dupp2.drop(dupp2.columns[[0,3,4,5,6,7,8,9,10,11,12,13]],1) 
scaler = StandardScaler().fit(dupp2)
dupp2 = scaler.transform(dupp2)
dupp2 = pd.DataFrame(dupp2, columns = ['latitude','longitude','time_diff','dist_diff'])
dupp2['userid'] = new_data['userid']
dupp2['day'] = new_data['day']
dupp2['month'] = new_data['month']
dupp2['date'] = new_data['date']
dupp2['hour'] = new_data['hour']
dupp2['year'] = new_data['year']
#dupp['holiday'] = data['holiday']
dupp2['period'] = new_data['period']
dupp2['cluster_grp'] = new_data['cluster_grp']




data = dupp.head(6918)
data = data.append(dupp2.head(1734), ignore_index = False)
data['near_cluster'] = near_cluster
data['clstr_ofthe_day'] = clstr_ofthe_day
data['clstr_ofthe_period'] = clstr_ofthe_period
data['prob_of_pres'] = prob_of_pres
data['clstr_of_friend'] = clstr_of_friend
data['nearest_cluster'] = nearest_cluster
print("total data ",len(data))
print("type of data ", data.dtypes)
data.to_csv("data_final_first6.csv", index = False)

### for end data
data_2 = dupp.tail(6918)
data_2 = data_2.append(dupp2.tail(1734), ignore_index = False)

data_2['near_cluster'] = near_cluster2
data_2['clstr_ofthe_day'] = clstr_ofthe_day2
data_2['clstr_ofthe_period'] = clstr_ofthe_period2
data_2['prob_of_pres'] = prob_of_pres2
data_2['clstr_of_friend'] = clstr_of_friend2
data_2['nearest_cluster'] = nearest_cluster2
#print("total data ",len(data))
#print("type of data ", data.dtypes)
data_2.to_csv("data_final_last6.csv", index = False)
'''



#part3
#sys.stdout = open("US_output_top1.txt", "w")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
def process_test2(df, model,nn,cc,data_no_scale,backup2,test_2,data_no_scale_2):
    correct =0
    total =0
    x = df['userid']
    x = x.unique()
    X= []
    y = []
    m,d,p,yr,grp,dt,near_cls_grp,clsr_day,clstr_period,pp,clstr_frnd=[],[],[],[],[],[],[],[],[],[],[]
    lat,long=[],[]
    t_diff,d_diff = [],[]
    wcd,wcn=[],[]
    nc = []
    for i in x:
        total += 1
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])
        

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        y.append(out)
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        wcd.append(week_cluster(i,backup2,0))
        wcn.append(week_cluster(i,backup2,1))

    data_no_scale = data_no_scale_2
    df = test_2
    x = df['userid']
    x = x.unique()
    for i in x:
        total += 1
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])
        

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        y.append(out)
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        wcd.append(week_cluster(i,backup2,0))
        wcn.append(week_cluster(i,backup2,1))
    

    y = np.array(y)
    y = np_utils.to_categorical(y, nn)
    d = np.array(d)
    d = np.reshape(d,(d.shape[0],d.shape[1],1))
    nc = np.array(nc)
    nc = np.reshape(nc,(nc.shape[0],nc.shape[1],1))
    dt = np.array(dt)
    dt = np.reshape(dt,(dt.shape[0],dt.shape[1],1))
    p = np.array(p)
    p = np.reshape(p,(p.shape[0],p.shape[1],1))
    m = np.array(m)
    m = np.reshape(m,(m.shape[0],m.shape[1],1))
    yr = np.array(yr)
    yr = np.reshape(yr,(yr.shape[0],yr.shape[1],1))
    grp = np.array(grp)
    grp = np.reshape(grp,(grp.shape[0],grp.shape[1],1))
    near_cls_grp = np.array(near_cls_grp)
    near_cls_grp = np.reshape(near_cls_grp,(near_cls_grp.shape[0],near_cls_grp.shape[1],1))
    clsr_day = np.array(clsr_day)
    clsr_day = np.reshape(clsr_day,(clsr_day.shape[0],clsr_day.shape[1],1))
    clstr_period = np.array(clstr_period)
    clstr_period = np.reshape(clstr_period,(clstr_period.shape[0],clstr_period.shape[1],1))
    clstr_frnd = np.array(clstr_frnd)
    clstr_frnd = np.reshape(clstr_frnd,(clstr_frnd.shape[0],clstr_frnd.shape[1],1))
    lat = np.array(lat)
    lat = np.reshape(lat,(lat.shape[0],lat.shape[1],1))
    long = np.array(long)
    long = np.reshape(long,(long.shape[0],long.shape[1],1))
    t_diff = np.array(t_diff)
    t_diff = np.reshape(t_diff,(t_diff.shape[0],t_diff.shape[1],1))
    d_diff = np.array(d_diff)
    d_diff = np.reshape(d_diff,(d_diff.shape[0],d_diff.shape[1],1))
    pp = np.array(pp)
    pp = np.reshape(pp,(pp.shape[0],pp.shape[1],1))
    wcd = np.array(wcd)
    wcd = np.reshape(wcd,(wcd.shape[0],wcd.shape[1],1))
    wcn = np.array(wcn)
    wcn = np.reshape(wcn,(wcn.shape[0],wcn.shape[1],1))
        
    X =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,t_diff,d_diff,pp,nc,lat,long,wcd,wcn]
    scores = model.evaluate(X, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100)) 
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100)) 
    print("%s: %.2f%%" % (model.metrics_names[5], scores[5]*100))   


def process_test3(df, model,nn,cc,data_no_scale,backup2,test_2,data_no_scale_2):
    correct =0
    total =0
    x = df['userid']
    x = x.unique()
    feature = []
    X= []
    y = []
    total += 1
    m,d,p,yr,grp,dt,near_cls_grp,clsr_day,clstr_period,pp,clstr_frnd=[],[],[],[],[],[],[],[],[],[],[]
    lat,long=[],[]
    t_diff,d_diff = [],[]
    wcd,wcn=[],[]
    nc = []
    cof = []

    for i in x:
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])
        
        cof.append(sample['clsoffrnd'])

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        y.append(out)
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        
        
        week_data =pd.read_csv("week_first.csv")
        week_data = week_data[week_data['userid'] == i]
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)
    
    data_no_scale = data_no_scale_2
    df = test_2
    x = df['userid']
    x = x.unique()
    
    for i in x:
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])

        cof.append(sample['clsoffrnd'])
        

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        y.append(out)
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        
        #wcd.append(week_cluster(i,backup2,0,time))
        #wcn.append(week_cluster(i,backup2,1,time))
        week_data = pd.read_csv("week_last.csv")
        week_data = week_data[week_data['userid'] == i]
        #week_data = week_data.head(1)
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)

    y = np.array(y)
    y = np_utils.to_categorical(y, nn)
    d = np.array(d)
    d = np.reshape(d,(d.shape[0],d.shape[1],1))
    nc = np.array(nc)
    nc = np.reshape(nc,(nc.shape[0],nc.shape[1],1))
    dt = np.array(dt)
    dt = np.reshape(dt,(dt.shape[0],dt.shape[1],1))
    p = np.array(p)
    p = np.reshape(p,(p.shape[0],p.shape[1],1))
    m = np.array(m)
    m = np.reshape(m,(m.shape[0],m.shape[1],1))
    yr = np.array(yr)
    yr = np.reshape(yr,(yr.shape[0],yr.shape[1],1))
    grp = np.array(grp)
    grp = np.reshape(grp,(grp.shape[0],grp.shape[1],1))
    near_cls_grp = np.array(near_cls_grp)
    near_cls_grp = np.reshape(near_cls_grp,(near_cls_grp.shape[0],near_cls_grp.shape[1],1))
    clsr_day = np.array(clsr_day)
    clsr_day = np.reshape(clsr_day,(clsr_day.shape[0],clsr_day.shape[1],1))
    clstr_period = np.array(clstr_period)
    clstr_period = np.reshape(clstr_period,(clstr_period.shape[0],clstr_period.shape[1],1))
    clstr_frnd = np.array(clstr_frnd)
    clstr_frnd = np.reshape(clstr_frnd,(clstr_frnd.shape[0],clstr_frnd.shape[1],1))
    lat = np.array(lat)
    lat = np.reshape(lat,(lat.shape[0],lat.shape[1],1))
    long = np.array(long)
    long = np.reshape(long,(long.shape[0],long.shape[1],1))
    t_diff = np.array(t_diff)
    t_diff = np.reshape(t_diff,(t_diff.shape[0],t_diff.shape[1],1))
    d_diff = np.array(d_diff)
    d_diff = np.reshape(d_diff,(d_diff.shape[0],d_diff.shape[1],1))
    pp = np.array(pp)
    pp = np.reshape(pp,(pp.shape[0],pp.shape[1],1))
    wcd = np.array(wcd)
    wcd = np.reshape(wcd,(wcd.shape[0],wcd.shape[1],1))
    wcn = np.array(wcn)
    wcn = np.reshape(wcn,(wcn.shape[0],wcn.shape[1],1))
       
    cof = np.array(cof)
    cof = np.reshape(cof,(cof.shape[0],cof.shape[1],1))

    ## this is current
    X =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,t_diff,d_diff,pp,nc,lat,long,wcd,wcn,cof]
    XX =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,pp,nc,lat,long,wcd,wcn,cof]
    scores = model.evaluate(X, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], scores[3]*100)) 
    print("%s: %.2f%%" % (model.metrics_names[4], scores[4]*100)) 
    print("%s: %.2f%%" % (model.metrics_names[5], scores[5]*100)) 

def process_test(df, model,nn,cc,data_no_scale,backup2,test_2,data_no_scale_2,a_user,backup):
    correct =0
    ohno=0
    total =0
    x = df['userid']
    x = x.unique()
    avg1,avg2, avg3,avg4 = 0.00,0.00,0.00,0.00
    elem = []
    poi = []
    feature = []
    in_five = 0
    ad = 0.00
    for i in x:
        X= []
        total += 1
        m,d,p,yr,grp,dt,near_cls_grp,clsr_day,clstr_period,pp,clstr_frnd=[],[],[],[],[],[],[],[],[],[],[]
        lat,long=[],[]
        t_diff,d_diff = [],[]
        wcd,wcn=[],[]
        nc = []
        cof = []
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])
        
        cof.append(sample['clsoffrnd'])

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        
        #wcd.append(week_cluster(i,backup2,0,time))
        #wcn.append(week_cluster(i,backup2,1,time))
        week_data =pd.read_csv("week_first.csv")
        week_data = week_data[week_data['userid'] == i]
        #week_data = week_data.head(1)
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)


        d = np.array(d)
        d = np.reshape(d,(d.shape[0],d.shape[1],1))
        nc = np.array(nc)
        nc = np.reshape(nc,(nc.shape[0],nc.shape[1],1))
        dt = np.array(dt)
        dt = np.reshape(dt,(dt.shape[0],dt.shape[1],1))
        p = np.array(p)
        p = np.reshape(p,(p.shape[0],p.shape[1],1))
        m = np.array(m)
        m = np.reshape(m,(m.shape[0],m.shape[1],1))
        yr = np.array(yr)
        yr = np.reshape(yr,(yr.shape[0],yr.shape[1],1))
        grp = np.array(grp)
        grp = np.reshape(grp,(grp.shape[0],grp.shape[1],1))
        near_cls_grp = np.array(near_cls_grp)
        near_cls_grp = np.reshape(near_cls_grp,(near_cls_grp.shape[0],near_cls_grp.shape[1],1))
        clsr_day = np.array(clsr_day)
        clsr_day = np.reshape(clsr_day,(clsr_day.shape[0],clsr_day.shape[1],1))
        clstr_period = np.array(clstr_period)
        clstr_period = np.reshape(clstr_period,(clstr_period.shape[0],clstr_period.shape[1],1))
        clstr_frnd = np.array(clstr_frnd)
        clstr_frnd = np.reshape(clstr_frnd,(clstr_frnd.shape[0],clstr_frnd.shape[1],1))
        lat = np.array(lat)
        lat = np.reshape(lat,(lat.shape[0],lat.shape[1],1))
        long = np.array(long)
        long = np.reshape(long,(long.shape[0],long.shape[1],1))
        t_diff = np.array(t_diff)
        t_diff = np.reshape(t_diff,(t_diff.shape[0],t_diff.shape[1],1))
        d_diff = np.array(d_diff)
        d_diff = np.reshape(d_diff,(d_diff.shape[0],d_diff.shape[1],1))
        pp = np.array(pp)
        pp = np.reshape(pp,(pp.shape[0],pp.shape[1],1))
        wcd = np.array(wcd)
        wcd = np.reshape(wcd,(wcd.shape[0],wcd.shape[1],1))
        wcn = np.array(wcn)
        wcn = np.reshape(wcn,(wcn.shape[0],wcn.shape[1],1))
       
        cof = np.array(cof)
        cof = np.reshape(cof,(cof.shape[0],cof.shape[1],1))

        ## this is current
        X =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,t_diff,d_diff,pp,nc,lat,long,wcd,wcn,cof]
        XX =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,pp,nc,lat,long,wcd,wcn,cof]
        yhat = model.predict(X, verbose=0)
        print("max valued label ", np.argmax(yhat), "actual ",out)
        #total += 1
        
        pred_lat = cc[np.argmax(yhat)][0]
        pred_lon = cc[np.argmax(yhat)][1]
        x = np.matmul(yhat,cc)
        dist_km = distance(act_lat,act_lon,pred_lat,pred_lon)
        avg1 += dist_km
        dist_km2 = distance(act_lat,act_lon,x[0][0],x[0][1])
        avg2 += dist_km2
        
        if (out == np.argmax(yhat)):
            correct +=1

            '''
            ###our section
            venue = []
        
            dd,mdd = avgdist_incluster(backup2,i,np.argmax(yhat),cc,time)
            sum,venue = numberOfcluster(backup2,i,math.ceil(mdd),cc,np.argmax(yhat),venue,time)
            venue = np.array(venue)
            venue = np.unique(venue, axis = 0)
            poi.append(sum)

            act = data_no_scale[data_no_scale['userid'] == i]
            act = act.to_numpy()
            score,feature = score_poi(i, backup2, instance[len-1][5], instance[len-1][10], np.argmax(yhat),venue,time,act[len-1][10],feature,backup)
            print("@@@@ actual poi ",act[len-1][10])
            for j in venue:
                print("venue ",j," score ", score[j])
            yes = is_in_topk(act[len-1][10],score,5)
            if (yes == 1):
                in_five += 1
            print("is it in top 5? ",yes)
            
            if (yes == 0):
                dx,xxx = min_dist_find(act[len-1][10],score,5)
                
                if xxx == 0:
                    ohno += 1
                    dx = 8.00
                ad += dx
            '''

                
            
       
    ad2 = 0.00     
    first = in_five
    data_no_scale = data_no_scale_2
    df = test_2
    x = df['userid']
    x = x.unique()
    fc = correct
    
    for i in x:
        X= []
        total += 1
        m,d,p,yr,grp,dt,near_cls_grp,clsr_day,clstr_period,pp,clstr_frnd=[],[],[],[],[],[],[],[],[],[],[]
        lat,long=[],[]
        t_diff,d_diff = [],[]
        wcd,wcn=[],[]
        nc = []
        cof = []
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])

        cof.append(sample['clsoffrnd'])
        

        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]
        
        #wcd.append(week_cluster(i,backup2,0,time))
        #wcn.append(week_cluster(i,backup2,1,time))
        week_data = pd.read_csv("week_last.csv")
        week_data = week_data[week_data['userid'] == i]
        #week_data = week_data.head(1)
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)


        d = np.array(d)
        d = np.reshape(d,(d.shape[0],d.shape[1],1))
        nc = np.array(nc)
        nc = np.reshape(nc,(nc.shape[0],nc.shape[1],1))
        dt = np.array(dt)
        dt = np.reshape(dt,(dt.shape[0],dt.shape[1],1))
        p = np.array(p)
        p = np.reshape(p,(p.shape[0],p.shape[1],1))
        m = np.array(m)
        m = np.reshape(m,(m.shape[0],m.shape[1],1))
        yr = np.array(yr)
        yr = np.reshape(yr,(yr.shape[0],yr.shape[1],1))
        grp = np.array(grp)
        grp = np.reshape(grp,(grp.shape[0],grp.shape[1],1))
        near_cls_grp = np.array(near_cls_grp)
        near_cls_grp = np.reshape(near_cls_grp,(near_cls_grp.shape[0],near_cls_grp.shape[1],1))
        clsr_day = np.array(clsr_day)
        clsr_day = np.reshape(clsr_day,(clsr_day.shape[0],clsr_day.shape[1],1))
        clstr_period = np.array(clstr_period)
        clstr_period = np.reshape(clstr_period,(clstr_period.shape[0],clstr_period.shape[1],1))
        clstr_frnd = np.array(clstr_frnd)
        clstr_frnd = np.reshape(clstr_frnd,(clstr_frnd.shape[0],clstr_frnd.shape[1],1))
        lat = np.array(lat)
        lat = np.reshape(lat,(lat.shape[0],lat.shape[1],1))
        long = np.array(long)
        long = np.reshape(long,(long.shape[0],long.shape[1],1))
        t_diff = np.array(t_diff)
        t_diff = np.reshape(t_diff,(t_diff.shape[0],t_diff.shape[1],1))
        d_diff = np.array(d_diff)
        d_diff = np.reshape(d_diff,(d_diff.shape[0],d_diff.shape[1],1))
        pp = np.array(pp)
        pp = np.reshape(pp,(pp.shape[0],pp.shape[1],1))
        wcd = np.array(wcd)
        wcd = np.reshape(wcd,(wcd.shape[0],wcd.shape[1],1))
        wcn = np.array(wcn)
        wcn = np.reshape(wcn,(wcn.shape[0],wcn.shape[1],1))

        cof = np.array(cof)
        cof = np.reshape(cof,(cof.shape[0],cof.shape[1],1))

        X =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,t_diff,d_diff,pp,nc,lat,long,wcd,wcn,cof] 
        XX =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,pp,nc,lat,long,wcd,wcn,cof] 
        yhat = model.predict(X, verbose=0)
        print("max valued label ", np.argmax(yhat), "actual ",out)
        
        pred_lat = cc[np.argmax(yhat)][0]
        pred_lon = cc[np.argmax(yhat)][1]
        x = np.matmul(yhat,cc)
        dist_km = distance(act_lat,act_lon,pred_lat,pred_lon)
        avg1 += dist_km
        dist_km2 = distance(act_lat,act_lon,x[0][0],x[0][1])
        avg2 += dist_km2
        
        if (out == np.argmax(yhat)):
            correct +=1

            '''
            ####our section
            venue = []
            dd,mdd = avgdist_incluster(backup2,i,np.argmax(yhat),cc,time)
            sum,venue = numberOfcluster(backup2,i,math.ceil(mdd),cc,np.argmax(yhat),venue,time)
            venue = np.array(venue)
            venue = np.unique(venue, axis = 0)
            poi.append(sum)

            act = data_no_scale[data_no_scale['userid'] == i]
            act = act.to_numpy()
            score,feature = score_poi(i, backup2, instance[len-1][5], instance[len-1][10], np.argmax(yhat),venue,time,act[len-1][10],feature,backup)
            print("@@@@ actual poi ",act[len-1][10])
            for j in venue:
                print("venue ",j," score ", score[j])
            yes = is_in_topk(act[len-1][10],score,5)
            if (yes == 1):
                in_five += 1
            print("is it in top 5? ",yes)
            
            if (yes == 0):
                
                dx,xxx = min_dist_find(act[len-1][10],score,5)
                ad2 += dx
                if xxx == 0:
                    ohno += 1
            '''
        
        #top_k_values, top_k_indices = tf.nn.top_k(yhat, k=3)
        #top_k = top_k_indices.numpy()

        #venue = []
        #dd,mdd = avgdist_incluster(backup2,i,top_k[0][0],cc)
        #sum,venue = numberOfcluster(backup2,i,math.ceil(mdd),cc,top_k[0][0],venue)

        #dd,mdd = avgdist_incluster(backup2,i,top_k[0][1],cc)
        #sum2,venue = numberOfcluster(backup2,i,math.ceil(dd),cc,top_k[0][1],venue)
        
        #dd,mdd = avgdist_incluster(backup2,i,top_k[0][2],cc)
        #sum2,venue = numberOfcluster(backup2,i,math.ceil(dd),cc,top_k[0][2],venue)
        
        #venue = np.array(venue)
        #print("venue ", venue)

        #poi.append(sum)
        #score = score_poi(i, backup2, instance[len-1][5], instance[len-1][10], np.argmax(yhat),venue,time)
        #act = data_no_scale[data_no_scale['userid'] == i]
        #act = act.to_numpy()
        #print("@@@@ actual poi ",act[len-1][10])
        #for j in venue:
        #    print("venue ",j," score ", score[j])
        #yes = is_in_topk(act[len-1][10],score,5)
        #if (yes == 1):
        #    in_five += 1
        #print("is it in top 5? ",yes)

    print("total ", total, "correct ", correct," percent accurate",(correct/total)*100.00)
    print("first cluster correct ",fc," second cluster correct ", correct - fc)
    
    '''
    print("total in top five ",in_five)
    print("percentage in top five ",(in_five/total) * 100.00)
    second = in_five - first
    print("first section true ",first," second section true ", second)
    
    print("average distance in first part ",ad/fc," average distance in second part ", ad2/(correct - fc))
    print("total average dist ", (ad+ad2)/(correct))
    print("ohno issues ",ohno)
    '''
    
    
    
    #feature = np.array(feature)
    #feature = pd.DataFrame(feature, columns = ['d1','d2','d3','d4','ad1','ad2','ad3','ad4','time_act','meet_imp','friend_choice','far','popularity','y'])
    #feature.to_csv("ffft.csv", index = False)
    


def process_test_feature(df, nn,cc,data_no_scale,backup2,test_2,data_no_scale_2):
    x = df['userid']
    x = x.unique()
    poi = []
    feature = []
    for i in x:
        X= []
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]

        
        print("int(out) ", int(out))
        venue = []
        dd,mdd = avgdist_incluster(backup2,i,int(out),cc)
        sum,venue = numberOfcluster(backup2,i,math.ceil(mdd),cc,int(out),venue)
        poi.append(sum)
        act = data_no_scale[data_no_scale['userid'] == i]
        act = act.to_numpy()
        #print("@@@@ actual poi ",act[len-1][10])
        feature = score_poi_test(i, backup2, instance[len-1][5], instance[len-1][10], int(out),venue,act[len-1][10], feature,time)
            
            
    data_no_scale = data_no_scale_2
    df = test_2
    x = df['userid']
    x = x.unique()
    for i in x:
        X= []
        ins = data_no_scale[data_no_scale.userid == i]
        instance = df[df.userid == i]

        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        ins = ins.to_numpy()
        act_lat = ins[len-1][1]
        act_lon = ins[len-1][2]

        
        print("int(out) ", int(out))
        dd,mdd = avgdist_incluster(backup2,i,int(out),cc)
        sum,venue = numberOfcluster(backup2,i,math.ceil(mdd),cc,int(out))
        poi.append(sum)
        act = data_no_scale[data_no_scale['userid'] == i]
        act = act.to_numpy()
        #print("@@@@ actual poi ",act[len-1][10])
        feature = score_poi_test(i, backup2, instance[len-1][5], instance[len-1][10], int(out),venue,act[len-1][10],feature,time)
    feature = np.array(feature)
    feature = pd.DataFrame(feature, columns = ['d1','d2','d3','d4','ad1','ad2','ad3','ad4','y'])
    feature.to_csv("feature.csv", index = False)


def training_model(df,model,nn,data_no_scale,backup2,data_no_scale_2,train_2):
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
    
    for i in x:
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])

        cof.append(sample['clsoffrnd'])
        


        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])
        
        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        #print("out ", out)
        y = np.append(y,out)
        
        #wcd.append(week_cluster(i,backup2,0,time))
        #wcn.append(week_cluster(i,backup2,1,time))
        week_data = pd.read_csv("week_first.csv")
        week_data = week_data[week_data['userid'] == i]
        #week_data = week_data.head(1)
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)


    df = train_2
    x = df['userid']
    x = x.unique()
    
    for i in x:
        instance = df[df.userid == i]
        sample = instance
        d.append(sample['day'])
        dt.append(sample['date'])
        m.append(sample['month'])
        yr.append(sample['year'])
        p.append(sample['period'])
        t_diff.append(sample['time_diff'])
        clsr_day.append(sample['clstr_ofthe_day'])
        clstr_period.append(sample['clstr_ofthe_period'])
        
        cof.append(sample['clsoffrnd'])


        sample = sample.head(5)
        lat.append(sample['latitude'])
        long.append(sample['longitude'])
        d_diff.append(sample['dist_diff'])
        grp.append(sample['cluster_grp'])
        near_cls_grp.append(sample['near_cluster'])
        pp.append(sample['prob_of_pres'])
        nc.append(sample['nearest_cluster'])
        clstr_frnd.append(sample['clstr_of_friend'])
        
        instance = instance.to_numpy()
        len = instance.shape[0]
        col = instance.shape[1]
        out = instance[len-1][11]
        time = instance[len-1][9]*365*24 + instance[len-1][6] * 30 * 24 + instance[len-1][7] * 24 + instance[len-1][8] 
        #print("out ", out)
        y = np.append(y,out)
        
        #wcd.append(week_cluster(i,backup2,0,time))
        #wcn.append(week_cluster(i,backup2,1,time))
        week_data = pd.read_csv("week_last.csv")
        week_data = week_data[week_data['userid'] == i]
        #week_data = week_data.head(1)
        wd = np.empty(7)
        wn = np.empty(7)
        for ind in week_data.index:
            wd[0],wd[1],wd[2]=int(week_data['wd1'][ind]),int(week_data['wd2'][ind]),int(week_data['wd3'][ind])
            wd[3],wd[4],wd[5],wd[6] = int(week_data['wd4'][ind]),int(week_data['wd5'][ind]),int(week_data['wd6'][ind]),int(week_data['wd7'][ind])

            wn[0],wn[1],wn[2]=int(week_data['wn1'][ind]),int(week_data['wn2'][ind]),int(week_data['wn3'][ind])
            wn[3],wn[4],wn[5],wn[6] = int(week_data['wn4'][ind]),int(week_data['wn5'][ind]),int(week_data['wn6'][ind]),int(week_data['wn7'][ind])
            break
        wd = np.array(wd)
        wn = np.array(wn)
        wcd.append(wd)
        wcn.append(wn)




    y = np.array(y)
    y = np_utils.to_categorical(y, nn) #converting output into categorical values

    d = np.array(d)
    d = np.reshape(d,(d.shape[0],d.shape[1],1))
    dt = np.array(dt)
    dt = np.reshape(dt,(dt.shape[0],dt.shape[1],1))
    p = np.array(p)
    p = np.reshape(p,(p.shape[0],p.shape[1],1))
    nc = np.array(nc)
    nc = np.reshape(nc,(nc.shape[0],nc.shape[1],1))
    m = np.array(m)
    m = np.reshape(m,(m.shape[0],m.shape[1],1))
    yr = np.array(yr)
    yr = np.reshape(yr,(yr.shape[0],yr.shape[1],1))
    grp = np.array(grp)
    grp = np.reshape(grp,(grp.shape[0],grp.shape[1],1))
    near_cls_grp = np.array(near_cls_grp)
    near_cls_grp = np.reshape(near_cls_grp,(near_cls_grp.shape[0],near_cls_grp.shape[1],1))
    clsr_day = np.array(clsr_day)
    clsr_day = np.reshape(clsr_day,(clsr_day.shape[0],clsr_day.shape[1],1))
    clstr_period = np.array(clstr_period)
    clstr_period = np.reshape(clstr_period,(clstr_period.shape[0],clstr_period.shape[1],1))
    clstr_frnd = np.array(clstr_frnd)
    clstr_frnd = np.reshape(clstr_frnd,(clstr_frnd.shape[0],clstr_frnd.shape[1],1))
    lat = np.array(lat)
    lat = np.reshape(lat,(lat.shape[0],lat.shape[1],1))
    long = np.array(long)
    long = np.reshape(long,(long.shape[0],long.shape[1],1))
    t_diff = np.array(t_diff)
    t_diff = np.reshape(t_diff,(t_diff.shape[0],t_diff.shape[1],1))
    d_diff = np.array(d_diff)
    d_diff = np.reshape(d_diff,(d_diff.shape[0],d_diff.shape[1],1))
    pp = np.array(pp)
    pp = np.reshape(pp,(pp.shape[0],pp.shape[1],1))
    wcd = np.array(wcd)
    wcd = np.reshape(wcd,(wcd.shape[0],wcd.shape[1],1))
    wcn = np.array(wcn)
    wcn = np.reshape(wcn,(wcn.shape[0],wcn.shape[1],1))
    cof = np.array(cof)
    cof = np.reshape(cof,(cof.shape[0],cof.shape[1],1))

    ## this is current
    X =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,t_diff,d_diff,pp,nc,lat,long,wcd,wcn,cof]
    XX =[d,p,grp,near_cls_grp,clsr_day,clstr_period,clstr_frnd,pp,nc,lat,long,wcd,wcn,cof]
    hist = model.fit(X,y, epochs=250, batch_size=64, verbose=2)
    return model,hist

cluster_centers = pd.read_csv("cluster_center_0.002.csv")
nn = len(cluster_centers)
data = pd.read_csv("data_final_first6.csv")
data_2 = pd.read_csv("data_final_last6.csv")
#train = data.head(6918)
#test = data.tail(1734)
cluster_centers = cluster_centers.to_numpy()
data_no_scale = pd.read_csv("data_before_scale_first6.csv")
data_no_scale_2 = pd.read_csv("data_before_scale_last6.csv")
#train_2 = data_2.head(6918)
#test_2 = data_2.tail(1734)
backup2 = pd.read_csv("backup2_first6.csv")
#print("cc shape ", cluster_centers.shape)

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


#a_user = pd.read_csv("all_user_period.csv")
#clstr = []z = Dropout(0.25+0.15)(z)
#for ind in a_user.index:
#    lat = a_user['latitude'][ind]
#    lon = a_user['longitude'][ind]
#    cl = clstr_find(lat,lon,cluster_centers,nn)
#    clstr.append(cl)
#a_user['cluster_grp'] = clstr
#a_user.to_csv("all_user_cluster.csv", index = False)

a_user = pd.read_csv("all_user_cluster.csv")

clsoffrnd = []
for ind in data.index:
    c = cluster_of_friend(data['userid'][ind],data['day'][ind],data['period'][ind],a_user,backup)
    clsoffrnd.append(c)
data['clsoffrnd'] = clsoffrnd

clsoffrnd = []
for ind in data_2.index:
    c = cluster_of_friend(data_2['userid'][ind],data_2['day'][ind],data_2['period'][ind],a_user,backup)
    clsoffrnd.append(c)
data_2['clsoffrnd'] = clsoffrnd

train = data.head(6918)
test = data.tail(1734)

train_2 = data_2.head(6918)
test_2 = data_2.tail(1734)

maxlen = 6
input_1=Input(shape=(maxlen,)) #for date
embed_1=Dense(10+5,input_dim=maxlen)(input_1)
em_1=Flatten()(embed_1)
em1_model=Model(inputs=input_1,outputs=em_1)

input_2=Input(shape=(maxlen,)) #for day
embed_2=Dense(10+5,input_dim=maxlen)(input_2)
em_2=Flatten()(embed_2)
em2_model=Model(inputs=input_2,outputs=em_2)

input_3=Input(shape=(maxlen,)) #for month_cat
embed_3=Dense(10+5,input_dim=maxlen)(input_3)
em_3=Flatten()(embed_3)
em3_model=Model(inputs=input_3,outputs=em_3)

input_4=Input(shape=(maxlen,)) #for year_cat
embed_4=Dense(10+5,input_dim=maxlen)(input_4)
em_4=Flatten()(embed_4)
em4_model=Model(inputs=input_4,outputs=em_4)

input_5=Input(shape=(maxlen-1,)) #for cluster_grp
embed_5=Dense(10+5+3,input_dim=maxlen-1)(input_5)
em_5=Flatten()(embed_5)
em5_model=Model(inputs=input_5,outputs=em_5)

input_6=Input(shape=(maxlen,)) #for period
embed_6=Dense(10+5,input_dim=maxlen)(input_6)
em_6=Flatten()(embed_6)
em6_model=Model(inputs=input_6,outputs=em_6)

input_7=Input(shape=(maxlen-1,)) #for near_cluster
embed_7=Dense(10+5+3,input_dim=maxlen-1)(input_7)
em_7=Flatten()(embed_7)
em7_model=Model(inputs=input_7,outputs=em_7)

input_8=Input(shape=(maxlen,)) #for cluster of the day
embed_8=Dense(10+5,input_dim=maxlen)(input_8)
em_8=Flatten()(embed_8)
em8_model=Model(inputs=input_8,outputs=em_8)

input_9=Input(shape=(maxlen,)) #for cluster of the period
embed_9=Dense(10+5,input_dim=maxlen)(input_9)
em_9=Flatten()(embed_9)
em9_model=Model(inputs=input_9,outputs=em_9)

input_91=Input(shape=(maxlen-1,)) #for cluster of top friend of the day and time
embed_91=Dense(10+5,input_dim=maxlen-1)(input_91)
em_91=Flatten()(embed_91)
em91_model=Model(inputs=input_91,outputs=em_91)

input_10=Input(shape=(maxlen-1,),dtype='float64') #for latitude
embed_10=Dense(4+5,input_dim=maxlen-1)(input_10)
em_10=Flatten()(embed_10)
em10_model=Model(inputs=input_10,outputs=em_10)

input_11=Input(shape=(maxlen-1,),dtype='float64') #for longitude
embed_11=Dense(4+5,input_dim=maxlen-1)(input_11)
em_11=Flatten()(embed_11)
em11_model=Model(inputs=input_11,outputs=em_11)

input_12=Input(shape=(maxlen,),dtype='float64') #for time_diff
embed_12=Dense(2+5,input_dim=maxlen)(input_12)
em_12=Flatten()(embed_12)
em12_model=Model(inputs=input_12,outputs=em_12)

input_13=Input(shape=(maxlen-1,),dtype='float64') #for dist_diff
embed_13=Dense(2+5,input_dim=maxlen-1)(input_13)
em_13=Flatten()(embed_13)
em13_model=Model(inputs=input_13,outputs=em_13)

input_14=Input(shape=(maxlen-1,),dtype='float64') #for prob_of_pres
embed_14=Dense(10+5,input_dim=maxlen-1)(input_14)
em_14=Flatten()(embed_14)
em14_model=Model(inputs=input_14,outputs=em_14)

input_15=Input(shape=(maxlen-1,)) #for nearest_cluster of a cluster
embed_15=Dense(10+5, input_dim=maxlen-1)(input_15)
em_15=Flatten()(embed_15)
em15_model=Model(inputs=input_15,outputs=em_15)

input_16=Input(shape=(7,)) #for week day cluster
embed_16=Dense(10+5,input_dim=7)(input_16)
em_16=Flatten()(embed_16)
em16_model=Model(inputs=input_16,outputs=em_16)

input_17=Input(shape=(7,)) #for week night cluster
embed_17=Dense(10+5,input_dim=7)(input_17)
em_17=Flatten()(embed_17)
em17_model=Model(inputs=input_17,outputs=em_17)

input_18=Input(shape=(maxlen,)) #for cluster of top friend
embed_18=Dense(10+5,input_dim=maxlen)(input_18)
em_18=Flatten()(embed_18)
em18_model=Model(inputs=input_18,outputs=em_18)

'''
##this is current actual model

combined = concatenate([em2_model.output, 
                        em6_model.output,em5_model.output, em7_model.output,em8_model.output,
                        em9_model.output, em91_model.output,em12_model.output,
                        em13_model.output, em14_model.output,em15_model.output,em10_model.output,em11_model.output,em16_model.output,em17_model.output,em18_model.output])
'''
combined = concatenate([em2_model.output, 
                        em6_model.output,em5_model.output, em7_model.output,em8_model.output,
                        em9_model.output, em91_model.output,em12_model.output,
                        em13_model.output,em14_model.output,em15_model.output,em10_model.output,em11_model.output,em16_model.output,em17_model.output,em18_model.output])






# this four line is for lstm and attention
#z = RepeatVector(1)(combined)
#z = LSTM(50,return_sequences=True,dropout=0.5,recurrent_dropout=0.2)(z)
#z = Attention()(z)
#z = Dropout(0.5)(z)
#z = Dense(10, activation='relu')(z)
#z = Dense(nn,activation='softmax')(z)


'''
#this section is for transformer block
#embed_dim = 903  # Embedding size for each token
embed_dim = 899  # Embedding size for each token
num_heads = 3  # Number of attention heads ###actual was 3
ff_dim = 8  # Hidden layer size in feed forward network inside transformer
z = RepeatVector(1)(combined)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
z = layers.Dropout(0.7)(z)
z = transformer_block(z)
z = layers.Dropout(0.7)(z) ###new
z = layers.GlobalAveragePooling1D()(z)
z = layers.Dropout(0.7)(z) 
z = Dense(nn,activation='softmax')(z)
'''

z = RepeatVector(1)(combined)
#z = Dropout(0.5)(z)
z = Conv1D(200, 1, activation = 'relu')(z)
#z = Dense(500, activation='relu')(z)
z = Dropout(0.25)(z)
z = Conv1D(100, 1, activation = 'relu')(z)
####z = Dropout(0.25)(z)

#z = Conv1D(80, 1, activation = 'relu')(z) ##new
#z = Dropout(0.10)(z) ##new

z = Dense(50, activation='relu')(z)
z = tf.keras.layers.SimpleRNN(1000, return_sequences=False, dropout=0.15,recurrent_dropout=0.15)(z)
#z = tf.keras.layers.SimpleRNN(300, return_sequences=False, dropout=0,recurrent_dropout=0)(z)
z = Dense(nn,activation='softmax')(z)

##this is current actual model
model = Model(inputs=[em2_model.input, 
em6_model.input,em5_model.input, em7_model.input, em8_model.input,
em9_model.input,em91_model.input,em12_model.input,em13_model.input ,em14_model.input,em15_model.input,em10_model.input,em11_model.input,em16_model.input,em17_model.input,em18_model.input], outputs=z)

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TopKCategoricalAccuracy()])
model,hist=training_model(train,model,nn,data_no_scale,backup2,data_no_scale_2,train_2)


process_test(test, model,nn,cluster_centers,data_no_scale,backup2,test_2,data_no_scale_2,a_user,backup)
process_test3(test, model,nn,cluster_centers,data_no_scale,backup2,test_2,data_no_scale_2)
print("test RNN ")
