import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import tensorflow as tf
import math

from sklearn.cluster import MeanShift,KMeans, estimate_bandwidth

def clustering(data):
    #coordinate = data.as_matrix(columns = ['latitude','longitude'])
    coordinate = data[['latitude','longitude']].to_numpy()
    bandwidth = estimate_bandwidth(coordinate, quantile = 0.002) #reducing quantile value increases number of clusters
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(coordinate)
    labels = meanshift.labels_
    cluster_centers = meanshift.cluster_centers_
    n_clusters_ = len(np.unique(labels))
    data['cluster_grp'] = np.nan
    for i in range(len(coordinate)):
        data['cluster_grp'].iloc[i] = labels[i]
    return data,n_clusters_, cluster_centers

def distance(lat1,lon1, lat2, lon2):
    #lat1 = pos1[:, 0]
    #lon1 = pos1[:, 1]
    #lat1, lon1 = origin
    #lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    
    return d
def clstr_find(lat,long,cluster_center,nn):
    min_dist = float('inf') #100000000
    ind = 0
    for i in range(nn):
        dist = distance(lat,long,cluster_center[i][0],cluster_center[i][1])
        if(dist < min_dist):
            min_dist = dist
            ind = i
    return ind

def find_near_cluster(data,all_user,cluster_center,nn):
    cluster = []
    for ind in data.index:
        t = data['time'][ind]
        u = data['userid'][ind]
        df = all_user[all_user['userId'] == u]
        df1 = df[df['time'] < t]
        df2 = df[df['time'] > t]
        if(len(df1)==0):
            cluster.append(data['cluster_grp'][ind])
        else:
            df1 = df1.tail(1)
            df1 = df1.to_numpy()
            lat = df1[0][1]
            long = df1[0][2]
            clstr = clstr_find(lat,long,cluster_center,nn)
            cluster.append(clstr)
    #data['near_cluster'] = cluster
    return cluster

def cutlen(data,len): #for data greater then length len
    value_counts = data['userid'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    #print(counts)
    df = pd.DataFrame(columns=['userid','latitude','longitude','venue_catagory','day','month','date','hour','year','holiday','venue_cat','City','period'])
    counts = counts.drop(counts.columns[[1]],1)
    counts = counts.to_numpy()
    counts = counts.flatten()
    for i in counts:
        d = data[data.userid == i]
        d = d.head(len)
        df = df.append(d, ignore_index = True)
    return df
def cutlen2(data,len): #for data greater then length len
    value_counts = data['userid'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    #print(counts)
    df = pd.DataFrame(columns=['userid','latitude','longitude','venue_catagory','day','month','date','hour','year','holiday','venue_cat','City','period'])
    counts = counts.drop(counts.columns[[1]],1)
    counts = counts.to_numpy()
    counts = counts.flatten()
    for i in counts:
        d = data[data.userid == i]
        d = d.tail(len)
        df = df.append(d, ignore_index = True)
    return df
def save_index_for_frnd(df, length):
    mydict = {}
    user = df['userid'].unique()
    for i in user:
        x = df[df['userid'] == i]
        cnt,ind = 0,0
        for j in x.index:
            ind += 1
            if cnt == 0:
                cnt += 1
                pm,pd,ph,py,pv = x['month'][j],x['date'][j],x['hour'][j],x['year'][j],x['venue_cat'][j]
                continue
            if (pm == x['month'][j] and pd == x['date'][j] and ph == x['hour'][j] and py == x['year'][j] and pv == x['venue_cat'][j]):
                continue
            else:
                pm,pd,ph,py,pv = x['month'][j],x['date'][j],x['hour'][j],x['year'][j],x['venue_cat'][j]
                cnt += 1
            if cnt == length:
                mydict[i] = ind
                break
        if cnt < length:
            mydict[i] = ind
    return mydict
''' finding top cluster for a user on a given day'''
def top_clstr_day(df,day,user,a_user,time): ##df -> take backup2, remove userid2,revisit, POI_total_check. remove duplicate. do clustering --> df
    df = df[df['userid'] == user]
    df = df[df['time'] < time]
    if len(df) != 0:
        df = df[df['day'] == day]
        if len(df) != 0:
            value_counts = df['cluster_grp'].value_counts()
            counts = pd.DataFrame(value_counts)
            counts = counts.reset_index()
            counts.columns = ['unique_id', 'counts']
            counts = counts.to_numpy()
            return counts[0][0]



    df = a_user[a_user['userId'] == user]
    df = df[df['time'] < time]
    if len(df) == 0:
        print("2")
    else:
        df = df[df['day'] == day]
    if(len(df) == 0):
        df = a_user[a_user['userId'] == user]
        df = df[df['time'] < time]
        if len(df) == 0:
            print("ohh my god")
            return 0
    value_counts = df['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    #counts = counts.head(1)
    #print("counts %%%%%%%%% ",counts)
    counts = counts.to_numpy()
    return counts[0][0]
''' finding top cluster for a user on a given period'''
def top_clstr_period(df,period,user,a_user,time): ##df --> backup2
    df = df[df['userid'] == user]
    df = df[df['time'] < time]
    if len(df) != 0:
        df = df[df['period'] == period]
        if len(df) != 0:
            value_counts = df['cluster_grp'].value_counts()
            counts = pd.DataFrame(value_counts)
            counts = counts.reset_index()
            counts.columns = ['unique_id', 'counts']
            counts = counts.to_numpy()
            return counts[0][0]


    df = a_user[a_user['userId'] == user]
    df = df[df['time'] < time]
    if len(df) == 0:
        print("2")
    else:
        df = df[df['period'] == period]
    if(len(df) == 0):
        df = a_user[a_user['userId'] == user]
        df = df[df['time'] < time]
        if len(df) == 0:
            print("ohh my god")
            return 0
    value_counts = df['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']
    #counts = counts.head(1)
    #print("counts %%%%%%%%% ",counts)
    counts = counts.to_numpy()
    return counts[0][0]

''' this functions purpose is to find the probability of being in a particular cluster (clstr) at a given day and period '''
def prob_of_presense(day,period,user,df,clstr): ##df --> all meet up data // backup2
    #df = df[df['userid'] == user] ###old line
    df = pd.read_csv("all_user_cluster.csv") #newline
    df = df[df['userId'] == user]
    length = len(df)
    df = df[df['day']==day]
    df = df[df['period']==period]
    df = df[df['cluster_grp'] == clstr]
    return len(df)/(length * 1.00)


''' this functions purpose is to first find the top friend for a particular given day. after that, find the near cluster
of the friend to the close given time '''
def cluster_friend(user,day,period,all_user,cluster_center,nn,t, backup): ##df --> all meet up data // backup
    xx = backup[backup['userid'] == user]
    siz = len(xx)
    #xxx = xx.head(siz-1) #### change this line
    xxx = xx
    df = xxx
    df = df[df['day'] == day]
    #df = df[df['period'] == period]
    if(len(df) == 0):
        df = xxx[xxx['day'] == day]
        print("gotttttttttta")

    value_counts = df['userid2'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['frnd', 'counts']
    
    if(len(counts) == 0):
        print("ohh noo ", user," ",day," ",period," ",len(xx)," ",len(df))
    '''
    if len(counts) >= 2:
        counts = counts.to_numpy()
        top_friend = counts[0][0]
        top_friend2 = counts[1][0]
        df = all_user[all_user['userId'] == top_friend]
        df1 = df[df['time'] < t]
        df2 = df[df['time'] > t]
        if(len(df1)==0):
            df1 = df2
        df1 = df1.tail(1)
        df1 = df1.to_numpy()
        lat = df1[0][1]
        long = df1[0][2]
        clstr = clstr_find(lat,long,cluster_center,nn)

        df = all_user[all_user['userId'] == top_friend2]
        df1 = df[df['time'] < t]
        df2 = df[df['time'] > t]
        if(len(df1)==0):
            df1 = df2
        df1 = df1.tail(1)
        df1 = df1.to_numpy()
        lat = df1[0][1]
        long = df1[0][2]
        clstr2 = clstr_find(lat,long,cluster_center,nn)
        if (clstr == clstr2):
            return clstr,cnt
        else:
            cnt += 1
            return clstr,cnt
    '''
    counts = counts.to_numpy()
    top_friend = counts[0][0]
    df = all_user[all_user['userId'] == top_friend]
    df1 = df[df['time'] < t]
    df2 = df[df['time'] > t]
    if(len(df1)==0):
        print("oh nooo it is a shit!!!")
        #df1 = df2
        df1 = all_user[all_user['time'] < t] ##added on 23 dec, 2021
        if len(df1) == 0:
            df1 = all_user.head(1)
    df1 = df1.tail(1)
    df1 = df1.to_numpy()
    lat = df1[0][1]
    long = df1[0][2]
    clstr = clstr_find(lat,long,cluster_center,nn)
    return clstr


##correction
def numberOfcluster(backup2,user,length,cc,cluster,pois,time):

    data = pd.read_csv("user_poi.csv")
    #data = data[data['time'] < time]
    data = data[data['userId'] == user]
    #data = data.drop(data.columns[[0,3,4,5,6,7,8,9,10,11,12,13]],1)
    data = data.drop(data.columns[[0,3,4,5,6,7,8]],1)
    data = data.drop_duplicates()
    sum = 0
    #pois = []
    for ind in data.index:
        lat = data['latitude'][ind]
        lon = data['longitude'][ind]
        dist = distance(lat,lon,cc[cluster][0],cc[cluster][1])
        if (dist <= length):
            sum += 1
            pois.append(data['venueid'][ind])
    #pois = np.array(pois)
    #pois = np.unique(pois, axis = 0)
    return sum,pois


'''
##actual
def numberOfcluster(backup2,user,length,cc,cluster,pois,time):
    data = backup2[backup2['userid'] == user]
    #data = data[data['cluster_grp'] == cluster] ####new
    #data = data.drop(data.columns[[0,3,4,5,6,7,8,9,10,11,12,13]],1)
    data = data.drop(data.columns[[0,4,5,6,7,8,9,11,12,13,14]],1)
    data = data.drop_duplicates()
    sum = 0
    #pois = []
    for ind in data.index:
        lat = data['latitude'][ind]
        lon = data['longitude'][ind]
        dist = distance(lat,lon,cc[cluster][0],cc[cluster][1])
        
        if (dist <= length):
            sum += 1
            pois.append(data['venue_cat'][ind])
        
    #pois = np.array(pois)
    #pois = np.unique(pois, axis = 0)

    return sum,pois
'''


def avgdist_incluster(backup2,user,cluster,cc,time):
    data = pd.read_csv("all_user_cluster.csv")
    data = data[data['userId'] == user]
    #data = data[data['time'] < time]
    data = data[data['cluster_grp'] == cluster]

    data = data.drop(data.columns[[0,3,4,5,6,7,8,9,10]],1)
    data = data.drop_duplicates()

    length = len(data)
    dist = 0.00
    maxdist = 0.00
    print("user  ", user," cluster ",cluster)
    if (length == 0):
        print("ohh shit!")
        return 0.00,0.00
    for ind in data.index:
        d = distance(data['latitude'][ind],data['longitude'][ind],cc[cluster][0],cc[cluster][1])
        dist += d
        if d >= maxdist:
            maxdist = d
    return dist / length, maxdist



'''
###actual
def avgdist_incluster(backup2,user,cluster,cc,time):
    data = backup2[backup2['userid'] == user]
    data = data[data['cluster_grp'] == cluster]
    length = len(data)
    dist = 0.00
    maxdist = 0.00
    print("user  ", user," cluster ",cluster)
    if (length == 0):
        print("ohh shit!")
        return 0.00,0.00
    for ind in data.index:
        d = distance(data['latitude'][ind],data['longitude'][ind],cc[cluster][0],cc[cluster][1])
        dist += d
        if d >= maxdist:
            maxdist = d
    return dist / length, maxdist
'''

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
def top_clstr_day_period(data,day,user,period,time): 
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
    '''
    data = pd.read_csv("all_user_cluster.csv") ###newline
    data = data[data['userId'] ==user]
    data = data[data['time'] < time]
    df = data[data['day'] == day]
    df1 = df[df['period'] == period]'''

    value_counts = df1['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']

    if len(counts) > 0:
        counts = counts.to_numpy()
        return counts[0][0]
    
    df1 = df

    value_counts = df1['cluster_grp'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['unique_id', 'counts']

    if len(counts) > 0:
        counts = counts.to_numpy()
        return counts[0][0]
    
    return 0 

def week_cluster(user,backup2,period,time):
    #data = backup2[backup2['userid'] == user] ### old
    
    data = pd.read_csv("all_user_cluster.csv") ###newline
    data = data[data['userId'] ==user] ###new line
    data = data[data['time'] < time] ### new
    df = np.empty(7)
    for i in range(7):
        d = top_clstr_day_period2(data,i,user,period)
        df[i] = d
    df = np.array(df)
    return df

def nearest_cluster_func(cc,clstr,nn):
    min_dist = float('inf')
    ind = -1
    for i in range(nn):
        if i == clstr:
            continue
        dist = distance(cc[i][0],cc[i][1],cc[clstr][0],cc[clstr][1])
        if (dist <= min_dist):
            min_dist = dist
            ind = i
    return ind
def score_poi22(user, backup2, day, period, cluster,pois,tt,act,feature,backup):
    all_user = pd.read_csv("all_user_period.csv")
    score = {}
    for i in pois:
        score[i] = 0.00
    print("**** for user ",user)
    for i in pois:
        data = backup2[backup2['userid'] == user]
        data = data[data['time'] < tt]
        length = len(data)
        data_1 = data[data['venue_cat'] == i] #for meetup probability at poi i
        data_2 = data_1[data_1['day'] == day] # for meetup probability at poi i on day "day"
        data_3 = data_1[data_1['period'] == period] # for meetup probability at poi i on period "period"
        data_4 = data_2[data_2['period'] == period]  # for meetup probability at poi i on period "period" and day "day"

        
        ##for all data
        lat, lon = 0.00, 0.00
        for ii in data_1.index:
            lat, lon = data_1['latitude'][ii], data_1['longitude'][ii]
            break


        ###for poi popularity
        total_len = len(all_user)
        poi_total = all_user[all_user['latitude'] == lat]
        poi_total = poi_total[poi_total['longitude'] == lon]
        poi_popularity = len(poi_total)/total_len


        all_user = all_user[all_user['userId'] == user]
        time = []
        for ind in all_user.index:
            mon = all_user['month'][ind]
            dat = all_user['date'][ind]
            year = all_user['year'][ind]
            ho = all_user['hour'][ind] 
            t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
            time.append(t)
        all_user['time'] = time 
        all_user = all_user[all_user['time'] < tt]
        length2 = len(all_user)
        au_data1 = all_user[all_user['latitude'] == lat]
        au_data1 = all_user[all_user['longitude'] == lon]
        au_data2 = au_data1[au_data1['day'] == day]
        au_data3 = au_data1[au_data1['period'] == period]
        au_data4 = au_data2[au_data2['period'] == period]
        if length2 == 0:
            ad1,ad2,ad3,ad4 = 0.00,0.00,0.00,0.00
        else:
            ad1,ad2,ad3,ad4 = len(au_data1)/length2,len(au_data2)/length2,len(au_data3)/length2,len(au_data4)/length2

        if length == 0:
            d1,d2,d3,d4 = 0.00,0.00,0.00,0.00
        else:
            d1,d2,d3,d4 = len(data_1)/length,len(data_2)/length,len(data_3)/length,len(data_4)/length
        print("for poi ",i," d1 d2 d3 d4 ",d1," ",d2," ",d3," ",d4)
        print("for poi ",i," ad1 ad2 ad3 ad4 ",ad1," ",ad2," ",ad3," ",ad4)
        ##score[i] = d1 * 15000 + d2 * 30000 + d3 * 22000 + d4 * 40000 + ad1 * 2000 + ad2 * 14000 + ad3 * 4000 + ad4 * 18000

        score[i] = d1 * 4200 + d2 * 2000 + d3 * 5000 + d4 * 3500 + ad1 * 3000 + ad2 * 2500 + ad3 * 4000 + ad4 * 3700 + poi_popularity *  2000
    return score,feature

def score_poi(user, backup2, day, period, cluster,pois,tt,act,feature,backup):
    #all_user = pd.read_csv("all_user_period.csv")
    all_user = pd.read_csv("all_user_poi.csv")
    pp = pd.read_csv("poi_feature.csv")
    frnd = backup[backup['userid'] == user]
    frnd = frnd['userid2'].unique()
    #user_data = pd.read_csv("user_poi.csv")
    user_data = pd.read_csv("all_user_poi.csv")
    user_data2 = user_data
    user_data = user_data[user_data['userId'] == user]
    rogg = pd.read_csv("US_ROG.csv")
    score = {}
    for i in pois:
        score[i] = 0.00
    print("**** for user ",user)
    
    for i in pois:
        data = backup2[backup2['userid'] == user]
        data = data[data['time'] < tt]
        length = len(data)
        data_1 = data[data['venue_cat'] == i] #for meetup probability at poi i
        data_2 = data_1[data_1['day'] == day] # for meetup probability at poi i on day "day"
        data_3 = data_1[data_1['period'] == period] # for meetup probability at poi i on period "period"
        data_4 = data_2[data_2['period'] == period]  # for meetup probability at poi i on period "period" and day "day"

        
        ##for all data
        lat, lon = 0.00, 0.00
        for ii in data_1.index:
            lat, lon = data_1['latitude'][ii], data_1['longitude'][ii]
            break

        '''
        ###for poi popularity
        total_len = len(all_user)
        poi_total = all_user[all_user['latitude'] == lat]
        poi_total = poi_total[poi_total['longitude'] == lon]
        poi_popularity = len(poi_total)/total_len
        '''

        
        ###for poi popularity
        total_len = len(user_data2)
        poi_total = user_data2
        poi_total = poi_total[poi_total['venueid'] == i]
        poi_popularity = len(poi_total)/total_len

        '''
        all_user = all_user[all_user['userId'] == user]
        time = []
        for ind in all_user.index:
            mon = all_user['month'][ind]
            dat = all_user['date'][ind]
            year = all_user['year'][ind]
            ho = all_user['hour'][ind] 
            t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
            time.append(t)
        all_user['time'] = time '''
        
        ####new
        all_user = user_data[user_data['userId'] == user]


        all_user = all_user[all_user['time'] < tt] ####%%%%&&&& newly added
        length2 = len(all_user)
        #au_data1 = all_user[all_user['latitude'] == lat]
        #au_data1 = all_user[all_user['longitude'] == lon]

        prd = []
        for ind in all_user.index:
            h = all_user['hour'][ind]
            if h >= 7 and h <= 19: #day
                prd.append(0)
            else:
                prd.append(1)
        all_user['period'] = prd

        au_data1 = all_user[all_user['venueid'] == i]
        au_data2 = au_data1[au_data1['day'] == day]
        au_data3 = au_data1[au_data1['period'] == period]
        au_data4 = au_data2[au_data2['period'] == period]
        if length2 == 0:
            ad1,ad2,ad3,ad4 = 0.00,0.00,0.00,0.00
        else:
            ad1,ad2,ad3,ad4 = len(au_data1)/length2,len(au_data2)/length2,len(au_data3)/length2,len(au_data4)/length2

        if length == 0:
            d1,d2,d3,d4 = 0.00,0.00,0.00,0.00
        else:
            d1,d2,d3,d4 = len(data_1)/length,len(data_2)/length,len(data_3)/length,len(data_4)/length
        print("for poi ",i," d1 d2 d3 d4 ",d1," ",d2," ",d3," ",d4)
        print("for poi ",i," ad1 ad2 ad3 ad4 ",ad1," ",ad2," ",ad3," ",ad4)


        
        #################new features
        
        p = pp[pp['venueid'] == i]
        if (len(p) != 1):
            print("alert!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        day_act = 0.00
        night_act = 0.00
        meet_imp = 0.00
        for ix in p.index:
            day_act = p['day_act'][ix]
            night_act = p['night_act'][ix]
            meet_imp = p['meet_imp'][ix]
        time_act = night_act
        if period == 0:
            ###time_act = day_act changing this line
            time_act = 1.00 - night_act
        
        '''
        #####cluster popularity
        xp = pd.read_csv("all_user_cluster.csv")
        xp = xp[xp['latitude'] == lat]
        xp = xp[xp['longitude'] == lon]
        loc = 0
        for ssp in xp.index:
            loc = xp['cluster_grp'][ssp]
            break
        xp = pd.read_csv("all_user_cluster.csv")
        xp = xp[xp['userId'] == user]
        pres = xp[xp['cluster_grp'] == loc]
        anomaly = len(pres)/len(xp)'''
        
        score[i] = d1 * 2500 + d2 * 1500 + d3 * 2900 + d4 * 1700 + ad1 * 2300 + ad2 * 2700 + ad3 * 2900 + ad4 * 3300  ##52.59, w/p poi_popularity, 52.76 with Poi_pop 1000, 52.94 with 5000
        #score[i] = d1 * 10000 + d2 * 30000 + d3 * 22000 + d4 * 40000 + ad1 * 2000 + ad2 * 14000 + ad3 * 4000 + ad4 * 18000 + poi_popularity * 5000 #with poi 15000 -> 52.24

        
        ###actual
        #score[i] =(d1 * 15000 + d2 * 30000 + d3 * 22000 + d4 * 40000 + ad1 * 2000 + ad2 * 14000 + ad3 * 4000 + ad4 * 18000 + poi_popularity * 12000 + time_act * 4000) ##53.28 wo poi_p,with poi 12000 -> 53.97, with 20000 -> 51.9
        #score[i] += (time_act * 4000 + meet_imp * 10)

        '''
        ### new metrices
        score[i] = d1 * 2500 + d2 * 1500 + d3 * 2900 + d4 * 1700 ##accuracy 28%
        score[i] += ad1 * 2300 + ad2 * 2700 + ad3 * 2900 + ad4 * 3300 + time_act * 500 
        score[i] += meet_imp * 200 + leng * 2000 + far * 600 + poi_popularity * 1700'''
        
        '''
        score[i] = d1 * 15000 + d2 * 30000 + d3 * 22000 + d4 * 40000 ###accuracy: 27.68
        score[i] += ad1 * 2000 + ad2 * 14000 + ad3 * 4000 + ad4 * 18000 + time_act * 4000
        score[i] += meet_imp * 10 + leng * 1700 + (far/1000.00) * 500 + poi_popularity * 4000'''

        '''
        score[i] = d1 * 2500 + d2 * 1500 + d3 * 2900 + d4 * 1700 ##accuracy 29.23% top five
        score[i] += ad1 * 2300 + ad2 * 2700 + ad3 * 2900 + ad4 * 3300 + time_act * 500 
        score[i] += (meet_imp/100.00) * 200 + leng * 2000 + (far/100.00) * 600 + poi_popularity * 1700'''
        
        '''
        score[i] = d1 * 2500 + d2 * 1500 + d3 * 2900 + d4 * 1700
        score[i] += ad1 * 2300 + ad2 * 2700 + ad3 * 2900 + ad4 * 3300 + time_act * 500 
        score[i] += (meet_imp/100.00) * 200 + leng * 2000 + (far/100.00) * 600 + poi_popularity * 1700 '''

        ###$score[i] = d1 * 2500 + d2 * 1500 + d3 * 2900 + d4 * 1700
        ###$score[i] += ad1 * 2300 + ad2 * 2700 + ad3 * 2900 + ad4 * 3300 + time_act * 500 
        
        
        leng = 0.00 
        
        for j in frnd:
            frnd_poi2 = user_data2[user_data2['userId'] == j]
            frnd_poi = frnd_poi2[frnd_poi2['venueid'] == i]
            leng += len(frnd_poi)/len(frnd_poi2)
        
        rog = rogg[rogg['user'] == user]
        r = 0.00
        clt,cln = 0.00,0.00
        for rg in rog.index:
            r = rog['ROG'][rg]
            clt = rog['center_lat'][rg]
            cln = rog['center_lon'][rg]
        far = distance(lat,lon,clt,cln)
        far = r - far 
        

        print("for poi ",i," time_act meet_imp friend_choice dist_rog popularity",time_act," ",meet_imp," ",leng," ",far," ",poi_popularity)
        score[i] += time_act * 500 + (meet_imp/100.00) * 200 + leng * 1500  + poi_popularity * 500 #leng * 1500 

        #score[i] = score[i] + anomaly * 5000
        '''
        if anomaly > 0.00:
            score[i] = score[i] + anomaly * 5000
        else:
            score[i] = 0.00'''
        
        a = np.empty(14)
        a[0],a[1],a[2],a[3]=d1,d2,d3,d4
        a[4],a[5],a[6],a[7]=ad1,ad2,ad3,ad4
        a[8] = time_act
        a[9] = meet_imp
        a[10] = leng
        a[11] = far
        a[12] = poi_popularity
        if(act == i):
            a[13] = 1
        else:
            a[13] = 0
        feature.append(a)
    
    return score,feature

def is_in_topk(cluster,score,k):
    cnt = 0
    yes = 0
    prev = -1
    for key, value in sorted(score.items(), key=lambda kv: kv[1], reverse=True):
        '''
        if prev == -1:
            prev = value
            cnt += 1
        else:
            if value == prev:
                vvv = 1
            else:
                cnt += 1
                if cnt > k:
                    break
        if key == cluster:
            yes = 1
            break
        
        '''
        
        if key == cluster:
            yes = 1
            break
        cnt += 1
        if cnt == k:
            break 
    return yes

def min_dist_find(cluster,score,k):
    cnt = 0
    xxx = 0
    data = pd.read_csv("POI_US.csv")
    mndst = float('inf')
    for key, value in sorted(score.items(), key=lambda kv: kv[1], reverse=True):
        x1 = data[data['venueid']==key]
        x1 = x1.to_numpy()
        x2 = data[data['venueid']==cluster]
        x2 = x2.to_numpy()
        d = distance(x1[0][1], x1[0][2], x2[0][1], x2[0][2])
        if d < mndst:
            mndst = d
            xxx = 1
        cnt += 1
        if cnt == k:
            break 
    return mndst,xxx

def score_poi_test(user, backup2, day, period, cluster,pois,act, feature,tt):
    all_user = pd.read_csv("all_user_period.csv")
    for i in pois:
        data = backup2[backup2['userid'] == user]
        data = data[data['time'] < tt]
        length = len(data)
        data_1 = data[data['venue_cat'] == i] #for meetup probability at poi i
        data_2 = data_1[data_1['day'] == day] # for meetup probability at poi i on day "day"
        data_3 = data_1[data_1['period'] == period] # for meetup probability at poi i on period "period"
        data_4 = data_2[data_2['period'] == period]  # for meetup probability at poi i on period "period" and day "day"

        
        ##for all data
        lat, lon = 0.00, 0.00
        for ii in data_1.index:
            lat, lon = data_1['latitude'][ii], data_1['longitude'][ii]
            break
        ###for poi popularity
        total_len = len(all_user)
        poi_total = all_user[all_user['latitude'] == lat]
        poi_total = poi_total[poi_total['longitude'] == lon]
        poi_popularity = len(poi_total)/total_len

        all_user = all_user[all_user['userId'] == user]
        time = []
        for ind in all_user.index:
            mon = all_user['month'][ind]
            dat = all_user['date'][ind]
            year = all_user['year'][ind]
            ho = all_user['hour'][ind] 
            t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
            time.append(t)
        all_user['time'] = time 
        all_user = all_user[all_user['time'] < tt]
        length2 = len(all_user)
        au_data1 = all_user[all_user['latitude'] == lat]
        au_data1 = all_user[all_user['longitude'] == lon]
        au_data2 = au_data1[au_data1['day'] == day]
        au_data3 = au_data1[au_data1['period'] == period]
        au_data4 = au_data2[au_data2['period'] == period]
        if length2 == 0:
            ad1,ad2,ad3,ad4 = 0.00,0.00,0.00,0.00
        else:
            ad1,ad2,ad3,ad4 = len(au_data1)/length2,len(au_data2)/length2,len(au_data3)/length2,len(au_data4)/length2

        if length == 0:
            d1,d2,d3,d4 = 0.00,0.00,0.00,0.00
        else:
            d1,d2,d3,d4 = len(data_1)/length,len(data_2)/length,len(data_3)/length,len(data_4)/length
        a = np.empty(10)
        a[0],a[1],a[2],a[3]=d1,d2,d3,d4
        a[4],a[5],a[6],a[7]=ad1,ad2,ad3,ad4
        a[8] = poi_popularity
        if(act == i):
            a[9] = 1
        else:
            a[9] = 0
        feature.append(a)
    return feature

def cluster_of_friend(user,day,period,all_user,backup): ###########
    xx = backup[backup['userid'] == user]
    siz = len(xx)
    #xxx = xx.head(siz-1) #### change this line
    xxx = xx
    df = xxx
    df = df[df['day'] == day]
    '''
    df = df[df['period'] == period]
    if(len(df) == 0):
        df = xxx[xxx['day'] == day]
        print("gotttttttttta")'''

    value_counts = df['userid2'].value_counts()
    counts = pd.DataFrame(value_counts)
    counts = counts.reset_index()
    counts.columns = ['frnd', 'counts']
    
    if(len(counts) == 0):
        print("ohh noo ", user," ",day," ",period," ",len(xx)," ",len(df))
    counts = counts.to_numpy()
    top_friend = counts[0][0]
    df = all_user[all_user['userId'] == top_friend]
    
    return top_clstr_day_period2(df,day,top_friend,period)


def score_poi_last(user, backup2, day, period, cluster,pois,tt):
    all_user = pd.read_csv("all_user_period.csv")
    score = {}
    for i in pois:
        score[i] = 0.00
    print("**** for user ",user)
    for i in pois:
        data = backup2[backup2['userid'] == user]
        data = data[data['time'] < tt]
        length = len(data)
        data_1 = data[data['venue_cat'] == i] #for meetup probability at poi i
        data_2 = data_1[data_1['day'] == day] # for meetup probability at poi i on day "day"
        data_3 = data_1[data_1['period'] == period] # for meetup probability at poi i on period "period"
        data_4 = data_2[data_2['period'] == period]  # for meetup probability at poi i on period "period" and day "day"

        
        ##for all data
        lat, lon = 0.00, 0.00
        for ii in data_1.index:
            lat, lon = data_1['latitude'][ii], data_1['longitude'][ii]
            break


        total_len = len(all_user)
        poi_total = all_user[all_user['latitude'] == lat]
        poi_total = poi_total[poi_total['longitude'] == lon]
        poi_popularity = len(poi_total)/total_len

        all_user = all_user[all_user['userId'] == user]
        time = []
        for ind in all_user.index:
            mon = all_user['month'][ind]
            dat = all_user['date'][ind]
            year = all_user['year'][ind]
            ho = all_user['hour'][ind] 
            t = year * 365 * 24 + mon * 30 * 24+ dat * 24 + ho
            time.append(t)
        all_user['time'] = time 
        all_user = all_user[all_user['time'] < tt]
        length2 = len(all_user)
        au_data1 = all_user[all_user['latitude'] == lat]
        au_data1 = all_user[all_user['longitude'] == lon]
        au_data2 = au_data1[au_data1['day'] == day]
        au_data3 = au_data1[au_data1['period'] == period]
        au_data4 = au_data2[au_data2['period'] == period]
        if length2 == 0:
            ad1,ad2,ad3,ad4 = 0.00,0.00,0.00,0.00
        else:
            ad1,ad2,ad3,ad4 = len(au_data1)/length2,len(au_data2)/length2,len(au_data3)/length2,len(au_data4)/length2

        if length == 0:
            d1,d2,d3,d4 = 0.00,0.00,0.00,0.00
        else:
            d1,d2,d3,d4 = len(data_1)/length,len(data_2)/length,len(data_3)/length,len(data_4)/length
        print("for poi ",i," d1 d2 d3 d4 ",d1," ",d2," ",d3," ",d4)
        print("for poi ",i," ad1 ad2 ad3 ad4 ",ad1," ",ad2," ",ad3," ",ad4)
        score[i] = d1 * 4200 + d2 * 2000 + d3 * 5000 + d4 * 3500 + ad1 * 3000 + ad2 * 2500 + ad3 * 4000 + ad4 * 3700 + poi_popularity *  2000
    return score
        