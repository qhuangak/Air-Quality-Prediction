
import pandas as pd
import numpy as np
from datetime import datetime

#reading the dataSet
print('start import data')

#import grid weather dataset
grid_weather1 = pd.read_csv("data/gridWeather_201701-201803.csv",usecols=[0,3,4,5,6,7,8])
grid_weather1.rename(columns={'stationName':'grid_station'}, inplace=True) 
grid_weather1.rename(columns={'wind_speed/kph':'wind_speed'}, inplace=True)

#transfer time format
grid_weather1['utc_time']=grid_weather1['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
grid_weather1['utc_time']=pd.to_datetime(grid_weather1['utc_time'])
grid_weather2 = pd.read_csv("data/gridWeather_201804.csv",usecols=[1,2,4,5,6,7,8])
grid_weather2.rename(columns={'station_id':'grid_station'}, inplace=True) 
grid_weather2.rename(columns={'time':'utc_time'}, inplace=True) 
grid_weather2['utc_time']=grid_weather2['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
grid_weather=grid_weather1.append(grid_weather2,ignore_index=True)
test_grid_weather = pd.read_csv("data/gridWeather_20180501-20180502.csv")
test_grid_weather.rename(columns={'station_id':'grid_station'}, inplace=True)

#import air quality dataset
air_quality1 = pd.read_csv("data/airQuality_201701-201801.csv",usecols=[0,1,2,3,6])
air_quality2 = pd.read_csv("data/airQuality_201802-201803.csv",usecols=[0,1,2,3,6])

#change time type from string to datetime
air_quality1['utc_time']=air_quality1['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
air_quality2['utc_time']=air_quality2['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
air_quality3 = pd.read_csv("data/aiqQuality_201804.csv",usecols=[1,2,3,4,7])
air_quality3.rename(columns={'time':'utc_time'}, inplace=True)
air_quality3.rename(columns={'PM25_Concentration': 'PM2.5'}, inplace=True)
air_quality3.rename(columns={'PM10_Concentration': 'PM10'}, inplace=True) 
air_quality3.rename(columns={'O3_Concentration': 'O3'}, inplace=True) 
air_quality3.rename(columns={'station_id': 'stationId'}, inplace=True)
air_quality3['utc_time']=air_quality3['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

#combine all air quality data
air_quality=air_quality1.append(air_quality2,ignore_index=True).append(air_quality3,ignore_index=True)
air_quality.rename(columns={'stationId': 'station_id'}, inplace=True)
 
#import grid station dataset
grid_station = pd.read_csv("data/Beijing_grid_weather_station.csv", names=['station_id', 'latitude','longitude'])

#import air quality station dataset
station_Beijing={
        'station_id':['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq','nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq','yungang_aq','gucheng_aq'],
        'longitude':[116.417,116.407,116.339,116.352,116.397,116.461,116.287,116.174,116.207,116.279,116.146,116.184],
        'latitude':[39.929,39.886,39.929,39.878,39.982,39.937,39.987,40.09,40.002,39.863,39.824,39.914]  
        }
station_Beijing=pd.DataFrame(station_Beijing)
station_Beijing['location']='urban'

station_suburban={
        'station_id':['fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq','shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq','yanqin_aq'],
        'longitude':[116.136,116.404,116.506,116.663,116.655,116.23,116.106,117.1,116.628,116.832,115.972],
        'latitude':[39.742,39.718,39.795,39.886,40.127,40.217,39.937,40.143,40.328,40.37,40.453]       
        }
station_suburban=pd.DataFrame(station_suburban)
station_suburban['location']='suburban'

station_others={
        'station_id':['dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq','yongledian_aq','yufa_aq','liulihe_aq'],
        'longitude':[116.22,115.988,116.911,117.12,116.783,116.3,116],
        'latitude':[40.292,40.365,40.499,40.1,39.712,39.52,39.58]       
        }
station_others=pd.DataFrame(station_others)
station_others['location']='others'

station_traffic={
        'station_id':['qianmen_aq','yongdingmennei_a','xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq'],
        'longitude':[116.395,116.394,116.349,116.368,116.483],
        'latitude':[39.899,39.876,39.954,39.856,39.939]  
        }
station_traffic=pd.DataFrame(station_traffic)
station_traffic['location']='traffic'

station_aq=station_Beijing.append(station_suburban,ignore_index=True).append(station_others,ignore_index=True).append(station_traffic,ignore_index=True)
print('end import data')
print('start data preprocessing part')
#the function of fill missing value
def filldata(x,aq_dataframe):
    print('start fill in'+x)
    consequent_nan_cnt = 0
    for i in range(1,len(aq_dataframe)):
        if np.isnan(aq_dataframe.iloc[i][x]):
            consequent_nan_cnt += 1
        else:
            start_index = i - consequent_nan_cnt - 1
            start = aq_dataframe.iloc[start_index][x]
            end_index = i
            end = aq_dataframe.iloc[end_index][x]
            interval = (end - start) / (consequent_nan_cnt + 1)
            for j in range(consequent_nan_cnt):
                aq_dataframe.loc[start_index + j + 1,x] = start + interval * (j + 1)
            consequent_nan_cnt = 0
    print('end fill in'+x)


#fill missing data in air quality table
filldata('PM2.5',air_quality)
filldata('PM10',air_quality)
filldata('O3',air_quality)

# the function of find nearest grid station for every aq_station
def findNearestGStation(x,y):
    """from air station coordinate, get index of gird, x is longitude, y is latitude

    """
    x_mod=(x-115)%0.1        #get mod of coordinate
    y_mod=(y-39)%0.1

    if x_mod<=0.05:          #if mod smaller than 0.05, x belongs to the left grid
        x1=int((x-115)/0.1)
    else:
        x1=int((x-115)/0.1)+1
        
    if y_mod<=0.05:          #if mod smaller than 0.05, y belongs to the down grid
        y1=int((y-39)/0.1)
    else:
        y1=int((y-39)/0.1)+1
        
    index=x1*21+y1    #get index of grid station by coordinate
    
    return grid_station.loc[index,'station_id']

#find corresponding gird station for each air quality station
station_aq['grid_station']=None                                                
station_aq['grid_station']= station_aq.apply(lambda x: findNearestGStation(x.longitude,x.latitude),axis=1)

#delete repetitive data in grid weather data
grid_weather =  grid_weather.drop_duplicates(subset=['grid_station', 'utc_time'], keep='first')

#combine aq_station information with its air quality data
new_air_quality = pd.merge(air_quality, station_aq, how='left', on=['station_id'])

#combine air quality data with each weather data
new_air_quality1 = pd.merge(new_air_quality, grid_weather, how='left', on=['grid_station', 'utc_time'])

print('end data preprocessing part')

print('start featrue engineering part')
#copy dataframe seperatly
aq_PM25=new_air_quality1
aq_PM10=new_air_quality1
aq_O3=new_air_quality1

#extract the 48 hours pollution data as 48 featrues
def get_48_aq(aq_dataframe,number,pollution): 
    
    for i in range(number):
        aq_dataframe[i+1]=aq_dataframe[pollution]
        aq_dataframe[i+1]=aq_dataframe[i+1].shift(i+48)
      
#delete data in extra time             
get_48_aq(aq_PM25,48,'PM2.5')
aq_PM25['utc_time'] = pd.to_datetime(aq_PM25['utc_time'])
aq_PM25 = aq_PM25[aq_PM25['utc_time'] >= datetime(2017,1,5,14)]

get_48_aq(aq_PM10,48,'PM10')
aq_PM10['utc_time'] = pd.to_datetime(aq_PM10['utc_time'])
aq_PM10 = aq_PM10[aq_PM10['utc_time'] >= datetime(2017,1,5,14)]

get_48_aq(aq_O3,48,'O3')
aq_O3['utc_time'] = pd.to_datetime(aq_O3['utc_time'])
aq_O3 = aq_O3[aq_O3['utc_time'] >= datetime(2017,1,5,14)]

#drop extra attribute
aq_PM25.drop(['utc_time','grid_station','longitude','latitude','station_id'],axis=1,inplace=True)
aq_PM10.drop(['utc_time','grid_station','longitude','latitude','station_id'],axis=1,inplace=True)
aq_O3.drop(['utc_time','grid_station','longitude','latitude','station_id'],axis=1,inplace=True)

#one hot location label
aq_PM25=pd.get_dummies(aq_PM25)
aq_PM10=pd.get_dummies(aq_PM10)
aq_O3=pd.get_dummies(aq_O3)

print('end featrue engineering part')

print('start model fitting part')
#split dataset into train and test
from sklearn.model_selection import train_test_split
train_PM25, test_PM25 = train_test_split(aq_PM25, test_size=0.2, random_state=0)
train_PM10, test_PM10 = train_test_split(aq_PM10, test_size=0.2, random_state=0)
train_O3, test_O3 = train_test_split(aq_O3, test_size=0.2, random_state=0)

x_train_PM25=train_PM25.drop(['PM2.5','PM10','O3'],axis=1)
y_PM25_train =train_PM25['PM2.5']
x_train_PM10=train_PM10.drop(['PM2.5','PM10','O3'],axis=1)
y_PM10_train =train_PM10['PM10']
x_train_O3=train_O3.drop(['PM2.5','PM10','O3'],axis=1)
y_O3_train =train_O3['O3']

x_test_PM25=test_PM25.drop(['PM2.5','PM10','O3'],axis=1)
y_PM25_test=test_PM25['PM2.5']
x_test_PM10=test_PM10.drop(['PM2.5','PM10','O3'],axis=1)
y_PM10_test=test_PM10['PM10']
x_test_O3=test_O3.drop(['PM2.5','PM10','O3'],axis=1)
y_O3_test=test_O3['O3']

#the score function
def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

#build PM25 predict model
from lightgbm.sklearn import LGBMRegressor
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model_PM25 = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=600,min_child_weight=9, subsample=0.7, colsample_bytree=0.5,learning_rate=0.01, reg_lambda=0.4)
model_PM25.fit(x_train_PM25, y_PM25_train)
y_PM25_pred = model_PM25.predict(x_test_PM25)
loss_PM25 = smape(y_PM25_test, y_PM25_pred)

#build PM10 predict model
model_PM10 = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=600,min_child_weight=10, subsample=0.7, colsample_bytree=0.6,learning_rate=0.01, reg_lambda=0.4)
model_PM10.fit(x_train_PM10, y_PM10_train)
y_PM10_pred = model_PM10.predict(x_test_PM10)
loss_PM10 = smape(y_PM10_test, y_PM10_pred)
   
#build O3 predict model
model_O3 = LGBMRegressor(num_leaves=40,max_depth=7,n_estimators=600,min_child_weight=10, subsample=0.7, colsample_bytree=0.7,learning_rate=0.01, reg_lambda=0.4)
model_O3.fit(x_train_O3, y_O3_train)
y_O3_pred = model_O3.predict(x_test_O3)
loss_O3 = smape(y_O3_test, y_O3_pred)

print('end model fitting part')

print('start testing data preprocessing part')
#merge test station with test grid data
test_station_aq=pd.merge(test_grid_weather, station_aq, how='right', on=['grid_station'])

#choose the time between 4.27 and 4.30
new_air_quality_2 = new_air_quality[(new_air_quality['utc_time'] <= datetime(2018,4,30,23))&(new_air_quality['utc_time'] > datetime(2018,4,26,23))]

#new a dataframe to add the lost time data
new={
     'utc_time':[],
     'station_id':[],    
     }
new=pd.DataFrame(new)

#loop to add the time
for station in range(35):
    for i in range(96):
        day=27+int(i/24)
        hour=i%24
        int(hour)
        new.loc[station*96+i,'utc_time']= datetime(2018,4,day,hour)
        new.loc[station*96+i,'station_id']=station_aq.loc[station,'station_id']

#transfer the time format
new['utc_time']=new['utc_time'].apply(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))
new['utc_time']=new['utc_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

#merge air_quality with full air quality
new_air_quality_2=pd.merge(new_air_quality_2, new, how='right',sort=True, on=['station_id','utc_time'])

#call filldata method
filldata('PM2.5',new_air_quality_2)
filldata('PM10',new_air_quality_2)
filldata('O3',new_air_quality_2)

#preprocessing test air quality data
test_station_aq.rename(columns={'time':'utc_time'}, inplace=True) 
test_station_aq.drop(['id','weather','grid_station','longitude','latitude'],axis=1,inplace=True)

#combine data between 2018.4.27 to 2018.5.2
new_air_quality_2=new_air_quality_2.append(test_station_aq,ignore_index=True)
new_air_quality_2.drop(['longitude','latitude','grid_station'],axis=1,inplace=True)

#create a new table to store the dataset after sorting
new_air_quality_3={
     'utc_time':[],
     'station_id':[],
     'PM2.5':[],
     'PM10':[],
     'O3':[],
     'humidity':[],
     'pressure':[],
     'temperature':[],
     'wind_direction':[],
     'wind_speed':[],
     'location':[]
        }
new_air_quality_3=pd.DataFrame(new_air_quality_3)

#sort the dataset as the sample submission
index=0
for i in range(len(station_aq)):
    for m in range(len(new_air_quality_2)):
        if new_air_quality_2.loc[m,'station_id']==station_aq.loc[i,'station_id']:
            new_air_quality_3.loc[index]=new_air_quality_2.loc[m]
            index+=1
            
print('end testing data preprocessing part')   
print('start testing data feature engineering and predict part') 
#copy to three dataset 
test_PM25=new_air_quality_3
test_PM10=new_air_quality_3
test_O3=new_air_quality_3
    
#delete data before 2018.5.1 0:00             
get_48_aq(test_PM25,48,'PM2.5')
test_PM25['utc_time'] = pd.to_datetime(test_PM25['utc_time'])
test_PM25 = test_PM25[test_PM25['utc_time'] >= datetime(2018,5,1,0)]
test_PM25.drop(['station_id','utc_time','PM2.5','PM10','O3'],axis=1,inplace=True)

#get dummies of location
test_PM25=pd.get_dummies(test_PM25)

#predict the data
PM25_pred=model_PM25.predict(test_PM25)
             
get_48_aq(test_PM10,48,'PM2.5')
test_PM10['utc_time'] = pd.to_datetime(test_PM10['utc_time'])
test_PM10 = test_PM10[test_PM10['utc_time'] >= datetime(2018,5,1,0)]
test_PM10.drop(['station_id','utc_time','PM2.5','PM10','O3'],axis=1,inplace=True)
test_PM10=pd.get_dummies(test_PM10)
PM10_pred=model_PM10.predict(test_PM10)
            
get_48_aq(test_O3,48,'PM2.5')
test_O3['utc_time'] = pd.to_datetime(test_O3['utc_time'])
test_O3 = test_O3[test_O3['utc_time'] >= datetime(2018,5,1,0)]
test_O3.drop(['station_id','utc_time','PM2.5','PM10','O3'],axis=1,inplace=True)
test_O3=pd.get_dummies(test_O3)
O3_pred=model_O3.predict(test_O3)

print('end testing data feature engineering and predict part')

#define a change to positive function
def pos(test_pred):
    for i in range(len(test_pred)):
        if test_pred[i]<0:
            test_pred[i]=test_pred[i]*-1

#change negative to positive
pos(PM25_pred)
pos(PM10_pred)
pos(O3_pred)

#combine all the predict data to a dataframe
final_prediction = pd.DataFrame(columns = ['PM2.5', 'PM10', 'O3']) 
final_prediction['PM2.5']=PM25_pred
final_prediction['PM10']=PM10_pred
final_prediction['O3']=O3_pred

#output the result  
final_prediction.to_csv('submission.csv',columns=['PM2.5','PM10','O3'],header=True, index=True,index_label='test_id')













