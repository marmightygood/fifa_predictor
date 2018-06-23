
##Updated to work better on windows

import datetime as do
import os.path
from datetime import datetime as dt
from os.path import expanduser

import numpy
import pandas as pd
from geopy.distance import geodesic
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.externals import joblib
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import datetime

import predictor
import prepare_data
import configparser

if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.sections()
    config.read(os.path.join(root_dir,'config.ini')) 
    print (config)
     #prepared_schedule = prepare_data.schedule(os.path.join(root_dir, "data", "fifa-world-cup-2018-RussianStandardTime.csv"))
    prepare_data.training(os.path.join(root_dir, "data", config["training"]["training_data"]))

def training(training_data):
    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    #load data
    results = pd.read_csv(training_data,parse_dates=['date'], infer_datetime_format=True)

    #cities from csv file
    cities = pd.read_csv(os.path.join(root_dir,"data","cities.csv"))
    cities = cities.set_index(['city','country'])

    #countries from csv
    countries_geo = cities.groupby('country')['lat','lng'].mean()
    countries_pop = cities.groupby('country')['pop'].sum()
    countries = countries_geo.join(other=countries_pop, rsuffix='_pop')
    countries.to_csv(os.path.join(root_dir,"data","countries.csv"))

    #join results to city to get fixture location geocodes
    results = results.join(other=cities, on=["city", "country"], rsuffix="_playedat", how="left")

    #join teams to country to get team geocodes
    results = results.join(other=countries, on=["home_team"], rsuffix="_home", how="left")
    results = results.join(other=countries, on=["away_team"], rsuffix="_away", how="left")

    #convert dates hard_date = datetime.date(2013, 5, 2)
    hard_date = do.datetime(1800,11, 1)  
    results['date_delta'] = results['date'] - hard_date
    results['date_int'] = results['date_delta'].dt.days

    en = LabelEncoder()
    results['tournament'] = en.fit_transform(results['tournament'])
    joblib.dump(en,os.path.join(root_dir,"output", "label_encoder.please"))

    #fill in nas
    results['lat'] = results['lat'].fillna(results['lat'].min())
    results['lng'] = results['lng'].fillna(results['lng'].min())
    results['lat_home'] = results['lat_home'].fillna(results['lat_home'].min())
    results['lng_home'] = results['lng_home'].fillna(results['lng_home'].min())
    results['lat_away'] = results['lat_away'].fillna(results['lat_away'].min())
    results['lng_away'] = results['lng_away'].fillna(results['lng_away'].min())
    results['pop_home'] = results['pop_home'].fillna(results['pop_home'].min())
    results['pop_away'] = results['pop_away'].fillna(results['pop_away'].min())

    results['geodesic'] = results.apply(lambda x: geodesic((x['lat'],x['lng']), (x['lat_away'],   x['lng_away'])).kilometers, axis=1)
    results['geodesic_home'] = results.apply(lambda x: geodesic((x['lat'],x['lng']), (x['lat_home'],   x['lng_home'])).kilometers, axis=1)
 
    results.to_csv(os.path.join(root_dir,"output","lats_longs.csv"))

    #cut data for modelling
    results = results[['date_int','tournament','geodesic', 'geodesic_home','pop_home', 'pop_away','home_score','away_score']]

    #shuffle
    results = results.sample(frac=1)

    #review
    results.to_csv(os.path.join(root_dir,"output","prepared.csv"))

    #get numpy arrays
    x = results.values [:,0:6]
    y = results.values [:,6: 8]

    numpy.savetxt(os.path.join(root_dir,"output","x_prescaled.csv"), x, delimiter=",")
    numpy.savetxt(os.path.join(root_dir,"output","y_prescaled.csv"), y, delimiter=",")    

    #scale https://stackoverflow.com/questions/48458635/getting-very-bad-prediction-with-kerasregressor
    sc_X = StandardScaler()
    x = sc_X.fit_transform(x)
    sc_Y = StandardScaler()
    y = sc_Y.fit_transform(y)

    joblib.dump(sc_Y,os.path.join(root_dir,"output", "y_scaler.please"))
    joblib.dump(sc_X,os.path.join(root_dir,"output", "x_scaler.please"))

    #debug
    numpy.savetxt(os.path.join(root_dir,"output","x_scaled.csv"), x, delimiter=",")
    numpy.savetxt(os.path.join(root_dir,"output","y_scaled.csv"), y, delimiter=",")
    return x, y, sc_X, sc_Y

def schedule (schedule_data):
    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    #create output dir
    output = os.path.join(root_dir,"output")
    if not os.path.exists(output):
        os.makedirs(output)

    #load data
    schedule = pd.read_csv(os.path.join(root_dir, "data",schedule_data),parse_dates=['Date'], infer_datetime_format=True) 
    schedule.columns=['round','date','city','home_team','away_team','group','result']
    schedule["country"] = 'Russia'
    schedule["city"] = "Moscow"

    #cities from csv file
    cities = pd.read_csv(os.path.join(root_dir,"data","cities.csv"))
    cities = cities.set_index(['city','country'])

    #countries from csv
    countries_geo = cities.groupby('country')['lat','lng'].mean()
    countries_pop = cities.groupby('country')['pop'].sum()
    countries = countries_geo.join(other=countries_pop, rsuffix='_pop')
    countries.to_csv(os.path.join(root_dir,"data","countries.csv"))

    #join schedule to city to get fixture location geocodes
    schedule = schedule.join(other=cities, on=["city", "country"], rsuffix="_playedat", how="left")

    #join teams to country to get team geocodes
    schedule = schedule.join(other=countries, on=["home_team"], rsuffix="_home", how="left")
    schedule = schedule.join(other=countries, on=["away_team"], rsuffix="_away", how="left")

    #convert dates hard_date = datetime.date(2013, 5, 2)
    hard_date = do.datetime(1800,11, 1)  
    schedule['date_delta'] = schedule['date'] - hard_date
    schedule['date_int'] = schedule['date_delta'].dt.days

    #fill in nas (lat/lng are for Moscow)
    schedule['lat'] = schedule['lat'].fillna(55.7558)
    schedule['lng'] = schedule['lng'].fillna(31.6173)
    schedule['lat_home'] = schedule['lat_home'].fillna(schedule['lat_home'].min())
    schedule['lng_home'] = schedule['lng_home'].fillna(schedule['lng_home'].min())
    schedule['lat_away'] = schedule['lat_away'].fillna(schedule['lat_away'].min())
    schedule['lng_away'] = schedule['lng_away'].fillna(schedule['lng_away'].min())
    schedule['pop_home'] = schedule['pop_home'].fillna(schedule['pop_home'].min())
    schedule['pop_away'] = schedule['pop_away'].fillna(schedule['pop_away'].min())

    schedule.to_csv(os.path.join(root_dir,"output","pre_geocoding.csv"))

    schedule['geodesic'] = schedule.apply(lambda x: geodesic((x['lat'],x['lng']), (x['lat_away'],   x['lng_away'])).kilometers, axis=1)
    schedule['geodesic_home'] = schedule.apply(lambda x: geodesic((x['lat'],x['lng']), (x['lat_home'],   x['lng_home'])).kilometers, axis=1)
    schedule.to_csv(os.path.join(root_dir,"output","lats_longs.csv"))

    en = joblib.load(os.path.join(root_dir,"output", "label_encoder.please"))
    tournament = "FIFA World Cup"
    schedule['tournament'] = tournament
    schedule['tournament'] = en.transform(schedule['tournament'])

    #cut data for modelling
    schedule = schedule[['date_int','tournament','geodesic','geodesic_home', 'pop_home', 'pop_away','result']]

    #review
    schedule.to_csv(os.path.join(root_dir,"output","schedule_prepared.csv"))

    #get numpy arrays
    x = schedule.values [:,0:6]


    #scale https://stackoverflow.com/questions/48458635/getting-very-bad-prediction-with-kerasregressor
    sc_X = joblib.load(os.path.join(root_dir,"output", "x_scaler.please")) 
    # sc_X = joblib.load(os.path.join(root_dir,"output", "x_scaler.please"))   
    numpy.savetxt(os.path.join(root_dir,"output","schedule_x.csv"), x, delimiter=",")

    print ("Fed to x-scaler")
    print (x)

    x = sc_X.transform(x)
    # y = sc_Y.fit_transform(y)

    #debug
    numpy.savetxt(os.path.join(root_dir,"output","schedule_x.csv"), x, delimiter=",")
    # numpy.savetxt(os.path.join(root_dir,"output","schedule_y.csv"), y, delimiter=",")
    return x