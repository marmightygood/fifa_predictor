
import datetime as do
import os.path
from datetime import datetime as dt
from os.path import expanduser

import numpy
import pandas as pd
from geopy.distance import geodesic
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler


def build_by_loading():
    #get home path
    home = expanduser("~")    
    model = load_model(os.path.join(home,"model.please"))
    return model 


def predict_outcome (date, home_team, away_team, city, country):
    #get home path
    home = expanduser("~")

    print ("Running prediction!")

    estimator = KerasRegressor(build_fn=build_by_loading, nb_epoch=5, batch_size=100, verbose=1)
    estimator.model = load_model(os.path.join(home,"model.please"))

    #cities from csv file
    cities = pd.read_csv(os.path.join(home,"cities.csv"))
    cities = cities.set_index(['city','country'])

    #countries from csv
    countries = cities.groupby('country').mean()

    scaler = joblib.load("scaler.please") 
    sc_X = joblib.load("scalerx.please") 

    #results = results[['date_int','lat','lng','lat_home','lng_home','lat_away','lng_away','home_score','away_score', 'pop_home', 'pop_away']]
    dt_date = dt.strptime(date, '%Y-%m-%d').date()
    hard_date = do.date(1800,11, 1)
    date_delta = dt_date - hard_date
    date_int = date_delta.days

    #lookup lat/long of fixture
    row = cities.loc[city, country]
    lat = row.lat[0]
    lng = row.lng[0]

    #lookup lat/long of home
    home = countries.loc[home_team]
    lat_home = home.lat
    lng_home = home.lng
    pop_home = home[2]

    #lookup lat/long of away
    away = countries.loc[away_team]
    lat_away = away.lat
    lng_away = away.lng
    pop_away = away[2]
    try:
        travel = geodesic({lat, lng},{lat_away, lng_away}).kilometers
    except:
        travel = 0
        pass

    data_frame = pd.DataFrame(data = [{date_int,travel, pop_home, pop_away}])
    
    x = sc_X.transform(data_frame)
    
    prediction = estimator.predict(x)
    prediction = prediction.reshape(1,-1)
    print ("{0} vs {1}".format(home_team,away_team))
    print(scaler.inverse_transform(prediction))
if __name__ == "__main__":
    predict_outcome('2018-05-25', 'Brazil', 'Spain', 'London', 'United Kingdom')
    predict_outcome('2018-05-10', 'France', 'Spain', 'Wellington', 'New Zealand')
    predict_outcome('2018-05-10', 'France', 'Spain', 'Paris', 'France')
    predict_outcome('2018-05-09', 'Spain', 'France', 'Barcelona', 'Spain')
    predict_outcome('2018-05-09', 'New Zealand', 'Australia', 'Sydney', 'Australia')
