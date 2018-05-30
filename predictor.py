
import pandas as pd
import numpy

from datetime import datetime as dt
import datetime as do

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense

from os.path import expanduser
import os.path

print ("Running predictions!")

#get home path
home = expanduser("~")

def build_by_loading():
    model = load_model(os.path.join(home,"model.please"))
    return model 

estimator = KerasRegressor(build_fn=build_by_loading, nb_epoch=1, batch_size=1, verbose=1)
estimator.model = load_model(os.path.join(home,"model.please"))

#cities from csv file
cities = pd.read_csv(os.path.join(home,"cities.csv"))
cities = cities.set_index(['city','country'])

#countries from csv
countries = cities.groupby('country').mean()

def predict_outcome (date, home, away, city, country, model):

    scaler = joblib.load("scaler.please") 

    #results = results[['date_int','lat','lng','lat_home','lng_home','lat_away','lng_away','home_score','away_score', 'pop_home', 'pop_away']]
    dt_date = dt.strptime(date, '%Y-%m-%d').date()
    hard_date = do.date(1000,11, 1)
    date_delta = dt_date - hard_date
    date_int = date_delta.days

    #lookup lat/long of fixture
    row = cities.loc[city, country]
    lat = row.lat[0]
    lng = row.lng[0]

    #lookup lat/long of home
    home = countries.loc[home]
    lat_home = home.lat
    lng_home = home.lng
    pop_home = home[2]

    #lookup lat/long of away
    away = countries.loc[away]
    lat_away = away.lat
    lng_away = away.lng
    pop_away = away[2]

    data_frame = pd.DataFrame(data = [{date_int, lat, lng, lat_home, lng_home, lat_away, lng_away, pop_home, pop_away}])
    
    sc_X = MinMaxScaler()
    x = sc_X.fit_transform(data_frame)
    
    prediction = model.predict(x)
    prediction = prediction.reshape(1,-1)    
    print(scaler.inverse_transform(prediction))

predict_outcome('2018-05-25', 'Brazil', 'Spain', 'London', 'United Kingdom', estimator)
predict_outcome('2018-05-10', 'France', 'Spain', 'London', 'United Kingdom', estimator)
predict_outcome('2018-05-09', 'Spain', 'France', 'Barcelona', 'Spain', estimator)
