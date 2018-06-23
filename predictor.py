
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
from sklearn.preprocessing import StandardScaler
import prepare_data
import configparser


def build_by_loading():
    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__)) 
    model = load_model(os.path.join(root_dir,"output", "model.please"))
    return model 


def predict_single_outcome (date, home_team, away_team, city, country, epochs, batch_size):
    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    print ("Running prediction!")

    en = joblib.load(os.path.join(root_dir,"output", "label_encoder.please"))
    tournament = ["FIFA World Cup"]
    tournament = en.transform(tournament)[0]

    estimator = KerasRegressor(build_fn=build_by_loading, nb_epoch=epochs, batch_size=batch_size, verbose=1)
    estimator.model = load_model(os.path.join(root_dir,"output", "model.please"))

    #cities from csv file
    cities = pd.read_csv(os.path.join(root_dir,"data", "cities.csv"))
    cities = cities.set_index(['city','country'])

    #countries from csv
    countries =  pd.read_csv(os.path.join(root_dir,"data","countries.csv"))

    scaler = joblib.load(os.path.join(root_dir,"output", "y_scaler.please")) 
    sc_X = joblib.load(os.path.join(root_dir,"output", "x_scaler.please")) 

    #results = results[['date_int','lat','lng','lat_home','lng_home','lat_away','lng_away','home_score','away_score', 'pop_home', 'pop_away']]
    dt_date = dt.strptime(date, '%Y-%m-%d')
    hard_date = dt.strptime('1800-11-01', '%Y-%m-%d')
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
        home_local = geodesic({lat, lng},{lat_home, lng_home}).kilometers
    except:
        travel = 0
        home_local = 0
        pass


    data_frame = pd.DataFrame(columns = ['date_int', 'tournament', 'travel', 'home_local', 'pop_home', 'pop_away'])
    data_frame.loc[0] = [date_int, tournament, travel, home_local, pop_home, pop_away]
    print ("Fed to x-scaler")
    print (data_frame)

    x = sc_X.transform(data_frame)

    print ("Fed to estimator")
    print (x)

    prediction = estimator.predict(x)

    prediction = prediction.reshape(1,-1)
    print ("{0} vs {1}".format(home_team,away_team))
    #print(prediction)
    print(scaler.inverse_transform(prediction))

def predict_list(x):

    root_dir = os.path.dirname(os.path.realpath(__file__))
    epochs = int(config["hyperparameters"]["epochs"])
    batch_size = int(config["hyperparameters"]["batch_size"])
    estimator = KerasRegressor(build_fn=build_by_loading, nb_epoch=epochs, batch_size=batch_size, verbose=1)
    estimator.model = load_model(os.path.join(root_dir,"output", "model.please"))

    print ("Fed to estimator")
    print (x)
    prediction = estimator.predict(x)
    scaler = joblib.load(os.path.join(root_dir,"output", "y_scaler.please"))          
    #return prediction
    return scaler.inverse_transform(prediction)

if __name__ == "__main__":

    root_dir = os.path.dirname(os.path.realpath(__file__))
    config = configparser.ConfigParser()
    config.sections()
    config.read(os.path.join(root_dir,'config.ini')) 


    import time
    timestr = time.strftime("%Y%m%d_%H%M%S")


    ##schedule
    print("Predictions of world cup schedule")
    prepared_schedule = prepare_data.schedule('fifa-world-cup-2018-RussianStandardTime.csv')
    predictions = predict_list (prepared_schedule)

    predictions = pd.DataFrame(predictions, columns=['home_score','away_score'])
    schedule = pd.read_csv(os.path.join(root_dir,"data", 'fifa-world-cup-2018-RussianStandardTime.csv'))

    predictions = pd.concat((schedule,predictions), axis=1)
    predictions.to_csv(os.path.join(root_dir,"output", 'predictions_schedule_'+timestr+'.csv'))

    ##original data
    # print("Predictions on original data")
    # prepared_results = pd.read_csv(os.path.join(root_dir,"output", 'x_scaled.csv'))
    # predictions = predict_list (prepared_results)

    #predictions = pd.DataFrame(predictions, columns=['home_score','away_score'])
    #fullresults = pd.read_csv(os.path.join(root_dir,"data", "results.csv"))

    #predictions = pd.concat((fullresults,predictions), axis=1)
    #predictions.to_csv(os.path.join(root_dir,"output", 'predictions_results.csv'))

    print("Predicting singletons")
    predict_single_outcome('2018-06-16', 'France', 'Australia', 'Moscow', 'Russia', config["hyperparameters"]["epochs"], config["hyperparameters"]["batch_size"])
#4,1,16/06/2018 13:00,Kazan Arena,France,Australia,Group C,,1.6349037,5.3673425
