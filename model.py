
import pandas as pd
import datetime
from datetime import datetime as dt
import datetime as do

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from os.path import expanduser
import os.path
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import numpy


print ("Building model!")

#build a model
# Function to create model, required for KerasClassifier
def model_builder(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=9, kernel_initializer=init, activation='relu'))
	model.add(Dense(8, kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model   

def predict_outcome (date, home, away, city, country, model, scaler):

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

try:
    #get home path
    home = expanduser("~")

    #load data
    results = pd.read_csv(os.path.join(home,"results.csv"),parse_dates=['date'], infer_datetime_format=True)

    #cities from csv file
    cities = pd.read_csv(os.path.join(home,"cities.csv"))
    cities = cities.set_index(['city','country'])

    #countries from csv
    countries_geo = cities.groupby('country')['lat','lng'].mean()
    countries_pop = cities.groupby('country')['pop'].sum()
    countries = countries_geo.join(other=countries_pop, rsuffix='_pop')
    countries.to_csv(os.path.join(home,"countries.csv"))

    #join results to city to get fixture location geocodes
    results = results.join(other=cities, on=["city", "country"], rsuffix="_playedat", how="left")

    #join teams to country to get team geocodes
    results = results.join(other=countries, on=["home_team"], rsuffix="_home", how="left")
    results = results.join(other=countries, on=["away_team"], rsuffix="_away", how="left")

    #convert dates hard_date = datetime.date(2013, 5, 2)
    hard_date = datetime.date(1872,11, 1)
    results['date_delta'] = results['date'] - hard_date
    results['date_int'] = results['date_delta'].dt.days

    #fill in nas
    results['lat'] = results['lat'].fillna(results['lat'].mean())
    results['lng'] = results['lng'].fillna(results['lng'].mean())
    results['lat_home'] = results['lat_home'].fillna(results['lat_home'].mean())
    results['lng_home'] = results['lng_home'].fillna(results['lng_home'].mean())
    results['lat_away'] = results['lat_away'].fillna(results['lat_away'].mean())
    results['lng_away'] = results['lng_away'].fillna(results['lng_away'].mean())
    results['pop_home'] = results['pop_home'].fillna(results['pop_home'].mean())
    results['pop_away'] = results['pop_away'].fillna(results['pop_away'].mean())

    #cut data for modelling
    #results = results[['date','home_team','away_team','tournament','city','country','neutral','city_ascii','lat','lng','pop','iso2','iso3','province','lat_home','lng_home','lat_away','lng_away','home_score','away_score']]
    results = results[['date_int','lat','lng','lat_home','lng_home', 'pop_home','lat_away','lng_away', 'pop_away','home_score','away_score']]

    #shuffle
    results = results.sample(frac=1)

    #review
    results.to_csv(os.path.join(home,"prepared.csv"))

    #split to test and train
    train, test = train_test_split(results, test_size=0.2)

    #get numpy arrays
    x = results.values [:,0:9]
    y = results.values [:,9: 10]

    #scale https://stackoverflow.com/questions/48458635/getting-very-bad-prediction-with-kerasregressor
    sc_X = MinMaxScaler()
    x = sc_X.fit_transform(x)
    sc_Y = MinMaxScaler()
    y = sc_Y.fit_transform(y)

    #debug
    numpy.savetxt(os.path.join(home,"x.csv"), x, delimiter=",")
    numpy.savetxt(os.path.join(home,"y.csv"), y, delimiter=",")
    
    # Run model
    print ("Running regressor")
    estimator = KerasRegressor(build_fn=model_builder, epochs=200, batch_size=100, verbose=1)
    kfold = KFold(n_splits=10)
    print ("Scoring results")
    results = cross_val_score(estimator, x, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    # grid search epochs, batch size and optimizer
    # optimizers = ['rmsprop', 'adam']
    # init = ['glorot_uniform', 'normal', 'uniform']
    # epochs = [50, 100, 150]
    # batches = [5, 10, 20]
    # param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    # grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
    # grid_result = grid.fit(x, y)
    
    # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))

    #refit https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor
    estimator.fit(x,y)

    #save the model
    print ("Saving")
    estimator.model.save(os.path.join(home,"model.please"))
    joblib.dump(sc_Y, "scaler.please")

    #sanity check
    predict_outcome('2018-05-25', 'Brazil', 'Spain', 'London', 'United Kingdom', estimator,sc_Y)
    predict_outcome('2018-05-10', 'France', 'Spain', 'London', 'United Kingdom', estimator, sc_Y)
    predict_outcome('1995-05-09', 'Spain', 'New Zealand', 'Barcelona', 'Spain', estimator,sc_Y)
except Exception as e:
    print(e)