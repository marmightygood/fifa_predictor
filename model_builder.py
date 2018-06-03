
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
from sklearn.preprocessing import MinMaxScaler

import predictor
import prepare_data


#build a model
# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=4, kernel_initializer=init, activation='relu'))
	model.add(Dense(168, kernel_initializer=init, activation='relu'))
	model.add(Dense(2, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

if __name__ == "__main__":

    print ("Building model!")

    #get home path
    home = expanduser("~")

    x, y, sc_X, sc_Y = prepare_data.prepare("results.csv")

    # Run model
    print ("Running regressor")
    estimator = KerasRegressor(build_fn=create_model, epochs=200, batch_size=100, verbose=1)
    kfold = KFold(n_splits=10)
    print ("Scoring results")
    results = cross_val_score(estimator, x, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    estimator.fit(x,y)

    #save the model
    print ("Saving")
    estimator.model.save(os.path.join(home,"model.please"))

    #sanity check
    predictor.predict_outcome('2018-05-25', 'Brazil', 'Spain', 'London', 'United Kingdom')
    predictor.predict_outcome('2018-05-10', 'France', 'Spain', 'London', 'United Kingdom')
    predictor.predict_outcome('1995-05-09', 'Spain', 'New Zealand', 'Barcelona', 'Spain')   
