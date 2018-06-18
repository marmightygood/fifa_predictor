
import datetime as do
import os.path
from datetime import datetime as dt
from os.path import expanduser

import numpy
import pandas as pd
from geopy.distance import geodesic
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import SimpleRNNCell
from keras.layers import Activation
from keras.layers import Flatten
from keras.optimizers import SGD

import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.externals import joblib
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

import predictor
import prepare_data


#build a model
# Function to create model, required for KerasClassifier
def create_model(init='glorot_uniform', hidden_layer_count = 2, feature_count = 6, output_count= 2):
    # create model
    model = Sequential()

    #optimizer
    optimizer = SGD(lr=0.01)

    #input neurons = number of features plus 2
    input_neurons = feature_count + 2

    #input layer
    model.add(Dense(input_neurons, input_dim=feature_count, kernel_initializer=init, activation='relu'))

    #add hidden layers
    hidden_layers_added = 0
    neurons = input_neurons + 2
    while hidden_layers_added < hidden_layer_count:
        model.add(Dense(neurons, kernel_initializer=init))#, activation='relu'))
        model.add(Dropout(0.1, noise_shape=None, seed=None))      
        hidden_layers_added += 1
        neurons += 2

    #output layer
    model.add(Dense(output_count, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

if __name__ == "__main__":

    print ("Building model!")

    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    # x, y, sc_X, sc_Y = prepare_data.training(os.path.join(root_dir, "data", "fullresults.csv")
    x = numpy.loadtxt(os.path.join(root_dir,"output","x_scaled.csv"),  delimiter=",")
    y = numpy.loadtxt(os.path.join(root_dir,"output","y_scaled.csv"),  delimiter=",")    

    # Run model
    print ("Running regressor")
    estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=10, verbose=1, hidden_layer_count=2, feature_count=len(x[0]), output_count= len(y[0]))
    kfold = KFold(n_splits=10)
    print ("Scoring results")
    results = cross_val_score(estimator, x, y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    history = estimator.fit(x,y)

    #save the model
    print ("Saving")
    estimator.model.save(os.path.join(root_dir,"output","model.please"))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    print ("plt.Show()")    
    plt.savefig(os.path.join(root_dir,"output","accuracy.jpg"))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    print ("plt.Show()")
    plt.savefig(os.path.join(root_dir,"output","loss.jpg"))