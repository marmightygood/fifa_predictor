
import pandas as pd
import datetime
from datetime import datetime as dt
import datetime as do

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from os.path import expanduser
import os.path
from keras.wrappers.scikit_learn import KerasRegressor
from geopy.distance import geodesic

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy
import predictor
import prepare_data
import model_builder

if __name__ == "__main__":
    print ("Grid searching!")

    #get home path
    root_dir = os.path.dirname(os.path.realpath(__file__))

    x, y, sc_X, sc_Y = prepare_data.training(os.path.join(root_dir,"data", "results.csv"))

    # create model
    model = KerasRegressor(build_fn=model_builder.create_model, verbose=1, feature_count=len(x[0]), output_count= len(y[0]))

    # grid search epochs, batch size and optimizer
    init = ['normal']
    epochs = [10]
    batches = [20]
    marginal_input_neurons = [4,8]
    hidden_layer_counts = [10]
    param_grid = dict(epochs=epochs, batch_size=batches,hidden_layer_count=hidden_layer_counts, init=init, marginal_input_neurons = marginal_input_neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(x, y)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
