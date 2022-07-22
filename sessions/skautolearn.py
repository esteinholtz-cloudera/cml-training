# Copyright 2020 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # scikit-learn Example

# This example demonstrates a simple regression modeling
# task using the using the
# [`LinearRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# class in the 
# [`sklearn.linear_model`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# module. The goal is to predict the arrival delay of
# flights based on the departure delay.

# The code in this file must be run in a Python 3 session
# and requires the scikit-learn library. If this code 
# fails to run in a Python 3 session, install
# scikit-learn by running `!pip3 install -U scikit-learn`


# ## Import Modules

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from joblib import dump, load


# ## Prepare Data

# Read the flights data from the CSV file into a pandas
# DataFrame
flights_pd = pd.read_csv('data/flights.csv')

# Drop unneeded columns then drop rows with missing
# values or outliers
flights_clean_pd = flights_pd \
  .dropna() \
  .loc[flights_pd.dep_delay < 400, :]

# Separate the features (x) and targets (y)
features = flights_clean_pd.filter(['year', 'month', 'day', 'sched_dep_time', 'hour','minute'])
targets = flights_clean_pd.filter(['dep_delay'])

# Split the features and targets each into an 80% 
# training sample and a 20% test sample, using 
# scikit-learn's 
# [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
# function
train_x, test_x, train_y, test_y = train_test_split(
  features,
  targets,
  test_size=0.2
)

# ## Make Predictions

# See what predictions the trained model generates for
# five new records (feature values only)
d = {
    'year'          :[2013, 2013, 2013],
    'day'           :[6.0,  2.0, 11.0],
    'hour'          :[11,   17,   22],
    'minute'        :[11,   17,   22],
    'month'         :[1,    6,    10],
    'sched_dep_time':[1100, 500, 1700]}
new_data = pd.DataFrame(data=d)


# ## AutoSklearn
import autosklearn
import autosklearn.classification


model2 = autosklearn.classification.AutoSklearnClassifier()


model2.fit(train_x, train_y)
dump(model2, 'autosklearn.joblib')


# ### Evaluate model2

model2.score(test_x, test_y)


# ### Interpret model2

test_pred = model2.predict(test_x)