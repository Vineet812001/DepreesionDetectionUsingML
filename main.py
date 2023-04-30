# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import os
from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC,RandomOverSampler,KMeansSMOTE
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle

data  = pd.read_csv('Finaldata1.csv')
data=data.drop('Unnamed: 0',axis=1)
# Now let's replace the '?' values with numpy nan
for column in data.columns:
    count = data[column][data[column]=='?'].count()
    if count!=0:
        data[column] = data[column].replace('?',np.nan)

# We can map the categorical values like below:
data['sex'] = data['sex'].map({'F': 0, 'M': 1})

# except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
# so instead of mapping indvidually, let's do a smarter work
for column in data.columns:
    if len(data[column].unique()) == 2:
        data[column] = data[column].map({'f': 0, 't': 1})

# this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.

lblEn = LabelEncoder()
data['Class'] =lblEn.fit_transform(data['Class'])

imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
new_array=imputer.fit_transform(data) # impute the missing values
    # convert the nd-array returned in the step above to a Dataframe
new_data=pd.DataFrame(data=np.round(new_array), columns=data.columns)

from sklearn.model_selection import train_test_split

X = new_data.drop(['Class'],axis=1)
y = new_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Creating Random Forest Model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train, y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))

filename = 'tyroid-prediction-model-final.pkl'
pickle.dump(random_forest_model, open(filename, 'wb'))
