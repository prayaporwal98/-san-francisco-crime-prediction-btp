# san-francisco-crime-prediction-bt

this project is given on kaggle as San Francisco Crime Classification Predict the category of crimes that occurred in the city by the bay

I have completed this project using python. i have given detailed overview of the project in btp report file .

there are two datasets are given one as train dataset and second as test sataset the link to the data sets is https://www.kaggle.com/c/sf-crime/data.

we have to train our model using trin dataset and after training of model apply it on test dataset to evaluate the result.

This dataset contains incidents derived from SFPD Crime Incident Reporting system. The data ranges from 1/1/2003 to 5/13/2015. The training set and test set rotate every week, 
meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to training set. 

### Introduction

SF Crime Dataset
This dataset contains incidents derived from SFPD Crime Incident Reporting system. The data ranges from 1/1/2003 to 5/13/2015.
The training set and test set rotate every week, meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to training set.
The goal is to try to predict the category of crime that occurred in the city of San Francisco.

Data Fields
Dates - timestamp of the crime incident
Category - category of the crime incident (only in train.csv). This is the target variable you are going to predict.
Descript - detailed description of the crime incident (only in train.csv)
DayOfWeek - the day of the week
PdDistrict - name of the Police Department District
Resolution - how the crime incident was resolved (only in train.csv)
Address - the approximate street address of the crime incident
X - Longitude
Y - Latitude


## The libraries which we have to import are:-

# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import xgboost as xgb

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Metrics 
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

# Model Selection & Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space  import Real, Categorical, Integer


# Clustering
from sklearn.cluster import KMeans

# Mathematical Functions
import math


