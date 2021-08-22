import psycopg2
import csv
import pandas as pd
import os
import numpy as np
import pickle
from datetime import date
from dotenv import load_dotenv

load_dotenv()


# - load data from Red Shift
def load_data_valohai(query_name):
    """

    :param query_name: the name of the query (saved in sql file)
    :return: a data frame which is the output of the specified query
    """
    # idea_path = os.getenv('VH_REPOSITORY_DIR', os.getenv('path') + os.getcwd().rsplit('/', 1)[-1])
    idea_path = '/valohai/repository'
    query = open(idea_path + '/sql/' + query_name, 'r')
    print(os.getenv('dbname'))
    con = psycopg2.connect(dbname=os.getenv('dbname'), host=os.getenv('host'),
                           port=os.getenv('port'), user=os.getenv('user'), password=os.getenv('password'))
    cur = con.cursor()
    q = query.read()
    cur.execute(q)
    rows = cur.fetchall()
    # path = open(idea_path + os.getcwd().rsplit('/', 1)[-1] + '/data/' + query_name[:-4] + '.csv', 'w')
    OUTPUT_PATH = os.getenv('VH_OUTPUTS_DIR', idea_path + os.getcwd().rsplit('/', 1)[-1])
    path = open(os.path.join(OUTPUT_PATH, query_name[:-4] + '.csv'), 'w')
    myFile = csv.writer(path, delimiter=';')
    myFile.writerow(col[0] for col in cur.description)
    myFile.writerows(rows)
    query.close()
    # query_result = pd.read_csv(path, delimiter=';', header=0)
    path.close()
    # return query_result


# - load data from Red Shift
def load_data_old(query_name):
    """

    :param query_name: the name of the query (saved in sql file)
    :return: a data frame which is the output of the specified query
    """
    idea_path = os.getenv('path')
    query = open(idea_path + os.getcwd().rsplit('/', 1)[-1] + '/sql/' + query_name, 'r')
    con = psycopg2.connect(dbname=os.getenv('dbname'), host=os.getenv('host'),
                           port=os.getenv('port'), user=os.getenv('user'), password=os.getenv('password'))
    cur = con.cursor()
    q = query.read()
    cur.execute(q)
    rows = cur.fetchall()
    path = open(idea_path + os.getcwd().rsplit('/', 1)[-1] + '/data/' + query_name[:-4] + '.csv', 'w')
    myFile = csv.writer(path, delimiter=';')
    myFile.writerow(col[0] for col in cur.description)
    myFile.writerows(rows)
    query.close()
    path.close()
    query_result = pd.read_csv(idea_path + os.getcwd().rsplit('/', 1)[-1] + '/data/' + query_name[:0 - 4] + '.csv',
                               delimiter=';', header=0)
    return query_result


# - not used in this project
def get_cat_features(x):
    """
    :param x: a Data Frame
    :return: indices of categorical columns
    """
    return np.where((x.dtypes != np.float) & (x.dtypes != np.int))[0]


# - returns the names of the categorical features in the input (used for Catboost)
def get_cat_feature_names(X):
    """

    :param X: the input features (a DataFrame)
    :return: the names of the categorical features
    """
    return [col for col in X.columns if X.dtypes[col] not in [np.int, np.float]]


# - get the technologies names
def get_technologies():
    return ['maven', 'generic', 'buildinfo', 'docker', 'npm', 'pypi', 'gradle', 'nuget', 'yum',
            'helm', 'gems', 'debian', 'ivy', 'sbt', 'conan', 'bower', 'go', 'chef', 'gitlfs',
            'composer', 'puppet', 'conda', 'vagrant', 'cocoapods', 'cran', 'opkg', 'p2', 'vcs', 'alpine']


# - load the trials data, and the training leads data with a pickle.  If it was loaded from the db lately than load the
# corresponding pickle, otherwise load it from the db using load_data method. In the one hand we will not be up-to-date
# in each time we work with the training data, but on the other hand we will not load the queries each time

def load_train_trials_pickle(days_between_loads=7):
    """

    :param days_between_loads: how frequently we would like to load the data from the db
    (instead of the pickle which is not up-to-date)
    :return: the leads data of the training set  (labeled data), and the trials data
    """
    last_load_time = pickle.load(open(r'pickle/last_load_time.pkl', 'rb'))
    if (date.today() - last_load_time).days >= days_between_loads:
        df_train = load_data('get_data_train.sql')
        df_trials = load_data('get_trials.sql')
        pickle.dump(df_train, open(r'pickle/df_train.pkl', 'wb'))
        pickle.dump(df_trials, open(r'pickle/df_trails.pkl', 'wb'))
        last_load_time = date.today()
        pickle.dump(last_load_time, open(r'pickle/last_load_time.pkl', 'wb'))
    else:
        df_train = pickle.load(open(r'pickle/df_train.pkl', 'rb'))
        df_trials = pickle.load(open(r'pickle/df_trails.pkl', 'rb'))
    return df_train, df_trials
