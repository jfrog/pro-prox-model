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



