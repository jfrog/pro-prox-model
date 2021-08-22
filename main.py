from datetime import datetime
import os
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from utils.fe_utils import *
from sklearn.metrics import precision_recall_curve, auc
from utils.general_utils import *
import pickle
import boto
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import requests

load_dotenv()


def load_data(sql_file_name):
    load_data_valohai(sql_file_name)


def dummy_step1():
    df = pd.read_csv('/valohai/inputs/loaded_data/loaded_data.csv')
    # Here you do some stuff with your data...
    df.to_csv('/valohai/outputs/processed_data.csv')


def dummy_step2():
    processed_df = pd.read_csv('/valohai/inputs/processed_data/processed_data.csv')
    # Here you do some stuff with your data
    processed_df.to_csv('/valohai/outputs/data_with_predictions.csv')


def upload_to_s3():
    df_with_predictions = pd.read_csv('/valohai/inputs/data_with_predictions/data_with_predictions.csv')
    filename = 'final_prediction.csv'
    df_with_predictions.to_csv('/valohai/outputs/' + filename)
    AWS_KEY = os.getenv('AWS_KEY')
    AWS_SECRET = os.getenv('AWS_SECRET')
    AWS_BUCKET = boto.connect_s3(AWS_KEY, AWS_SECRET).get_bucket('prod-is-data-science-bucket')
    s3_upload_folder_path = 'csat_model/valohai/upload/'
    local_path = '/valohai/outputs/' + filename
    key = Key(AWS_BUCKET, s3_upload_folder_path + filename)
    key.set_contents_from_filename(local_path)
