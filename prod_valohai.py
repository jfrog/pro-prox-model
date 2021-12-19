import json
import os
import pickle
import boto
from scipy import stats
from boto.s3.key import Key

import pandas as pd
import numpy as np
from utils.general_utils import load_data_valohai, get_cat_feature_names
from utils.preprocessing_utils import pro_upsell_preprocess, consolidate_opps
from utils.model_utils import cv_evaluation
from dotenv import load_dotenv

from names_mapping import names_mapping
from utils.report_utils import binning_features, create_output_table
from utils.valohai_utils import load_score_and_model

load_dotenv()


def load_data(sql_file_name):
    load_data_valohai(sql_file_name)


def process_train():
    df_train = pd.read_csv('/valohai/inputs/loaded_data/fit.csv', sep=';')
    df_train = consolidate_opps(df_train.copy())
    df_train = pro_upsell_preprocess(df_train.copy())
    df_train.to_csv('/valohai/outputs/processed_data.csv', index=False)


def fit_evaluate(model, n_folds=5):
    df_train = pd.read_csv('/valohai/inputs/processed_data/processed_data.csv')
    cols_to_drop = [col for col in df_train.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    X, y = df_train.drop(cols_to_drop, axis=1).fillna(-1), df_train['class']
    score, clf = cv_evaluation(model=model, X=X, y=y, n_folds=n_folds, scoring='average_precision', n_iter=20)
    pickle.dump(clf, open('/valohai/outputs/' + model + '.sav', 'wb'))
    pr_auc_dict = {'pr_auc': score}
    print("pr_auc is: " + str(score))
    with open('/valohai/outputs/' + model + '_pr_auc.json', 'w') as outfile:
        json.dump(pr_auc_dict, outfile)


def choose_best_model():
    rf, rf_pr_auc = load_score_and_model('rf')
    etc, etc_pr_auc = load_score_and_model('etc')
    cbc, cbc_pr_auc = load_score_and_model('cbc')
    hist, hist_pr_auc = load_score_and_model('hist')

    scores = [rf_pr_auc, etc_pr_auc, cbc_pr_auc, hist_pr_auc]
    models = [rf, etc, cbc, hist]
    max_pr_auc = np.max(scores)
    top_model = models[np.argmax(scores)]

    print("Top model is: " + str(top_model) + " with pr_auc of: " + str(max_pr_auc))
    pickle.dump(top_model, open('/valohai/outputs/' + 'top_model.sav', 'wb'))


def predict_explain():
    df_test = pd.read_csv('/valohai/inputs/loaded_data/predict.csv', sep=';')
    df_test = pro_upsell_preprocess(df_test.copy())
    cols_to_drop = [col for col in df_test.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    X_test = df_test.drop(cols_to_drop, axis=1).fillna(-1)

    top_model = pickle.load(open('/valohai/inputs/top_model/top_model.sav', 'rb'))

    res_df = pd.DataFrame()
    res_df['account_id'] = df_test['account_id']
    res_df['prob'] = top_model.predict_proba(X_test)[:, 1]
    threshold_to_high = res_df['prob'].quantile(0.85)
    threshold_to_medium = res_df['prob'].quantile(0.7)
    res_df['rating'] = np.where(res_df['prob'] < threshold_to_high,
                                np.where(res_df['prob'] < threshold_to_medium, 'low', 'medium'), 'high')
    X_test_disc = binning_features(X_test.copy())
    output_df = create_output_table(res_df, top_model, X_test, 5, X_test_disc)
    mapping = names_mapping()
    output_df = output_df.replace({"feature": mapping})
    output_df.to_csv('/valohai/outputs/final_prediction.csv', index=False)


def upload_to_s3():
    df_with_predictions = pd.read_csv('/valohai/inputs/final_prediction/final_prediction.csv')
    filename = 'final_prediction.csv'
    df_with_predictions.to_csv('/valohai/outputs/' + filename, index=False)
    AWS_KEY = os.getenv('AWS_KEY')
    AWS_SECRET = os.getenv('AWS_SECRET')
    AWS_BUCKET = boto.connect_s3(AWS_KEY, AWS_SECRET).get_bucket('prod-is-data-science-bucket')
    s3_upload_folder_path = os.getenv('S3_PATH')
    local_path = '/valohai/outputs/' + filename
    key = Key(AWS_BUCKET, s3_upload_folder_path + filename)
    key.set_contents_from_filename(local_path)









