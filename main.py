import json
import os
import pickle
import boto
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from utils.general_utils import load_data_old, load_data_valohai, get_cat_feature_names
from utils.model_extensions_utils import FocalLossObjective
from utils.plot_utils import Evaluation
from utils.fe_utils import get_growth_features
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
import glob

load_dotenv()


def load_data(sql_file_name):
    load_data_valohai(sql_file_name)


def consolidate_opps():
    path = glob.glob('/valohai/inputs/loaded_data/*.csv')[0]
    df = pd.read_csv(path, sep=';')
    has_won = df.groupby('account_id', as_index=False).sum('class').loc[:, ['account_id', 'class']]
    has_won['has_won'] = has_won['class'].apply(lambda x: True if x > 0 else False)
    has_won.drop('class', axis=1, inplace=True)
    new_df = df.merge(has_won, on='account_id')
    df_did_win, df_did_not_win = new_df[new_df['has_won']], new_df[~new_df['has_won']]
    df_did_win = df_did_win[df_did_win['class'] == 1].groupby('account_id', as_index=False).min('relevant_date')
    df_did_not_win = df_did_not_win.groupby('account_id').sample(n=1, random_state=2)
    df = pd.concat([df_did_win, df_did_not_win])
    df = df.sample(frac=1, random_state=2).reset_index(drop=True)
    df.to_csv('/valohai/outputs/loaded_data.csv', index=False)


def process_df():
    df = pd.read_csv('/valohai/inputs/loaded_data/loaded_data.csv')
    # - remove zero variance features
    # cols_to_drop = [col for col in X_temp.select_dtypes(include=np.number).columns if np.std(X_temp[col]) == 0]
    # X_temp = X_temp.drop(cols_to_drop, axis=1)
    tech_cols = ['maven', 'generic', 'docker', 'npm', 'pypi', 'gradle', 'nuget']
    usage_cols = tech_cols + ['artifacts_count', 'artifacts_size', 'binaries_count', 'binaries_size', 'items_count',
                              'number_of_permissions', 'internal_groups', 'number_of_users', 'n_env', 'n_tech',
                              'n_repos']
    df['n_tech'] = (df[tech_cols] != 0).astype(int).sum(axis=1)
    df['n_tech.1'] = (df[[col + '.1' for col in tech_cols]] != 0).astype(int).sum(axis=1)
    df['n_tech.2'] = (df[[col + '.2' for col in tech_cols]] != 0).astype(int).sum(axis=1)
    df['n_tech.3'] = (df[[col + '.3' for col in tech_cols]] != 0).astype(int).sum(axis=1)
    df['n_tech.4'] = (df[[col + '.4' for col in tech_cols]] != 0).astype(int).sum(axis=1)
    df['n_repos'] = (df[tech_cols]).sum(axis=1)
    df['n_repos.1'] = (df[[col + '.1' for col in tech_cols]]).sum(axis=1)
    df['n_repos.2'] = (df[[col + '.2' for col in tech_cols]]).sum(axis=1)
    df['n_repos.3'] = (df[[col + '.3' for col in tech_cols]]).sum(axis=1)
    df['n_repos.4'] = (df[[col + '.4' for col in tech_cols]]).sum(axis=1)
    # X_temp['leading_tech'] = list(X_temp[tech_cols].idxmax(axis=1))
    # X_temp.loc[X_temp['leading_tech'].isin(['npm', 'gradle', 'pypi']), 'leading_tech'] = 'else'

    # - get trends features
    for col in usage_cols:
        growth_feature_monthly, growth_feature_quarter, df_fg = get_growth_features(col, df.copy())
        df[col + '_monthly_growth'] = growth_feature_monthly
        df[col + '_quarter_growth'] = growth_feature_quarter

    # - transform to category
    cat_features = get_cat_feature_names(df)
    for col in cat_features:
        df[col] = df[col].astype('category')

    # - drop usage features from the periods before the relevant-date
    cols_to_drop = [col for col in df.columns if '.1' in col or '.2' in col or '.3' in col or '.4' in col]
    df = df.drop(cols_to_drop, axis=1)
    df['artifacts/binaries_size'] = np.where(df['binaries_size'] == 0, 0, df['artifacts_size'] / df['binaries_size'])
    df['artifacts/binaries_count'] = np.where(df['binaries_count'] == 0, 0,
                                              df['artifacts_count'] / df['binaries_count'])
    df = df.drop(['total_employees_with_details', 'days_from_contact_added', 'territory', 'industry_group',
                  'total_employees_range'], axis=1)
    df.to_csv('/valohai/outputs/processed_data.csv', index=False)


def fit(model: str):
    new_df_proportioned = pd.read_csv('/valohai/inputs/processed_data/processed_data.csv')
    cols_to_drop = [col for col in new_df_proportioned.columns if 'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    X, y = new_df_proportioned.drop(cols_to_drop, axis=1).fillna(-1), new_df_proportioned['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    clf = None
    if model == 'rf':
        clf = RandomForestClassifier(criterion='entropy', n_estimators=2000, min_samples_split=50,
                                     min_samples_leaf=20, max_features='sqrt', bootstrap=False,
                                     oob_score=False, random_state=44, class_weight='balanced')
    elif model == 'etc':
        clf = ExtraTreesClassifier(n_estimators=2000, min_samples_split=50, min_samples_leaf=20,
                                   class_weight='balanced', max_features='sqrt', random_state=44)
    elif model == 'cbc':
        clf = CatBoostClassifier(cat_features=get_cat_feature_names(X), auto_class_weights="Balanced", random_state=5,
                                 bootstrap_type='Bayesian', rsm=0.1, verbose=0)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
    pr_auc = auc(recall, precision)
    filename = model + '.sav'
    pickle.dump(clf, open('/valohai/outputs/' + filename, 'wb'))
    pr_auc_dict = {'pr_auc': pr_auc}
    print("pr_auc is: ")
    print(pr_auc)
    with open('/valohai/outputs/' + model + '_pr_auc.json', 'w') as outfile:
        json.dump(pr_auc_dict, outfile)


def choose():
    top_model = None
    max_pr_auc = 0.0
    rf = pickle.load(open('/valohai/inputs/rf/rf.sav', 'rb'))
    f = open('/valohai/inputs/rf_pr_auc/rf_pr_auc.json', "r")
    rf_pr_auc = json.loads(f.read())['pr_auc']

    etc = pickle.load(open('/valohai/inputs/etc/etc.sav', 'rb'))
    f = open('/valohai/inputs/etc_pr_auc/etc_pr_auc.json', "r")
    etc_pr_auc = json.loads(f.read())['pr_auc']

    cbc = pickle.load(open('/valohai/inputs/cbc/cbc.sav', 'rb'))
    f = open('/valohai/inputs/cbc_pr_auc/cbc_pr_auc.json', "r")
    cbc_pr_auc = json.loads(f.read())['pr_auc']

    if rf_pr_auc >= max_pr_auc:
        top_model = rf
        max_pr_auc = rf_pr_auc
    if etc_pr_auc >= max_pr_auc:
        top_model = etc
        max_pr_auc = etc_pr_auc
    if cbc_pr_auc >= max_pr_auc:
        top_model = cbc
        max_pr_auc = cbc_pr_auc

    print("Top model is: ")
    print(top_model)
    print("With pr_auc of: " + str(max_pr_auc))
    filename = 'top_model_predictive_cse.sav'
    pickle.dump(top_model, open('/valohai/outputs/' + filename, 'wb'))
    with open('/valohai/outputs/' + filename + '.metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)


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
