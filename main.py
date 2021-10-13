import json
import os
import pickle
# import boto
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
import shap
from dotenv import load_dotenv
import glob
import requests
from datetime import datetime
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
    path = glob.glob('/valohai/inputs/loaded_data/*.csv')[0]
    if 'predict' in path:
        df = pd.read_csv(path, sep=';')
    else:
        df = pd.read_csv(path)

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
    print(len(X.columns))
    for col in X.columns:
        print(col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    clf = None
    if model == 'rf':
        clf = RandomForestClassifier(criterion='entropy', n_estimators=2000, min_samples_split=50,
                                     min_samples_leaf=20, max_features='sqrt', bootstrap=False,
                                     oob_score=False, random_state=2, class_weight='balanced')
    elif model == 'etc':
        clf = ExtraTreesClassifier(n_estimators=2000, min_samples_split=50, min_samples_leaf=20,
                                   class_weight='balanced', max_features='sqrt', random_state=2)
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
    filename = 'top_model.sav'
    pickle.dump(top_model, open('/valohai/outputs/' + filename, 'wb'))


def ready_data_for_bars():
    df_for_bars = pd.read_csv('/valohai/inputs/data_for_bars/processed_data.csv', sep=';')
    cols_to_drop = [col for col in df_for_bars.columns if 'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    df_for_bars = df_for_bars.drop([col for col in df_for_bars.columns if col in cols_to_drop], axis=1)
    class_1 = df_for_bars[df_for_bars['class'] == 1]
    class_0 = df_for_bars[df_for_bars['class'] == 0]
    amount_of_1 = class_1.shape[0]
    amount_of_0 = class_0.shape[0]
    num_of_samples = int((0.855 / 0.135) * amount_of_1)
    class_0_sampled = class_0.sample(n=num_of_samples, random_state=0)
    new_df_proportioned = pd.concat([class_1, class_0_sampled])
    new_df_proportioned = new_df_proportioned.reset_index()
    new_df_proportioned = new_df_proportioned.drop('class', axis=1)
    new_df_proportioned.to_csv('/valohai/outputs/data_ready_for_bars.csv')


def make_bars():
    df_for_bars = pd.read_csv('/valohai/inputs/processed_data/processed_data.csv')
    cols = df_for_bars.columns
    X = df_for_bars.copy().drop([col for col in cols if 'class' in col or 'index' in col
                                 or 'Unnamed: 0' in col or 'account_id' in col or 'has_won' in col
                                 or 'relevant_date' in col or 'period_range' in col], axis=1)
    print(X.columns)
    print(X.shape)
    low_bar_for_predict = X.quantile(.3)
    print(low_bar_for_predict)
    high_bar_for_predict = X.quantile(.8)
    print(high_bar_for_predict)
    low_bar_for_predict.to_csv('/valohai/outputs/low_bar_for_predict.csv')
    high_bar_for_predict.to_csv('/valohai/outputs/high_bar_for_predict.csv')


def predict():
    accounts = pd.read_csv('/valohai/inputs/processed_data/processed_data.csv')
    low_bar_for_predict = pd.read_csv('/valohai/inputs/low_bar_for_predict/low_bar_for_predict.csv', header=None,
                                      index_col=0, squeeze=True)
    high_bar_for_predict = pd.read_csv('/valohai/inputs/high_bar_for_predict/high_bar_for_predict.csv', header=None,
                                       index_col=0, squeeze=True)
    top_model = pickle.load(open('/valohai/inputs/top_model/top_model.sav', 'rb'))

    print("Top model is: ")
    print(top_model)
    cols_to_drop = [col for col in accounts.columns if 'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    accounts_clean = accounts.drop(cols_to_drop, axis=1).fillna(-1)
    explainer = shap.TreeExplainer(top_model)
    shap_mat = explainer.shap_values(accounts_clean)
    print(shap_mat)
    if len(np.array(shap_mat).shape) == 3:
        shap_mat = shap_mat[1]

    shap_df = pd.DataFrame(shap_mat, columns=accounts_clean.columns)
    accounts['proba'] = top_model.predict_proba(accounts_clean)[:, 1]

    accounts['rating'] = accounts['proba'].apply(
        lambda x: 'High' if x >= 0.6 else 'Medium' if x >= 0.35 else 'Low')

    final_payload = []
    for index, row in shap_df.iterrows():
        top_4 = row.nlargest(4)
        top_dict = top_4[top_4.gt(0)].to_dict()

        for key in top_dict:
            true_val = accounts_clean.loc[index, key]
            prob = accounts.loc[index, 'proba']
            rating = accounts.loc[index, 'rating']
            case_id = accounts.loc[index, 'case_id']
            relative_value = 'High' if true_val >= high_bar_for_predict[key] else 'Medium' if true_val >= \
                                                                                              low_bar_for_predict[
                                                                                                  key] else 'Low'
            top_dict[key] = {'case_id': case_id,
                         'prob': prob,
                         'rating': rating,
                         'feature': key,
                         'feature_value': true_val,
                         'relative_value': relative_value,
                         'shap_importance': top_dict[key]}
            final_payload.append(top_dict[key])

    num_of_accounts = accounts.shape[0]
    num_of_high = accounts['rating'].value_counts()['High']
    high_percentage = round(((num_of_high / num_of_accounts) * 100), 2)
    num_of_medium = accounts['rating'].value_counts()['Medium']
    medium_percentage = round(((num_of_medium / num_of_accounts) * 100), 2)
    num_of_low = accounts['rating'].value_counts()['Low']
    low_percentage = round(((num_of_low / num_of_accounts) * 100), 2)
    message = "Pro ==> Pro-X Model - Out of " + str(num_of_accounts) + " cases, " + str(
        num_of_high) + " cases were marked as high risk (" + str(high_percentage) + "%), " \
              + str(num_of_medium) + " cases were marked as medium risk (" + str(medium_percentage) + "%), " \
                                                                                                      "and " + str(
        num_of_low) + " cases were marked as low risk (" + str(low_percentage) + "%)."

    print(message)
    now_str = str(datetime.today())
    dict_for_post = {'Message': message,
                     'Created_Date': now_str}
    try:
        requests.post(url="https://www.workato.com/webhooks/rest/90076a68-0ba3-4091-aa8d-9da27893cfd6/test",
                      data=json.dumps(dict_for_post))
        print("Successfully sent update to the data-science slack channel.")
    except:
        print("Failed to sent message to data-science slack channel. please check the Workato recipe related to this.")

    final_prediction = pd.DataFrame(final_payload)
    final_prediction.to_csv('/valohai/outputs/final_prediction.csv')


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
