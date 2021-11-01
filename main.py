import json
import os
import pickle
import boto
from scipy import stats
from boto.s3.key import Key
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from utils.general_utils import load_data_valohai, get_cat_feature_names
from utils.model_extensions_utils import FocalLossObjective
from utils.fe_utils import get_growth_features
import shap
from dotenv import load_dotenv
import glob
import requests
from datetime import datetime

load_dotenv()


def translate_feature(feature: str):
    mapping = {'artifacts_size': 'Storage: artifacts size',
               'artifacts_count': 'Storage: artifacts count',
               'binaries_size': 'Storage: binaries size',
               'binaries_count': 'Storage: binaries count',
               'items_count': 'Storage: items count',
               'number_of_permissions': 'Number of permissions',
               'internal_groups': 'Number of internal groups',
               'number_of_users': 'Number of users',
               'maven': 'Maven repositories',
               'generic': 'Generic repositories',
               'docker': 'Docker repositories',
               'n_ent_trials': 'Number of Enterprise trials last two years',
               'n_contacts': 'Number of contacts',
               'n_active_contracts': 'Number of active contracts',
               'is_cotermed': 'Is previously co-termed?',
               'months_from_upgrade': 'Number of months passed since upgrade from Pro to Pro X',
               'n_sessions_last_year': 'Number of sessions last year',
               'n_cases_last_year': 'Number of cases last year',
               'avg_resolution_days': 'Average resolution days for a case',
               'industry_group': 'Industry',
               'total_employees_range': 'Number of employees (range)',
               'company_age': 'Company age (years)',
               'seniority': 'Seniority in JFrog (months)',
               'days_from_contact_added': 'Number of days since last contact added',
               'days_from_artifacts_size_change': 'Number of days since artifacts size changed',
               'days_from_artifacts_count_change': 'Number of days since artifacts count changed',
               'days_from_binaries_size_change': 'Number of days since binaries size changed',
               'days_from_binaries_count_change': 'Number of days since binaries count changed',
               'days_from_items_count_change': 'Number of days since items count changed',
               'days_from_permissions_change': 'Number of days since number of permissions changed',
               'days_from_internal_groups_change': 'Number of days since number of internal groups changed',
               'days_from_users_change': 'Number of days since number of users changed',
               'total_security_policies': 'Number of secured policies (Xray)',
               'days_since_xray_task': 'Number of days since Xray task',
               'n_sent': 'Number of Emails sent',
               'days_since_reply': 'Number of days since the account replayed to an Email',
               'days_since_sent': 'Number of days since an Email sent to the account',
               'n_repos': 'Number of repositories',
               'leading_tech': 'Leading technology',
               'generic_monthly_growth': 'Generic monthly growth',
               'maven_monthly_growth': 'Generic monthly growth',
               'docker_monthly_growth': 'Docker monthly growth',
               'artifacts_count_monthly_growth': 'Artifacts count monthly growth',
               'artifacts_size_monthly_growth': 'Artifacts size monthly growth',
               'binaries_count_monthly_growth': 'Binaries count monthly growth',
               'binaries_size_monthly_growth': 'Binaries size monthly growth',
               'items_count_monthly_growth': 'Items count monthly growth',
               'number_of_users_monthly_growth': 'Number of users monthly growth',
               'n_repos_monthly_growth': 'Number of repositories monthly growth',
               'number_of_permissions_monthly_growth': 'Number of permissions monthly growth',
               'internal_groups_monthly_growth': 'Number of internal groups monthly growth',
               'generic_quarter_growth': 'Generic quarterly growth',
               'maven_quarter_growth': 'Generic quarterly growth',
               'docker_quarter_growth': 'Docker quarterly growth',
               'npm_quarter_growth': 'Npm quarterly growth',
               'artifacts_count_quarter_growth': 'Artifacts count quarterly growth',
               'artifacts_size_quarter_growth': 'Artifacts size quarterly growth',
               'binaries_count_quarter_growth': 'Binaries count quarterly growth',
               'binaries_size_quarter_growth': 'Binaries size quarterly growth',
               'items_count_quarter_growth': 'Items count quarterly growth',
               'number_of_users_quarter_growth': 'Number of users quarterly growth',
               'n_repos_quarter_growth': 'Number of repositories quarterly growth',
               'number_of_permissions_quarter_growth': 'Number of permissions quarterly growth',
               'internal_groups_quarter_growth': 'Number of internal groups quarterly growth',
               }

    if feature in mapping:
        translation = mapping[feature]
    else:
        translation = feature

    return translation


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
    cols_to_drop = [col for col in new_df_proportioned.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    X, y = new_df_proportioned.drop(cols_to_drop, axis=1).fillna(-1), new_df_proportioned['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    est = None
    params = None

    if model == 'rf':
        params = {'n_estimators': [200, 500, 1000],
                  'max_depth': stats.randint(3, 10),
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_split': stats.randint(3, 10),
                  'min_samples_leaf': stats.randint(1, 5)}
        est = RandomForestClassifier(criterion='entropy', n_estimators=2000, min_samples_split=5,
                                     min_samples_leaf=2, max_features='sqrt', bootstrap=True,
                                     oob_score=True, random_state=2, class_weight='balanced')
    elif model == 'etc':
        params = {'n_estimators': [200, 500, 1000],
                  'max_depth': stats.randint(3, 10),
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_split': stats.randint(3, 10),
                  'min_samples_leaf': stats.randint(1, 5),
                  'bootstrap': [True, False]}
        est = ExtraTreesClassifier(n_estimators=2000, min_samples_split=50, min_samples_leaf=20,
                                   class_weight='balanced', max_features='sqrt', random_state=2)
    elif model == 'cbc':
        params = {'iterations': [100, 250, 500, 1000],
                  'learning_rate': stats.uniform(0.01, 0.3),
                  'max_depth': stats.randint(3, 10),
                  'l2_leaf_reg': stats.reciprocal(a=1e-2, b=1e1),
                  'border_count': [5, 10, 20, 50, 100, 200],
                  'bootstrap_type': ['Bernoulli', 'Bayesian', 'MVS']}
        est = CatBoostClassifier(cat_features=get_cat_feature_names(X), auto_class_weights="Balanced", random_state=5,
                                 bootstrap_type='Bayesian', rsm=0.1, verbose=0, loss_function=FocalLossObjective(),
                                 eval_metric="Logloss")
    elif model == 'hist':
        params = {'max_iter': [100, 250, 500, 1000],
                  'max_leaf_nodes': stats.randint(2, 100),
                  'learning_rate': stats.uniform(0.01, 0.3),
                  'max_depth': stats.randint(3, 10),
                  'min_samples_leaf': stats.randint(1, 30)}
        est = HistGradientBoostingClassifier(categorical_features=get_cat_feature_names(X), verbose=0,
                                             random_state=5, loss="auto", scoring="Logloss")

    clf = RandomizedSearchCV(estimator=est, param_distributions=params,
                             scoring='average_precision',
                             refit=True, random_state=5, cv=4, n_iter=20, verbose=2, n_jobs=-1)
    clf.fit(X_train, y_train)
    clf_after_search = clf.best_estimator_
    probas = clf.predict_proba(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
    pr_auc = auc(recall, precision)
    filename = model + '.sav'
    pickle.dump(clf_after_search, open('/valohai/outputs/' + filename, 'wb'))
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

    hist = pickle.load(open('/valohai/inputs/hist/hist.sav', 'rb'))
    f = open('/valohai/inputs/hist_pr_auc/hist_pr_auc.json', "r")
    hist_pr_auc = json.loads(f.read())['pr_auc']

    if rf_pr_auc >= max_pr_auc:
        top_model = rf
        max_pr_auc = rf_pr_auc
    if etc_pr_auc >= max_pr_auc:
        top_model = etc
        max_pr_auc = etc_pr_auc
    if cbc_pr_auc >= max_pr_auc:
        top_model = cbc
        max_pr_auc = cbc_pr_auc
    if hist_pr_auc >= max_pr_auc:
        top_model = hist
        max_pr_auc = hist_pr_auc

    print("Top model is: ")
    print(top_model)
    print("With pr_auc of: " + str(max_pr_auc))
    filename = 'top_model.sav'
    pickle.dump(top_model, open('/valohai/outputs/' + filename, 'wb'))


def ready_data_for_bars():
    df_for_bars = pd.read_csv('/valohai/inputs/data_for_bars/processed_data.csv', sep=';')
    cols_to_drop = [col for col in df_for_bars.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
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
    processed_df_for_fit = pd.read_csv('/valohai/inputs/processed_data_for_fit/processed_data.csv')
    cols_to_drop = [col for col in processed_df_for_fit.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    processed_df_for_fit = processed_df_for_fit.drop(cols_to_drop, axis=1).fillna(-1)

    low_bar_for_predict = pd.read_csv('/valohai/inputs/low_bar_for_predict/low_bar_for_predict.csv', header=None,
                                      index_col=0, squeeze=True)
    high_bar_for_predict = pd.read_csv('/valohai/inputs/high_bar_for_predict/high_bar_for_predict.csv', header=None,
                                       index_col=0, squeeze=True)
    top_model = pickle.load(open('/valohai/inputs/top_model/top_model.sav', 'rb'))

    print("Top model is: ")
    print(top_model)
    cols_to_drop = [col for col in accounts.columns if
                    'period_range' in col or 'relevant_date' in col or 'account_id' in col
                    or 'class' in col or 'has_won' in col]
    accounts_clean = accounts.drop(cols_to_drop, axis=1).fillna(-1)
    explainer = shap.TreeExplainer(top_model)
    shap_mat = explainer.shap_values(accounts_clean)
    # print(shap_mat)
    if len(np.array(shap_mat).shape) == 3:
        shap_mat = shap_mat[1]

    shap_df = pd.DataFrame(shap_mat, columns=accounts_clean.columns)
    accounts['proba'] = top_model.predict_proba(accounts_clean)[:, 1]
    high_bar_for_proba = accounts['proba'].quantile(.85)
    low_bar_for_proba = accounts['proba'].quantile(.7)
    accounts['rating'] = accounts['proba'].apply(
        lambda x: 'High' if x >= high_bar_for_proba else 'Medium' if x >= low_bar_for_proba else 'Low')

    final_payload = []
    for index, row in shap_df.iterrows():
        top_10 = row.nlargest(10)
        top_dict = top_10[top_10.gt(0)].to_dict()
        bottom_10 = row.nsmallest(10)
        bottom_dict = bottom_10[bottom_10.lt(0)].to_dict()
        dicts = [top_dict, bottom_dict]

        for dict in dicts:
            for key in dict:
                true_val = accounts_clean.loc[index, key]
                prob = accounts.loc[index, 'proba']
                rating = accounts.loc[index, 'rating']
                account_id = accounts.loc[index, 'account_id']
                relative_value = 'High' if true_val >= high_bar_for_predict[key] else 'Medium' if true_val >= \
                                                                                                  low_bar_for_predict[
                                                                                                      key] else 'Low'
                dict[key] = {'account_id': account_id,
                             'prob': prob,
                             'rating': rating,
                             'feature': translate_feature(key),
                             'feature_value': true_val,
                             'relative_value': relative_value,
                             'shap_importance': dict[key]}
                final_payload.append(dict[key])

    num_of_accounts = accounts.shape[0]
    num_of_high = accounts['rating'].value_counts()['High']
    high_percentage = round(((num_of_high / num_of_accounts) * 100), 2)
    num_of_medium = accounts['rating'].value_counts()['Medium']
    medium_percentage = round(((num_of_medium / num_of_accounts) * 100), 2)
    num_of_low = accounts['rating'].value_counts()['Low']
    low_percentage = round(((num_of_low / num_of_accounts) * 100), 2)

    ### WHAT IF ANALYSIS
    scaler = StandardScaler()
    bad_accounts = accounts_clean[accounts['rating'] != 'High']
    pred_class_for_train_data = top_model.predict_proba(processed_df_for_fit)[:, 1]
    processed_df_for_fit['class'] = pred_class_for_train_data
    train_data_for_whatif = processed_df_for_fit.loc[processed_df_for_fit['class'] >= high_bar_for_proba, :].drop(
        'class', axis=1)
    cat_cols = get_cat_feature_names(train_data_for_whatif)
    train_data_for_whatif['cat_val'] = train_data_for_whatif[cat_cols].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    bad_accounts['cat_val'] = bad_accounts[cat_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    for index, row in bad_accounts.iterrows():
        print("train_data_for_whatif.shape")
        print(train_data_for_whatif.shape)
        train_data_subset = train_data_for_whatif.loc[train_data_for_whatif['cat_val'] == row['cat_val'], :]
        print("train_data_subset.shape")
        print(train_data_subset.shape)
        train_data_subset_w_instance = pd.concat([train_data_subset, row])
        print("train_data_subset_w_instance.shape")
        print(train_data_subset_w_instance.shape)
        train_data_subset_w_instance = train_data_subset_w_instance.drop(['cat_val', 0], axis=1)
        print("train_data_subset_w_instance.shape")
        print(train_data_subset_w_instance.shape)

        df_whatif_scaled = pd.DataFrame(scaler.fit_transform(train_data_subset_w_instance), columns=train_data_subset_w_instance.columns)
        print("df_whatif_scaled")
        print(df_whatif_scaled)
        df_whatif_scaled = df_whatif_scaled.fillna(0)
        print("df_whatif_scaled after fillna")
        print(df_whatif_scaled)
        sample = df_whatif_scaled.iloc[-1]
        print("sample.shape")
        print(sample.shape)
        df_whatif_scaled_wo_sample = df_whatif_scaled.iloc[:-1, :]
        print("df_whatif_scaled_wo_sample.shape")
        print(df_whatif_scaled_wo_sample.shape)
        dists = [euclidean(sample, df_whatif_scaled_wo_sample.iloc[i]) for i in (range(df_whatif_scaled_wo_sample.shape[0]))]
        print(dists)
        print(np.argmin(dists))
        print(train_data_subset.shape)
        print(df_whatif_scaled_wo_sample.shape)
        closet_obs1 = df_whatif_scaled_wo_sample.iloc[np.argmin(dists)]
        print(closet_obs1)
        closet_obs2 = train_data_subset.iloc[np.argmin(dists)]
        shap_values_train = shap.TreeExplainer(top_model).shap_values(closet_obs)
        shap_values_sample = shap.TreeExplainer(top_model).shap_values(row)
        print('SHAP 1')
        print(shap_values_train)
        print('SHAP 1')
        print(shap_values_sample)

        # TODO: predict class for train data V
        # TODO: filter only high class V
        # TODO: for both train data and new data, add column for categorical features V
        # TODO: for each instance of new data in iteration, bring train data of same categorical values V
        # TODO: attach the subset of train data with the current instance V
        # TODO: remove the newly added categorical column V
        # TODO: scale the concated df V
        # TODO: find nearest neighbour for the instance we care for from the other data set V
        # TODO: Calculate shap values for the neighboor's features and the current instance's features, calculate diffs
        # TODO: Create recommendations based on the top diff feature

    message = "Pro ==> Pro-X Model - Out of " + str(num_of_accounts) + " accounts, " + str(
        num_of_high) + " accounts were marked as high chance of upgrading (" + str(high_percentage) + "%), " \
              + str(num_of_medium) + " accounts were marked as medium chance of upgrading (" + str(
        medium_percentage) + "%), " \
                             "and " + str(
        num_of_low) + " accounts were marked as low chance of upgrading (" + str(low_percentage) + "%)."

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
    final_prediction.to_csv('/valohai/outputs/final_prediction.csv', index=False)


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
