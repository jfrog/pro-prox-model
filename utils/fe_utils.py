import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import json
import socket
import re
from utils.general_utils import get_cat_features, get_cat_feature_names, get_technologies
from utils.model_extensions_utils import FocalLossObjective
from sklearn.ensemble import IsolationForest
from scipy import stats
from collections import Counter
import hashlib


# - add the anomaly-score feature; optional in this project (currently not used)
def add_anomaly_score_feature(x_train, x_test):
    """

    :param x_train: The training set (a DataFrame)
    :param x_test: The tests set (a DataFrame)
    :return: x_train, x_test with the anomaly score feature (produced from one-class SVM;
    an unsupervised anomaly-detection algorithm. Note this feature might be different in its distribution
    between train and tests.
    """
    x_tr_numeric = x_train.select_dtypes(include=np.number).fillna(x_train.mean())
    ocsvm = OneClassSVM()
    x_train['anomaly_score'] = -1 * ocsvm.fit(x_tr_numeric).score_samples(x_tr_numeric)
    x_ts_numeric = x_test.select_dtypes(include=np.number).fillna(x_train.mean())
    x_test['anomaly_score'] = -1 * ocsvm.score_samples(x_ts_numeric)
    return x_train, x_test


# - note used in this project
def add_cluster_feature(X, is_train=True, model=None, scaler=None):
    features_to_keep = ['debian_repos',
                        'n_tech_std',
                        'a_count_avg/b_count_avg',
                        'b_count_avg',
                        'a_size_avg/b_size_avg',
                        'a_count_avg',
                        'b_size_avg',
                        'a_size_avg',
                        'number_of_users',
                        'gems_repos',
                        'maven_repos',
                        'n_tech',
                        'generic_repos',
                        'npm_repos',
                        'docker_repos']
    X = X[features_to_keep]
    if is_train:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        model = KMeans(n_clusters=4, init='k-means++', max_iter=3000, n_init=10, random_state=0)
        model.fit(X)
    else:
        X = scaler.transform(X)
    if is_train:
        return model.predict(X), model, scaler
    else:
        return model.predict(X)


# - not used in this project
def get_domains_validation(X):
    is_valid_list = []
    with open('cache/validated_domains.json', mode='r+', encoding='utf-8') as domain_validation_cache:
        domain_validation_dict = json.load(domain_validation_cache)
        for ind in tqdm(range(X.shape[0])):
            row = X.loc[ind, :]
            email_user_name = row['email'].split('@')[0]
            if email_user_name.isdigit() or 'tests' in email_user_name:
                is_valid = 0
            else:
                if row['is_general'] == 0:
                    if row['domain'] in domain_validation_dict:
                        is_valid = domain_validation_dict[row['domain']]
                    else:
                        try:
                            is_valid = 1 if verify_email(row['email']) \
                                else 1 if type(socket.gethostbyname(row['domain'])) == str else 0
                        except:
                            is_valid = 0
                        # is_valid = 1 if type(WHOIS(row['domain']).creation_date()) == type(datetime.date(2020, 1, 1)) else 0
                        temp_dict = {str(row['domain']): is_valid}
                        domain_validation_dict.update(temp_dict)
                        domain_validation_cache.seek(0)
                        json.dump(domain_validation_dict, domain_validation_cache)
                else:
                    is_valid = 1
            is_valid_list.append(is_valid)
    return is_valid_list


# - for each lead that provided a phone number - check if it valid
def is_phone_valid(phones):
    """

    :param phones: phones for each lead (a Series or list)
    :return: a list with either - 1. "Phone number was not provided" for those we didn't provide a phone number
                                  2. "valid" for those provided a valid phone number
                                  3. "invalid" for those provided an invalid phone number
    """
    is_valid_phone = []
    for phone in phones:
        if phone == 'unknown':
            is_valid_phone.append('Phone number was not provided')
        else:
            try:
                z = phonenumbers.parse(phone, None)
                is_valid_phone.append('valid' if phonenumbers.is_valid_number(z) else 'invalid')
            except:
                is_valid_phone.append('invalid')
    return is_valid_phone


# - for each lead extract its relevant set features (including: feature engineering, unify categories)
def fe_leads(X):
    """

    :param X: The input (a DataFrame)
    :return: a modified version of X (a DataFrame)
    """
    X['has_valid_phone'] = is_phone_valid(X.phone)
    X['country/territory'] = np.where(X['territory'] == 'APAC', 'APAC', X['country'])
    X['lead_source'] = np.where(X['leadsource'].isin(['other', 'Webinar', 'Gated Content']), 'other',
                                X['leadsource'])

    X['is_student_email'] = X['email'].str.contains('|'.join(['\.edu', '\.ac', '\.college'])).astype(int)
    X = X.drop(['leadsource', 'country', 'phone', 'email', 'relevant_date'], axis=1)
    analytics_cols = [col for col in X.columns if 'total_' in col]
    if len(analytics_cols) > 0:
        X['leading_analytics'] = list(X[analytics_cols].idxmax(axis=1))
    cs_cols = [col for col in X if 'CS-' in col]
    for col in cs_cols:
        X[col] = np.where(X[col] > 0, 1, 0)
    X['n_reasons / time to mql'] = np.where(X['days_to_mql'] == 0, 0,
                                            X['n_reasons'] / X['days_to_mql'])
    X['n_attempts / time in mql'] = np.where(X['days_to_decision'] == 0, 0,
                                             X['number_of_attempts'] / X['days_to_decision'])
    X['n_engagements / time in mql'] = np.where(X['days_to_decision'] == 0, 0,
                                                X['number_of_engagements'] / X['days_to_decision'])
    return X


# - for each lead that did a trial extract its usage-related features (e.g., number of technologies)
def fe_trials(X):
    """

    :param X: the input usage features (a DataFrame)
    :return: a modified version of X (a DataFrame)
    """
    technologies = get_technologies()
    tech_cols = [col for col in X.columns for tech in technologies if tech in col]
    X['n_tech'] = (X[tech_cols] != 0).astype(int).sum(axis=1)
    X['n_tech_std'] = np.std(X[tech_cols], axis=1)
    X['leading_tech'] = list(X[tech_cols].idxmax(axis=1))
    X['a_size_avg/b_size_avg'] = np.where(X['b_size_avg'] == 0, 0, X['a_size_avg'] / X['b_size_avg'])
    X['a_count_avg/b_count_avg'] = np.where(X['b_count_avg'] == 0, 0, X['a_count_avg'] / X['b_count_avg'])
    return X


# - get the trial scoring for each lead (using either external trials model or the training set)
def get_trial_scoring(X, cols_order=[], is_train=True, model=None, disc=None, only_trial_features=True,
                      supervised_model=True):
    """

    :param X: The input features (a DataFrame)
    :param cols_order: the cols models of the pre-trained trials model, relevant only if is_train = False
    :param is_train: a boolean value indicates if train a trials model or using a pretrained model
    :param model: the pretrained trials model
    :param disc: the discritization object (fitted with the scores of the pretrained trials model)
    :param only_trial_features: a boolean value indicates whether to use only trial features
    :param supervised_model: a boolean value indicates whether the pretrained model is supervised
    :return: a modified version of X that includes the trials category; that either:
             1. 'did trial without call-home' for those who did trial but without call-home data
             2. 'did not do trial' for those we did not do trial
             3. 'low / medium / high' for those who did trial with call home. This is the scores (from the model)
                discritized (using disc)
    """
    disc_mapping = {0.0: 'low', 1.0: 'medium', 2.0: 'high'}
    trial_cat = X.trial_cat
    X_trials = X[trial_cat == 1]
    missing_cols = [col for col in X.columns if len(X) != X[col].count()]  # - these are the usage trials columns
    if only_trial_features:
        X_trials = X_trials[missing_cols]
    X_trials = fe_trials(X_trials.copy())
    X_trials_numeric = X_trials.select_dtypes(include=np.number)
    if is_train:
        model = IsolationForest(random_state=0)
        model.fit(X_trials_numeric)
    else:
        cols_to_drop = [col for col in X_trials_numeric.columns if col not in cols_order]
        X_trials_numeric = X_trials_numeric.drop(cols_to_drop, axis=1)
        X_trials_numeric = X_trials_numeric[cols_order]
    if supervised_model:
        trial_scores = model.predict_proba(X_trials_numeric)[:, 1]
    else:
        trial_scores = np.abs(model.score_samples(X_trials_numeric))
    if is_train:
        disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        trial_scores_cat = disc.fit_transform(np.array(trial_scores).reshape(-1, 1)).reshape(-1)
    else:
        trial_scores_cat = disc.transform(np.array(trial_scores).reshape(-1, 1)).reshape(-1)
    trial_scores_mapped = np.array([disc_mapping[i] for i in trial_scores_cat.tolist()])
    trial_cat_feature = trial_cat.astype('object')
    trial_cat_feature[trial_cat == 1] = trial_scores_mapped
    trial_cat_feature[trial_cat == 2] = 'did trial without call-home'
    trial_cat_feature[trial_cat == 3] = 'did not do trial'
    X['trial_cat_feature'] = trial_cat_feature.astype('category')
    X = X.drop('trial_cat', axis=1)
    # - drop missing cols
    X = X.drop(missing_cols, axis=1)
    if is_train:
        return X, model, disc
    else:
        return X


# - not used in this project
def get_trial_scoring_clustering(X, is_train=True, model=None, scaler=None):
    disc_mapping = {0: 'low', 1: 'medium', 2: 'high', 3: 'very high'}
    trial_cat = X.trial_cat
    X_trials = X[trial_cat == 1]
    X_trials = fe_trials(X_trials.copy())
    X_trials_numeric = X_trials.select_dtypes(include=np.number)

    if is_train:
        trial_clusters, model, scaler = add_cluster_feature(X_trials_numeric, is_train=is_train)
    else:
        trial_clusters = add_cluster_feature(X_trials_numeric, is_train=is_train, model=model, scaler=scaler)

    trial_scores_mapped = np.array([disc_mapping[i] for i in trial_clusters.tolist()])
    trial_cat_feature = trial_cat.astype('object')
    trial_cat_feature[trial_cat == 1] = trial_scores_mapped
    trial_cat_feature[trial_cat == 2] = 'did trial without call-home'
    trial_cat_feature[trial_cat == 3] = 'did not do trial'
    X['trial_cat_feature'] = trial_cat_feature.astype('category')
    X = X.drop('trial_cat', axis=1)
    # - drop missing cols
    missing_cols = [col for col in X.columns if len(X) != X[col].count()]
    X = X.drop(missing_cols, axis=1)
    if is_train:
        return X, model, scaler
    else:
        return X


# - get the external trained trials model
def get_trials_model(df_leads, df_trials, supervised_model=True):
    """

    :param df_leads: a df contain all leads use for training / prediction
    :param df_trials: a df contain all trials from 2018
    :param supervised_model: a boolean value indicates whether to use a supervised model or not
    :return: fitted model, discritization object, the cols order of the trained model
    """
    df_trials = df_trials[~df_trials['email'].isin(pd.unique(df_leads['email']))]
    cols_to_drop = [col for col in df_trials.columns if
                    any(col_name in col for col_name in
                        ['.1', '.2', 'request_id', 'domain', 'email', 'trial_start_date'])]
    df_trials = df_trials.drop(cols_to_drop, axis=1)
    X = fe_trials(df_trials.drop('class', axis=1))
    X = X.select_dtypes(include=np.number)
    cols_order = X.columns
    if supervised_model:
        y = df_trials['class']
        model = CatBoostClassifier(cat_features=get_cat_feature_names(X), auto_class_weights='Balanced',
                                   verbose=0, random_state=5, loss_function=FocalLossObjective(), eval_metric="Logloss",
                                   bootstrap_type='Bayesian', rsm=0.1)
        model.fit(X, y)
        trial_scores = model.predict_proba(X)[:, 1]
    else:
        model = IsolationForest(random_state=0)
        model.fit(X)
        trial_scores = np.abs(model.score_samples(X))
    disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    disc.fit_transform(np.array(trial_scores).reshape(-1, 1)).reshape(-1)
    return model, disc, cols_order


def get_url_domain(url):
    x = re.match('(.)+?\.(.)+?\.', url)[0]
    index = len(x)
    only_domain = url[index:]
    if '.com' in only_domain:
        only_domain = 'com'
    return only_domain


def get_main_url(url):
    only_main_url_list = re.match('((.*\/\/)(.)*?)\/', url)
    if only_main_url_list:
        return only_main_url_list[1]
    else:
        return None


def get_amount_of_distinct_main_urls(list_of_urls):
    unique_urls_dict = {}
    if len(list_of_urls) > 0:
        for url in list_of_urls:
            main_url = get_main_url(url)
            if main_url:
                unique_urls_dict[main_url] = 1

        return len(unique_urls_dict)

    else:
        return 0


def get_file_name(path):
    file_name = re.match('(.)*\/((.)*)', path)[2]
    return file_name


def get_file_type(path):
    match = re.compile('((.)*\.)')
    catches = match.findall(path)
    file_type = path.replace(catches[0][0], '') if catches else 'no file type'
    return file_type


def get_version(path):
    match = re.compile('(((\d)+(\.|_|-)*)+)')
    catches = match.findall(path)
    version = max(catches, key=(lambda x: len(x[0])))[0] if catches else np.nan
    return version


def get_version_number(version):
    try:

        big_num = int(version[0])
        small_num = version[1:].replace('-', '').replace('_', '').replace('.', '')
        version_number = float(big_num) + float(small_num) / float((10 ** len(str(small_num))))
    except:
        version_number = 0

    return version_number


def aggregate_df_versioning_analysis(df):
    aggregation_dict = {}
    first_row = True
    prev_row = None
    # curr_row = None
    deployments_counter = 0
    module_to_files_dict = {}
    for index, row in df.iterrows():
        curr_row = row
        week_num = int((curr_row['date'].day - 1) / 7) + 1
        if curr_row['module_name'] in module_to_files_dict:
            module_to_files_dict[curr_row['module_name']].update({curr_row['file_name']: 1})
        else:
            module_to_files_dict[curr_row['module_name']] = {curr_row['file_name']: 1}
        # rows to be skipped: first row, after 28th day row, empty file name row, xml file type row
        if first_row or curr_row['module_name'] == '' or curr_row['date'].day > 28 or curr_row['file_type'] == 'xml':
            prev_row = curr_row
            first_row = False
            continue
        else:
            is_release = 1 if 'release' in curr_row['module_name'].lower() else 0
            is_snapshot = 1 if 'snapshot' in curr_row['module_name'].lower() else 0
            # If this is a whole new file name
            if curr_row['module_name'] != prev_row['module_name']:
                aggregation_dict[curr_row['module_name']] = {week_num: {'is_versioning_up': 0,
                                                                        'is_versioning_stuck': 0,
                                                                        'versioning_up_count': 0,
                                                                        'versioning_stuck_count': 0,
                                                                        'is_release': is_release,
                                                                        'is_snapshot': is_snapshot,
                                                                        'files_per_module': len(module_to_files_dict[
                                                                                                    curr_row[
                                                                                                        'module_name']])}}
            # If its a reoccurring file name
            else:
                # If this week already has its own row
                if week_num in aggregation_dict[curr_row['module_name']]:
                    aggregation_dict[curr_row['module_name']][week_num]['is_release'] = \
                        aggregation_dict[curr_row['module_name']][week_num]['is_release'] + is_release
                    aggregation_dict[curr_row['module_name']][week_num]['is_snapshot'] = \
                        aggregation_dict[curr_row['module_name']][week_num]['is_snapshot'] + is_snapshot
                    aggregation_dict[curr_row['module_name']][week_num]['files_per_module'] = len(
                        module_to_files_dict[curr_row['module_name']])
                    if curr_row['version_num_value'] > prev_row['version_num_value']:
                        aggregation_dict[curr_row['module_name']][week_num]['is_versioning_up'] = 1
                        aggregation_dict[curr_row['module_name']][week_num]['versioning_up_count'] += 1
                    elif curr_row['version_num_value'] == prev_row['version_num_value']:
                        aggregation_dict[curr_row['module_name']][week_num]['is_versioning_stuck'] = 1
                        aggregation_dict[curr_row['module_name']][week_num]['versioning_stuck_count'] += 1
                # If this is a new week for this existing file
                else:
                    aggregation_dict[curr_row['module_name']][week_num] = {'is_versioning_up': 0,
                                                                           'is_versioning_stuck': 0,
                                                                           'versioning_up_count': 0,
                                                                           'versioning_stuck_count': 0,
                                                                           'is_release': is_release,
                                                                           'is_snapshot': is_snapshot,
                                                                           'files_per_module': len(module_to_files_dict[
                                                                                                       curr_row[
                                                                                                           'module_name']])}
        prev_row = curr_row

    payload_for_df = []
    for file in aggregation_dict:
        for week_num in aggregation_dict[file]:
            temp_dict = {'module_name': file,
                         'week_num': week_num,
                         'versioning_up_count': aggregation_dict[file][week_num]['versioning_up_count'],
                         'versioning_stuck_count': aggregation_dict[file][week_num]['versioning_stuck_count'],
                         'is_release': aggregation_dict[file][week_num]['is_release'],
                         'is_snapshot': aggregation_dict[file][week_num]['is_snapshot'],
                         'files_per_module': aggregation_dict[file][week_num]['files_per_module']}
            payload_for_df.append(temp_dict)
    return pd.DataFrame(payload_for_df)


def aggregate_df_per_module(df):
    prev_row = None
    first_row = True
    versioning_up_sum, versioning_stuck_sum, is_snapshot_sum, is_release_sum, files_per_module_sum, week_count = 0, 0, 0, 0, 0, 1
    aggregate_list = []
    for index, row in df.iterrows():
        curr_row = row
        if first_row:
            print('im here')
            prev_row = curr_row
            first_row = False
            continue
        else:
            curr_row = row
            print(prev_row['module_name'])
            print(curr_row['module_name'])
            if curr_row['module_name'] != prev_row['module_name']:

                module_name = curr_row['module_name']
                temp_dict = {module_name: {'versioning_up_avg': versioning_up_sum / week_count,
                                           'versioning_stuck_avg': versioning_stuck_sum / week_count,
                                           'is_release_avg': is_release_sum / week_count,
                                           'is_snapshot_avg': is_snapshot_sum / week_count,
                                           'files_per_module_avg': files_per_module_sum / week_count}}
                aggregate_list.append(temp_dict)
                versioning_up_sum = curr_row['versioning_up_count']
                versioning_stuck_sum = curr_row['versioning_stuck_count']
                is_release_sum = curr_row['is_release']
                is_snapshot_sum = curr_row['is_snapshot']
                files_per_module_sum = curr_row['files_per_module']
                week_count = 1
            else:
                versioning_up_sum += curr_row['versioning_up_count']
                versioning_stuck_sum += curr_row['versioning_stuck_count']
                is_release_sum += curr_row['is_release']
                is_snapshot_sum += curr_row['is_snapshot']
                files_per_module_sum += curr_row['files_per_module']
                week_count += 1

        final_df = pd.DataFrame(aggregate_list)
        return final_df


def remove_outliers(df_, keep_csat=False):
    pre = df_.shape[0]
    z_arr = np.zeros(df_.shape)
    cols = df_.columns.drop('csat_score')
    for idx, col in enumerate(cols):
        z_arr[:, idx] = np.abs(stats.zscore(df_[col]))

    if keep_csat:
        no_outliers_idx = df_[(np.all(z_arr < 3, axis=1)) | ~(pd.isnull(df_['csat_score']))].index
        df_ = df_[(np.all(z_arr < 3, axis=1)) | ~(pd.isnull(df_['csat_score']))]
    else:
        no_outliers_idx = df_[np.all(z_arr < 3, axis=1)].index
        df_ = df_[np.all(z_arr < 3, axis=1)]

    post = df_.shape[0]
    amount = pre - post
    print('Amount of removed outliers: {} instances'.format(amount))
    return df_, no_outliers_idx


def get_points_size(data, scale=2):
    x, y = data.iloc[:, 0], data.iloc[:, 1]
    c = Counter(zip(x, y))
    return [scale * c[(xx, yy)] for xx, yy in zip(x, y)]


def get_sha1_id(summary):
    sha1_obj = hashlib.sha1()
    sha1_obj.update(summary.encode('utf-8'))
    return sha1_obj.hexdigest()


def get_growth_features(feature_name, df, return_only_q1_growth=True, normalize_growth=True, monthly=True):
    feature_cols = [col for col in df.columns if col.startswith(feature_name)][::-1]
    feature_cols_monthly = [col for col in feature_cols if '.1' not in col and '.2' not in col]
    feature_cols_quarter = [col for col in feature_cols if '.3' not in col and '.4' not in col]
    growth_feature_monthly = np.mean((np.diff(df[feature_cols_monthly]) / df[feature_cols_monthly[:2]]).fillna(0).replace(np.inf, 1), axis=1)
    growth_feature_quarter = np.mean((np.diff(df[feature_cols_quarter]) / df[feature_cols_quarter[:2]]).fillna(0).replace(np.inf, 1), axis=1)
    if normalize_growth:
        if monthly:
            df_fg = pd.DataFrame((np.diff(df[feature_cols_monthly]) / df[feature_cols_monthly[:2]]).fillna(0).replace(np.inf, 1),
                                 columns=['q2_' + feature_name + '_growth', 'q1_' + feature_name + '_growth'])
        else:
            df_fg = pd.DataFrame((np.diff(df[growth_feature_quarter]) / df[growth_feature_quarter[:2]]).fillna(0).replace(np.inf, 1),
                                 columns=['q2_' + feature_name + '_growth', 'q1_' + feature_name + '_growth'])
    else:
        if monthly:
            df_fg = pd.DataFrame(np.diff(df[feature_cols_monthly]),
                                 columns=['q2_' + feature_name + '_growth', 'q1_' + feature_name + '_growth'])
        else:
            df_fg = pd.DataFrame(np.diff(df[growth_feature_quarter]),
                                 columns=['q2_' + feature_name + '_growth', 'q1_' + feature_name + '_growth'])
    df_fg['binary_' + feature_name] = np.where(df_fg['q1_' + feature_name + '_growth'] < df_fg['q2_' + feature_name + '_growth'], 1, 0)
    if return_only_q1_growth:
        df_fg = df_fg.drop('q2_' + feature_name + '_growth', axis=1)
    return growth_feature_monthly, growth_feature_quarter,  df_fg