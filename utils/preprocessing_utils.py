import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy

from utils.fe_utils import fe_leads, add_anomaly_score_feature, get_trial_scoring, get_trial_scoring_clustering, \
    get_trials_model
from utils.general_utils import load_data_old, get_technologies

def Boruta_feature_selection(X_train, y_train, X_test, random_state=0, alpha=0.05):
    X_train_labeld = X_train[y_train != -1]
    y_train_labeld = y_train[y_train != -1]
    X_train_numeric = X_train_labeld.select_dtypes(include=np.number)
    # define Boruta feature selection method
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=random_state)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=random_state, alpha=alpha)

    # find all relevant features
    feat_selector.fit(X_train_numeric.values, y_train_labeld)
    cols_to_drop = [col for ind, col in enumerate(X_train_numeric.columns) if
                    not ((feat_selector.support_[ind]) | (feat_selector.support_weak_[ind]))]
    X_train_selected, X_test_selected = X_train.drop(cols_to_drop, axis=1), X_test.drop(cols_to_drop, axis=1)
    return X_train_selected, X_test_selected


def drop_by_correlation(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return cols_to_drop

def drop_by_correlation(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    cols_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return cols_to_drop

def preprocess_data(df=pd.DataFrame(), query_name=''):
    if len(df) == 0:
        df = load_data_old(query_name)
    #  clean the reason fields
    df.reason = df.apply(lambda row: row.reason
    if re.match("(.*?)(\.)(.*?)( action Change)", str(row.reason).lower()) == None
    else re.match("(.*?)(\.)(.*?)( action Change)", str(row.reason).lower()).group(3), axis=1)

    # make a list of activities names
    activities = df.reason.value_counts()
    activities = activities[activities.index != 'score']
    activities = list(activities.index)
    # create value of 1 and 0 per activity
    for activity in activities:
        df[activity] = df.apply(lambda row: 1
        if row['reason'] == activity else 0, axis=1)
    # turn to aggregated table
    agg_scoring_Data = df.groupby(['lead_full_id'], as_index=False)[activities].apply(lambda x: x.astype(int).sum())
    df = df[[col for col in df.columns if col == 'lead_full_id' or col not in agg_scoring_Data.columns]]

    indices = []
    for lead_id in pd.unique(df['lead_full_id']):
        df_temp = df[df['lead_full_id'] == lead_id]
        indices.append(df_temp['trial_number'].idxmax())

    df_max = df.loc[indices, :]

    technologies = get_technologies()
    tech_cols = [col for col in df_max.columns for tech in technologies if tech in col]
    df_max['n_tech'] = (df_max[tech_cols] != 0).astype(int).sum(axis=1)
    df_max['n_tech_std'] = np.std(df_max[tech_cols], axis=1)
    df_max['leading_tech'] = list(df_max[tech_cols].idxmax(axis=1))
    df_max['a_size_avg/b_size_avg'] = np.where(df_max['b_size_avg'] == 0, 0,
                                               df_max['a_size_avg'] / df_max['b_size_avg'])
    df_max['a_count_avg/b_count_avg'] = np.where(df_max['b_count_avg'] == 0, 0,
                                                 df_max['a_count_avg'] / df_max['b_count_avg'])

    full_df = df_max.merge(agg_scoring_Data, on='lead_full_id', how='inner', copy=False)
    return full_df


def load_train_test_old(query_name_train, query_name_test, label_name, oot=True):
    df_train = preprocess_data(query_name_train)
    cols_to_remove = [col for col in df_train.columns if
                      df_train.dtypes[col] != 'object' and np.std(df_train[col]) == 0]
    df_train = df_train.drop(cols_to_remove, axis=1)
    df_train = df_train.reset_index(drop=True)
    cols_to_drop = ['email', 'request_id', 'trial_start_date', 'request_id1', 'lead_full_id', 'country',
                    'lead_full_id.1', 'reason', 'activity_date', 'type', 'opp_label', 'purchase_label', 'opportunity']
    X_train, y_train = df_train.drop(cols_to_drop, axis=1), df_train[label_name]

    df_test = preprocess_data(query_name_test)
    df_test = df_test.reset_index(drop=True)
    X_test = df_test[[col for col in df_test.columns if col in X_train.columns]]
    cols_to_keep = list(set(list(X_train.columns)) & set(list(X_test.columns)))
    X_train, X_test = X_train[cols_to_keep], X_test[cols_to_keep]
    X_test = X_test[X_train.columns]
    test_requests = df_test['request_id']

    if oot:
        y_test = df_test[label_name]
        return X_train, y_train, X_test, y_test, test_requests
    else:
        return X_train, y_train, X_test, test_requests


def flatten_leads_data(df, order_df=False, min_to_keep=10, is_train=True, is_oot=False):
    df.reason = df.apply(lambda row: row.reason
    if re.match("(.*?)(\.)(.*?)( action change)", str(row.reason).lower()) == None
    else re.match("(.*?)(\.)(.*?)( action change)", str(row.reason).lower()).group(3), axis=1)

    if is_train:
        activities = df.reason.value_counts()[df.reason.value_counts() > min_to_keep]
    else:
        activities = df.reason.value_counts()
    activities = activities[activities.index != 'score']
    activities = list(activities.index)

    # create value of 1 and 0 per activity
    for activity in activities:
        df[activity] = df.apply(lambda row: 1 if row['reason'] == activity else 0, axis=1)

    # turn to aggregated table
    agg_scoring_data = df.groupby(['lead_full_id'], as_index=False)[activities].apply(lambda x: x.astype(int).sum())
    df = df[
        [col for col in df.columns if col == 'lead_full_id' or col not in agg_scoring_data.columns]]

    join_with = df.drop(['reason', 'activity_date'], axis=1).drop_duplicates()
    full_df = agg_scoring_data.merge(join_with, on='lead_full_id', how='inner', copy=False)
    full_df = full_df.reset_index(drop=True)

    # --- transform to pd.categorical - speeding up
    for col in full_df.columns:
        if full_df.dtypes[col] == 'object':
            full_df[col] = full_df[col].astype('category')
    full_df = full_df.drop_duplicates()

    closed_list_reasons = get_scoring_reasons()
    cols_to_drop = [col for col in activities if col not in closed_list_reasons]
    full_df = full_df.drop(cols_to_drop, axis=1)
    # - change these columns names to identify them as a group
    names_mapping = {activity: 'CS-' + activity for activity in activities}
    full_df = full_df.rename(columns=names_mapping)

    # - drop zero variance columns
    zero_variance_cols_numeric = [col for col in full_df.select_dtypes(include=np.number) if np.std(full_df[col]) == 0]
    zero_variance_cols_categoric = [col for col in full_df.select_dtypes(exclude=np.number)
                                    if len(pd.unique(full_df[col])) == 1 and col != 'relevant_date']
    zero_variance_cols = zero_variance_cols_numeric + zero_variance_cols_categoric
    full_df = full_df.drop(zero_variance_cols, axis=1)

    if order_df:
        full_df = full_df.sort_values(by='relevant_date').reset_index(drop=True)

    if is_train or is_oot:
        X, y = full_df.drop(['lead_full_id', 'class'], axis=1), full_df['class']
        return X, y
    else:
        X = full_df.drop('lead_full_id', axis=1)
        return X


def flatten_journeys_data(df, order_df=False):
    df.reason = df.apply(lambda row: row.reason
    if re.match("(.*?)(\.)(.*?)( action change)", str(row.reason).lower()) == None
    else re.match("(.*?)(\.)(.*?)( action change)", str(row.reason).lower()).group(3), axis=1)

    activities = df.reason.value_counts()[df.reason.value_counts() > 10]
    activities = activities[activities.index != 'score']
    activities = list(activities.index)

    # create value of 1 and 0 per activity
    for activity in activities:
        df[activity] = df.apply(lambda row: 1 if row['reason'] == activity else 0, axis=1)

    # turn to aggregated table
    agg_scoring_data = df.groupby(['lead_id', 'relevant_date'], as_index=False)[activities].apply(lambda x: x.astype(int).sum())
    df = df[[col for col in df.columns if col in ['lead_id', 'relevant_date'] or col not in agg_scoring_data.columns]]

    join_with = df.drop(['reason', 'activity_date'], axis=1).dropna().drop_duplicates()
    full_df = agg_scoring_data.merge(join_with, on=['lead_id', 'relevant_date'], how='inner', copy=False)
    full_df = full_df.reset_index(drop=True)

    # --- transform to pd.categorical - speeding up
    for col in full_df.columns:
        if full_df.dtypes[col] == 'object':
            full_df[col] = full_df[col].astype('category')
    full_df = full_df.drop_duplicates()
    # - change these columns names to identify them as a group
    names_mapping = {activity: 'CS-' + activity for activity in activities}
    full_df = full_df.rename(columns=names_mapping)

    # - drop zero variance columns
    zero_variance_cols = [col for col in full_df.select_dtypes(include=np.number) if np.std(full_df[col]) == 0]
    full_df = full_df.drop(zero_variance_cols, axis=1)

    if order_df:
        full_df = full_df.sort_values(by='relevant_date').reset_index(drop=True)
    X, y = full_df.drop(['lead_id', 'class'], axis=1), full_df['class']
    return X, y


def load_train_test(df_train, df_test, is_oot=True, clustering=False, external_trials_model=False, df_trials=None,
                    supervised_trials_model=True, add_anomaly_score=False):
    cols_to_drop = [col for col in df_train.columns if
                    any(col_name in col for col_name in ['.1', '.2', 'request_id', 'domain'])]
    df_train = df_train.drop(cols_to_drop, axis=1)
    df_test = df_test.drop(cols_to_drop, axis=1)
    X_train, y_train = flatten_leads_data(df_train, is_train=True)
    if is_oot:
        X_test, y_test = flatten_leads_data(df_test, is_train=False, is_oot=is_oot)
    else:
        X_test = flatten_leads_data(df_test, is_train=False, is_oot=is_oot)
    train_mql_dates = X_train['mql_date']
    X_train = X_train.drop('mql_date', axis=1)
    if is_oot:
        X_test = X_test.drop('mql_date', axis=1)
    X_train = fe_leads(X_train.copy())
    test_emails = X_test['email']
    X_test = fe_leads(X_test.copy())

    cols_to_keep = list(set(list(X_train.columns)) & set(list(X_test.columns)))
    X_train, X_test = X_train[cols_to_keep], X_test[cols_to_keep]
    X_test = X_test[X_train.columns]

    # --- add anomaly score feature
    if add_anomaly_score:
        X_train, X_test = add_anomaly_score_feature(X_train.copy(), X_test.copy())

    # --- add trial feature
    if not clustering:
        if not external_trials_model:
            X_train, model, disc = get_trial_scoring(X=X_train.copy(), is_train=True, only_trial_features=True)
            X_test = get_trial_scoring(X=X_test.copy(), is_train=False, model=model, disc=disc, only_trial_features=True)
        else:
            dff = pd.concat([df_train, df_test], axis=0)
            model, disc, cols_order = get_trials_model(df_leads=dff, df_trials=df_trials,
                                                       supervised_model=supervised_trials_model)
            X_train = get_trial_scoring(X=X_train.copy(), cols_order=cols_order, is_train=False, model=model,
                                        disc=disc, only_trial_features=True, supervised_model=supervised_trials_model)
            X_test = get_trial_scoring(X=X_test.copy(), cols_order=cols_order, is_train=False, model=model,
                                       disc=disc, only_trial_features=True, supervised_model=supervised_trials_model)
    else:
        X_train, model, scaler = get_trial_scoring_clustering(X=X_train.copy(), is_train=True)
        X_test = get_trial_scoring_clustering(X=X_test.copy(), is_train=False, model=model, scaler=scaler)
    if is_oot:
        return X_train, y_train, X_test, y_test, test_emails, train_mql_dates
    else:
        return X_train, y_train, X_test, test_emails, train_mql_dates


def feature_selection(model, X_train, X_test):
    selector = SelectFromModel(model, prefit=True, norm_order=1, threshold='median')
    feature_idx = selector.get_support()
    feature_names = X_train.columns[feature_idx]
    X_train_selected = selector.transform(X_train)
    X_train_selected = pd.DataFrame(X_train_selected, columns=feature_names)
    return X_train[X_train_selected.columns], X_test[X_train_selected.columns]


def get_scoring_reasons():
    return ['form - fills out web contact/buy now/get quote form',
            'academy - sign up',
            'academy - course completion ',
            'webinar generic - on demand',
            'on24 resource 2 viewed',
            'on24 resource 3 viewed',
            'social click link',
            'form - fills out trial form (tier 2 not targeted company)',
            '03 - webinar - attended',
            'web - visits undesirable web pages',
            'form - fills out partners trial or mp form',
            'form - fills out contact/buy now/get quote form',
            'webinar cloud - meeting',
            'form - fills out social linkedin',
            '04 - webinar - on demand',
            'website gated content - generic',
            'form - fills out trial enterprise form',
            'form - fills out social facebook',
            'content syndication - ziff davis generic - multi',
            'form - fills out marketo - contact us demo after webinar',
            'free tier from event cta',
            'webinar cloud - no show',
            'form - fills out trial form',
            'web - visits key web pages',
            'webinar cloud - attended',
            'form - fills out trial form (tier 2 targeted company)',
            '02 - webinar - no show',
            'drift conversation',
            'email - clicks link in email',
            'form - fills out marketo - on-demand webinar',
            'webinar cloud - on demand',
            'on24 q&a participant',
            'free tier pipelines checkbox',
            'form - fills out web - website gated content',
            'form - fills out orbitera form',
            'webinar generic - no show',
            'web - visits multiple web pages in 1 day',
            'trial dropout - visits the website',
            'webinar generic - meeting',
            'ziff davis - more than 1',
            'event - asked for a meeting',
            'on24 resource 1 viewed',
            'content syndication - ziff davis generic',
            'webinar generic - attended',
            'free tier commercial',
            'free tier 25%+ usage',
            'social share post',
            '01 - webinar - registered',
            'social like']