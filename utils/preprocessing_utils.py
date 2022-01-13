import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy

from utils.fe_utils import get_growth_features
from utils.general_utils import get_cat_feature_names


def pro_upsell_preprocess(df):
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
    return df


def consolidate_opps(df_train):
    has_won = df_train.groupby('account_id', as_index=False).sum('class').loc[:, ['account_id', 'class']]
    has_won['has_won'] = has_won['class'].apply(lambda x: True if x > 0 else False)
    has_won.drop('class', axis=1, inplace=True)
    new_df = df_train.merge(has_won[['account_id', 'has_won']], on='account_id')
    df_did_win, df_did_not_win = new_df[new_df['has_won']], new_df[~new_df['has_won']]
    df_did_win = df_did_win.merge(df_did_win.groupby('account_id', as_index=False).agg({'relevant_date': 'min'}),
                                  on=['account_id', 'relevant_date'])
    df_did_not_win = df_did_not_win.groupby('account_id', as_index=False).sample(n=1, random_state=2)
    df_train = pd.concat([df_did_win, df_did_not_win])
    df_train = df_train.sample(frac=1, random_state=2).reset_index(drop=True)
    return df_train


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


def feature_selection(model, X_train, X_test):
    selector = SelectFromModel(model, prefit=True, norm_order=1, threshold='median')
    feature_idx = selector.get_support()
    feature_names = X_train.columns[feature_idx]
    X_train_selected = selector.transform(X_train)
    X_train_selected = pd.DataFrame(X_train_selected, columns=feature_names)
    return X_train[X_train_selected.columns], X_test[X_train_selected.columns]


