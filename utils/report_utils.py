import re

import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import KBinsDiscretizer


def create_output_table(result_df, model, X_test, n_largest=5, X_test_disc=None):
    """

    :param result_df: a df contains 3 columns: account, class_pred, and prob
    :param model: a decision-tree-based model (e.g., random forest, catboost)
    :param X_test: the test set (df)
    :param n_largest: how many features to present (both negative and positive effect)
    :return: a df in shape (X_test.shape[0]*n_largest*2, 6), where each row contains a feature name and value
    that effect (either positive or negative) the output of this account. Each account has n_largest*2 rows, and
    class pred and prob are the same along these rows.
    """
    explainer = shap.TreeExplainer(model)
    shap_mat = explainer.shap_values(X_test)

    if len(np.array(shap_mat).shape) == 3:
        shap_mat = shap_mat[1]

    shap_df = pd.DataFrame(shap_mat, columns=X_test.columns)
    cols_to_drop = ['artifacts/binaries_size', 'artifacts/binaries_count', 'did_sessions_and_cases_last_year',
                    'territory', 'replys_to_sent', 'count_pro', 'days_from_indexed_repos_change',
                    'n_replys', 'n_calls', 'n_task_xray', 'engineers', 'devops_engineers', 'developers']
    cols_to_drop += [col for col in X_test.columns if '/seniority' in col]
    shap_df = shap_df.drop(cols_to_drop, axis=1, errors='ignore')
    X_test_disc = X_test_disc.reset_index(drop=True)
    shap_df = shap_df.where(~X_test_disc[shap_df.columns].isin(['other', 'unknown']), other=np.nan)
    shap_df = shap_df.where(X_test_disc != -1, other=np.nan)

    full_shap_df = pd.DataFrame(columns=X_test.columns)
    for col in X_test.columns:
        if col in shap_df.columns:
            full_shap_df[col] = shap_df[col]
        else:
            full_shap_df[col] = np.zeros(X_test.shape[0])

    order = np.argsort(-shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(-shap_df[shap_df > 0].values, axis=1)[:, :n_largest]
    largest_positive = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' positive' for i in range(n_largest)],
                                    index=shap_df.index)
    largest_positive[order != order1] = np.nan
    order = np.argsort(shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(shap_df[shap_df < 0].values, axis=1)[:, :n_largest]
    largest_negative = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' negative' for i in range(n_largest)],
                                    index=shap_df.index)
    largest_negative[order != order1] = np.nan
    result = pd.concat([largest_positive, largest_negative], axis=1)


    X_test_values = X_test_disc if X_test_disc is not None else X_test
    quantiles_lower = {col: np.quantile(X_test_values[col], 0.33) for col in X_test_values.columns if X_test_values.dtypes[col] in ('int', 'float')}
    quantiles_upper = {col: np.quantile(X_test_values[col], 0.66) for col in X_test_values.columns if X_test_values.dtypes[col] in ('int', 'float')}

    shap_lists = pd.Series(result.values.tolist())
    mask_array = pd.get_dummies(shap_lists.apply(pd.Series).stack()).sum(level=0)
    df_mask = pd.DataFrame(columns=X_test.columns)
    for col in X_test.columns:
        if col in mask_array.columns:
            df_mask[col] = mask_array[col]
    df_mask = df_mask[X_test.columns]
    df_filtered = X_test_values.where(df_mask > 0, other=np.nan)
    for col in df_filtered.columns:
        if df_filtered.dtypes[col] in ('int', 'float'):
            df_filtered[col] = np.where(pd.isnull(df_filtered[col]), np.nan,
                                        np.where(df_filtered[col] < quantiles_lower[col],
                                                 df_filtered[col].astype(str) + ' (low for Pro)',
                                                 np.where(df_filtered[col] > quantiles_upper[col],
                                                          df_filtered[col].astype(str) + ' (high for Pro)',
                                                          df_filtered[col].astype(str) + ' (around average for Pro)')))

    output_with_features = pd.concat([df_filtered, result_df], axis=1)
    output_with_features = pd.melt(output_with_features, id_vars=result_df.columns).dropna()
    output_with_shap = full_shap_df.where(df_mask > 0, other=np.nan)
    output_with_shap['account_id'] = result_df['account_id']
    output_with_shap = pd.melt(output_with_shap, id_vars=['account_id']).dropna()
    output_df = output_with_features.merge(output_with_shap, on=['account_id', 'variable'])
    output_df.columns = list(result_df.columns) + ['feature', 'feature_value', 'shap_importance']
    output_df['relative_value'] = np.nan
    output_df = output_df[['account_id', 'prob', 'rating', 'feature', 'feature_value', 'relative_value', 'shap_importance']]
    return output_df


def binning_features(X_test):
    cols_to_disc = [col for col in X_test.columns if 'growth' in col]
    disc_mapping = {0: 'very low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very high'}
    company_age_mapping = {0: 'very young', 1: 'young', 2: 'mature', 3: 'old', 4: 'very old'}
    X_test_disc = X_test.copy()

    for col in cols_to_disc:
        disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        if len(pd.unique(disc.fit_transform(np.array(X_test_disc.loc[~pd.isnull(X_test_disc[col]), col]).reshape(-1, 1)).reshape(-1))) == 5:
            X_test_disc.at[~pd.isnull(X_test_disc[col]), col] = disc.fit_transform(np.array(X_test_disc.loc[~pd.isnull(X_test_disc[col]), col]).reshape(-1, 1))
        else:
            disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            X_test_disc.at[~pd.isnull(X_test_disc[col]), col] = disc.fit_transform(np.array(X_test_disc[col]).reshape(-1, 1))
        X_test_disc = X_test_disc.replace({col: disc_mapping})

    if 'company_age' in list(X_test.columns):
        company_age_mapping = {'(0.0)': '(very young)', '(1.0)': '(young)', '(2.0)': '(mature)', '(3.0)': '(old)', '(4.0)': '(very old)'}
        disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        not_missing_inds = X_test_disc.index[X_test_disc.company_age != -1].tolist()
        X_test_disc.at[not_missing_inds, 'company_age'] = \
            X_test_disc.loc[not_missing_inds, 'company_age'].astype(str) + ' (' + \
            pd.Series(disc.fit_transform(np.array(X_test_disc.loc[not_missing_inds, 'company_age']).reshape(-1, 1)).reshape(-1),
                      index=not_missing_inds).astype(str) + ')'
        for key in company_age_mapping.keys():
            X_test_disc.at[not_missing_inds, 'company_age'] = X_test_disc.loc[not_missing_inds, 'company_age'] \
                .apply(lambda x: re.sub(re.escape(key), company_age_mapping[key], x))
        X_test_disc.at[X_test_disc.company_age == -1, 'company_age'] = 'unknown'
    return X_test_disc

