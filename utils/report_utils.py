import pandas as pd
import numpy as np
import shap
from tqdm import tqdm


def create_output_table(result_df, model, X_test, n_largest=5, X_test_disc=None):
    """

    :param result_df: a df contains 3 columns: account, class_pred, and prob
    :param model: a decision-tree-based model (e.g., random forest, catboost)
    :param X_test: the tests set (df)
    :param n_largest: how many features to present (both negative and positive effect)
    :return: a df in shape (X_test.shape[0]*n_largest*2, 6), where each row contains a feature name and value
    that effect (either positive or negative) the output of this account. Each account has n_largest*2 rows, and
    class pred and prob are the same along these rows.
    """
    explainer = shap.TreeExplainer(model)
    shap_mat = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_mat, columns=X_test.columns)
    cols_to_drop = ['n_reasons', 'clear_bit_cat']
    shap_df = shap_df.drop(cols_to_drop, axis=1)
    order = np.argsort(-shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(-shap_df[shap_df > 0].values, axis=1)[:, :n_largest]
    largest_positive = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' positive' for i in range(n_largest)],
                                    index=shap_df.index)
    largest_positive[order != order1] = ''
    order = np.argsort(shap_df.values, axis=1)[:, :n_largest]
    order1 = np.argsort(shap_df[shap_df < 0].values, axis=1)[:, :n_largest]
    largest_negative = pd.DataFrame(shap_df.columns[order],
                                    columns=[str(i + 1) + ' negative' for i in range(n_largest)],
                                    index=shap_df.index)
    largest_negative[order != order1] = ''
    result = pd.concat([largest_positive, largest_negative], axis=1)
    output_df = pd.DataFrame(columns=list(result_df.columns) + ['feature', 'feature_value', 'shap_importance'])
    ind = 0

    X_test_values = X_test_disc if X_test_disc is not None else X_test
    quantiles_lower = {col: np.quantile(X_test_values[col], 0.33) for col in X_test_values.columns if X_test_values.dtypes[col] in ('int', 'float')}
    quantiles_upper = {col: np.quantile(X_test_values[col], 0.66) for col in X_test_values.columns if X_test_values.dtypes[col] in ('int', 'float')}
    for i in tqdm(range(result_df.shape[0])):
        melted_features = list(result_df.loc[i, :])
        for j in range(result.shape[1]):
            curr_col = result.loc[i, result.columns[j]]
            curr_val = X_test_values.loc[i, curr_col]
            if curr_col != '':
                if not X_test_values.dtypes[curr_col] in ('int', 'float'):
                    vals = melted_features + [curr_col, curr_val, shap_df.loc[i, curr_col]]
                else:
                    if curr_val < quantiles_lower[curr_col]:
                        string_to_add = ' (low comparing to other leads)'
                    elif curr_val > quantiles_upper[curr_col]:
                        string_to_add = ' (high comparing to other leads)'
                    elif quantiles_lower[curr_col] <= curr_val <= quantiles_upper[curr_col]:
                        string_to_add = ' (around the average)'
                    vals = melted_features + [curr_col, str(curr_val) + string_to_add, shap_df.loc[i, curr_col]]
                output_df.loc[ind] = vals
                ind += 1
    return output_df


def create_output_table_fast(result_df, model, X_test, n_largest=5, X_test_disc=None):
    """

    :param result_df: a df contains 3 columns: account, class_pred, and prob
    :param model: a decision-tree-based model (e.g., random forest, catboost)
    :param X_test: the tests set (df)
    :param n_largest: how many features to present (both negative and positive effect)
    :return: a df in shape (X_test.shape[0]*n_largest*2, 6), where each row contains a feature name and value
    that effect (either positive or negative) the output of this account. Each account has n_largest*2 rows, and
    class pred and prob are the same along these rows.
    """
    explainer = shap.TreeExplainer(model)
    shap_mat = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_mat, columns=X_test.columns)
    cols_to_drop = ['n_reasons', 'clear_bit_cat', 'n_reasons / time to mql', 'n_attempts / time in mql',
                    'n_engagements / time in mql', 'total_other_form_submissions_event',
                    'total_other_website_engagements_event']
    shap_df = shap_df.drop(cols_to_drop, axis=1)
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
    df_filtered = X_test_values.where(df_mask == 1, other=np.nan)
    for col in df_filtered.columns:
        if df_filtered.dtypes[col] in ('int', 'float'):
            df_filtered[col] = df_filtered[col].apply(lambda x: np.nan if np.isnan(x)
                                                      else (str(x) + ' (low comparing to other leads)' if x < quantiles_lower[col]
                                                      else (str(x) + ' (high comparing to other leads)' if x > quantiles_upper[col]
                                                      else str(x) + ' (around the average)')))

    output_with_features = pd.concat([df_filtered, result_df], axis=1)
    output_with_features = pd.melt(output_with_features, id_vars=result_df.columns).dropna()
    output_with_shap = full_shap_df.where(df_mask == 1, np.nan)
    output_with_shap['email'] = result_df['email']
    output_with_shap = pd.melt(output_with_shap, id_vars=['email']).dropna()
    output_df = output_with_features.merge(output_with_shap, on=['email', 'variable'])
    output_df.columns = list(result_df.columns) + ['feature', 'feature_value', 'shap_importance']
    return output_df


def get_level_of_effect(output_df):
    output_df['level_of_effect'] = ''
    for i in tqdm(range(output_df.shape[0])):
        output_df_cur_feature = output_df[output_df.feature == output_df.loc[i, 'feature']]
        if output_df.loc[i, 'shap_importance'] > 0:
            median_positive = np.median(output_df_cur_feature.loc[output_df_cur_feature['shap_importance'] > 0, 'shap_importance'])
            if output_df.loc[i, 'shap_importance'] > median_positive:
                output_df.at[i, 'level_of_effect'] = 'strong'
            else:
                output_df.at[i, 'level_of_effect'] = 'weak'
        else:
            median_negative = np.median(output_df_cur_feature.loc[output_df_cur_feature['shap_importance'] <= 0, 'shap_importance'])
            if output_df.loc[i, 'shap_importance'] < median_negative:
                output_df.at[i, 'level_of_effect'] = 'strong'
            else:
                output_df.at[i, 'level_of_effect'] = 'weak'
    return output_df
