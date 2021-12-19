import pandas as pd
import numpy as np

from names_mapping import names_mapping
from utils.general_utils import load_data_old
from utils.model_utils import cv_evaluation
from utils.preprocessing_utils import consolidate_opps, pro_upsell_preprocess
from utils.report_utils import create_output_table, binning_features

model = 'cbc'

df_train = load_data_old('fit.sql')
df_train = consolidate_opps(df_train.copy())
df_train = pro_upsell_preprocess(df_train.copy())
cols_to_drop = [col for col in df_train.columns if
                'period_range' in col or 'relevant_date' in col or 'account_id' in col
                or 'class' in col or 'has_won' in col]
X_train, y_train = df_train.drop(cols_to_drop, axis=1).fillna(-1), df_train['class']
score, clf = cv_evaluation(model=model, X=X_train, y=y_train)

df_test = load_data_old('predict.sql')
df_test = pro_upsell_preprocess(df_test.copy())
cols_to_drop = [col for col in df_test.columns if
                'period_range' in col or 'relevant_date' in col or 'account_id' in col
                or 'class' in col or 'has_won' in col]
X_test = df_test.drop(cols_to_drop, axis=1).fillna(-1)
X_test_disc = binning_features(X_test.copy())

res_df = pd.DataFrame()
res_df['account_id'] = df_test['account_id']
res_df['prob'] = clf.predict_proba(X_test)[:, 1]
threshold_to_high = res_df['prob'].quantile(0.85)
threshold_to_medium = res_df['prob'].quantile(0.7)
res_df['rating'] = np.where(res_df['prob'] < threshold_to_high,
                            np.where(res_df['prob'] < threshold_to_medium, 'low', 'medium'), 'high')

output_df = create_output_table(res_df, clf, X_test, 5, X_test_disc)
mapping = names_mapping()
output_df = output_df.replace({"feature": mapping})