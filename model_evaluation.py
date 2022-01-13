import pandas as pd
import numpy as np

from utils.general_utils import load_data_old
from utils.model_utils import cv_evaluation
from utils.preprocessing_utils import consolidate_opps, pro_upsell_preprocess
from utils.plot_utils import Evaluation

model = 'cbc'
date_to_cut = '2021-01-01'

df = load_data_old('fit.sql')
df['relevant_date'] = pd.to_datetime(df.relevant_date).astype('datetime64[ns]')

df_train, df_test = df[df.relevant_date <= date_to_cut], df[df.relevant_date > date_to_cut]
df_train = consolidate_opps(df_train.copy())
df_test = consolidate_opps(df_test.copy())
#df = consolidate_opps(df.copy())
df_train = pro_upsell_preprocess(df_train.copy())
df_test = pro_upsell_preprocess(df_test.copy())

cols_to_drop = [col for col in df_train.columns if
                'period_range' in col or 'relevant_date' in col or 'account_id' in col
                or 'class' in col or 'has_won' in col]
X_train, y_train = df_train.drop(cols_to_drop, axis=1).fillna(-1), df_train['class']
score, clf = cv_evaluation(model=model, X=X_train, y=y_train)
X_test, y_test = df_test.drop(cols_to_drop, axis=1).fillna(-1), df_test['class']

# --- evaluation
# - AP
y_scores = clf.predict_proba(X_test)[:, 1]
Evaluation().plot_precision_recall_test(y_test, y_scores)

threshold = np.quantile(y_scores, 0.7)
# - confusion matrix
y_pred = np.where(y_scores > threshold, 1, 0)
Evaluation().plot_confusion_matrix(y_test, y_pred)

# - feature importance
feature_importance = pd.DataFrame()
feature_importance['feature'] = X_train.columns
feature_importance['fold_1'] = clf.feature_importances_
Evaluation(font_scale=2.0).plot_feature_importance(feature_importance.copy(), n_features_to_show=30)






