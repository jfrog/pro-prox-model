from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import scipy
from sklearn.model_selection import RandomizedSearchCV
from kneed import KneeLocator
from eli5.sklearn import PermutationImportance
from utils.general_utils import load_data_old, get_cat_feature_names
from utils.model_extensions_utils import FocalLossObjective
from utils.plot_utils import Evaluation
from utils.preprocessing_utils import Boruta_feature_selection, drop_by_correlation
from utils.fe_utils import get_growth_features

import seaborn as sns
feature_selection = False
random_search = False
drop_correlated_features = False

df = load_data_old('get_data_fit.sql')
eval = Evaluation()
# - use only the latest opp
#latest_opp = df.groupby('account_id', as_index=False).agg({'relevant_date': 'max'})
#df = df.merge(latest_opp, on=['account_id', 'relevant_date'])

cols_to_drop = [col for col in df.columns if 'period_range' in col or 'relevant_date' in col or 'account_id' in col
                or 'class' in col]
X, y = df.drop(cols_to_drop, axis=1).fillna(-1), df['class']
# - drop zero variance features
cols_to_drop = [col for col in X.select_dtypes(include=np.number).columns if np.std(X[col]) == 0]
X = X.drop(cols_to_drop, axis=1)

tech_cols = ['maven', 'generic', 'docker', 'npm', 'pypi', 'gradle', 'nuget']
usage_cols = tech_cols + ['artifacts_count', 'artifacts_size', 'binaries_count', 'binaries_size', 'items_count',
                          'number_of_permissions', 'internal_groups', 'number_of_users', 'n_env', 'n_tech', 'n_repos']
X['n_tech'] = (X[tech_cols] != 0).astype(int).sum(axis=1)
X['n_tech.1'] = (X[[col + '.1' for col in tech_cols]] != 0).astype(int).sum(axis=1)
X['n_tech.2'] = (X[[col + '.2' for col in tech_cols]] != 0).astype(int).sum(axis=1)
X['n_tech.3'] = (X[[col + '.3' for col in tech_cols]] != 0).astype(int).sum(axis=1)
X['n_tech.4'] = (X[[col + '.4' for col in tech_cols]] != 0).astype(int).sum(axis=1)
X['n_repos'] = (X[tech_cols]).sum(axis=1)
X['n_repos.1'] = (X[[col + '.1' for col in tech_cols]]).sum(axis=1)
X['n_repos.2'] = (X[[col + '.2' for col in tech_cols]]).sum(axis=1)
X['n_repos.3'] = (X[[col + '.3' for col in tech_cols]]).sum(axis=1)
X['n_repos.4'] = (X[[col + '.4' for col in tech_cols]]).sum(axis=1)
X['leading_tech'] = list(X[tech_cols].idxmax(axis=1))
X.loc[X['leading_tech'].isin(['npm', 'gradle', 'pypi']), 'leading_tech'] = 'else'

# - get trends features
for col in usage_cols:
    growth_feature_monthly, growth_feature_quarter, df_fg = get_growth_features(col, X.copy())
    X[col + '_monthly_growth'] = growth_feature_monthly
    X[col + '_quarter_growth'] = growth_feature_quarter
    #X = pd.concat([X.copy(), df_fg], axis=1)
    X[col + r'/seniority'] = X[col] / X['seniority']

# - transform to category
cat_features = get_cat_feature_names(X)
for col in cat_features:
    X[col] = X[col].astype('category')

# - drop usage features from the periods before the relevant-date
cols_to_drop = [col for col in X.columns if '.1' in col or '.2' in col or '.3' in col or '.4' in col]
X = X.drop(cols_to_drop, axis=1)
X['artifacts/binaries_size'] = np.where(X['binaries_size'] == 0, 0, X['artifacts_size'] / X['binaries_size'])
X['artifacts/binaries_count'] = np.where(X['binaries_count'] == 0, 0, X['artifacts_count'] / X['binaries_count'])

cbc = CatBoostClassifier(cat_features=get_cat_feature_names(X), auto_class_weights='Balanced', verbose=0,
                         random_state=5, loss_function=FocalLossObjective(), eval_metric="Logloss", bootstrap_type='Bayesian')
lgb = LGBMClassifier(class_weight='balanced', random_state=5)
clfs = [('catboost', cbc), ('lightgbm', lgb)]
stacking = StackingClassifier(estimators=clfs, final_estimator=LogisticRegression(class_weight='balanced'))
feature_importance = eval.plot_cv_precision_recall(clf=cbc, n_folds=5, n_repeats=1, X=X, y=y, stacking=False,
                                              random_state=2, threshold=0.25)
eval.plot_feature_importance(feature_importance.copy(), n_features_to_show=30)



##############################################################################################################################




# ----- out-of-time examination
date_to_cut = '2020-12-01'
df['relevant_date'] = pd.to_datetime(df.relevant_date)
X_train, X_test = X[df.relevant_date <= date_to_cut], X[df.relevant_date > date_to_cut]
y_train, y_test = y[df.relevant_date <= date_to_cut], y[df.relevant_date > date_to_cut]

if drop_correlated_features:
    cols_to_drop = drop_by_correlation(X_train, threshold=0.95)
    X_train, X_test = X_train.drop(cols_to_drop, axis=1), X_test.drop(cols_to_drop, axis=1)

# - feature selection via feature importance
if feature_selection:
    cbc = CatBoostClassifier(cat_features=get_cat_feature_names(X_train), auto_class_weights='Balanced', verbose=0,
                             random_state=5, loss_function=FocalLossObjective(), eval_metric="Logloss", bootstrap_type='Bayesian')
    feature_importance = eval.plot_cv_precision_recall(clf=cbc, n_folds=5, n_repeats=1,
                                                  X=X_train.reset_index(drop=True), y=y_train.reset_index(drop=True))
    feature_importance['average_importance'] = feature_importance[[f'fold_{fold_n + 1}' for fold_n in range(5)]].mean(axis=1)
    imp = feature_importance.sort_values(by='average_importance', ascending=False)['average_importance']
    kn = KneeLocator(range(1, imp.shape[0] + 1), imp,  S=10.0, curve='convex', direction='decreasing')
    features_to_keep = feature_importance.sort_values(by='average_importance', ascending=False).head(kn.knee)['feature']
    X_train = X_train[[col for col in features_to_keep]]
    X_test = X_test[X_train.columns]
# - feature selection
X_train, X_test = Boruta_feature_selection(X_train.copy(), y_train.copy(), X_test.copy(), alpha=0.05)

# - choose hyper-parameters
if random_search:
    catboost_params_grid = {'iterations': [100, 250, 500, 1000],
                            'learning_rate': scipy.stats.uniform(0.01, 0.3),
                            'max_depth': scipy.stats.randint(3, 10),
                            'l2_leaf_reg': scipy.stats.reciprocal(a=1e-2, b=1e1),
                            'border_count': [5, 10, 20, 50, 100, 200],
                            'bootstrap_type': ['Bernoulli', 'Bayesian', 'MVS']}
    lgb_params_grid = {'num_leaves': [5, 10, 20, 40, 60],
                       'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                       'min_child_samples': [5, 10, 15, 50, 100],
                       'max_depth': [-1, 5, 10, 20],
                       'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
                       'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                       'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
    lgb = LGBMClassifier(class_weight='balanced', random_state=5)
    cb = CatBoostClassifier(auto_class_weights='Balanced', verbose=0, cat_features=get_cat_feature_names(X_train),
                            loss_function=FocalLossObjective(), eval_metric="Logloss")
    rs = RandomizedSearchCV(estimator=lgb, param_distributions=lgb_params_grid, scoring='average_precision',
                            refit=True, random_state=42, cv=3, n_iter=25, verbose=2, n_jobs=-1)
    rs.fit(X_train, y_train)
    y_pred_proba_rs = rs.predict_proba(X_test)[:, 1]

# --- default params
cbc = CatBoostClassifier(cat_features=get_cat_feature_names(X_train), auto_class_weights='Balanced', verbose=0,
                         random_state=5, loss_function=FocalLossObjective(), eval_metric="Logloss", bootstrap_type='Bayesian')
lgb = LGBMClassifier(class_weight='balanced', random_state=5)

cbc.fit(X_train, y_train)
y_pred_proba_cb = cbc.predict_proba(X_test)[:, 1]
eval.plot_precision_recall_test(y_test, y_pred_proba_cb)

threshold = 0.3
y_pred = np.where(y_pred_proba_cb > threshold, 1, 0)
eval.plot_confusion_matrix(y_test, y_pred)

# - feature importance
feature_importance = pd.DataFrame()
feature_importance['feature'] = X_train.columns
feature_importance['fold_1'] = cbc.feature_importances_
eval.eval.plot_feature_importance(feature_importance.copy(), n_features_to_show=30)

# - explainer dashboard
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
explainer = ClassifierExplainer(cbc, X_test, y_test)
ExplainerDashboard(explainer).run()

# - identify attackers
X_train1, X_test1 = X_train.copy(), X_test.copy()
X_train1['is_test'] = 0
X_test1['is_test'] = 1

tr_ts = pd.concat([X_train1, X_test1], axis=0).reset_index(drop=True)
X1, y1 = tr_ts.drop('is_test', axis=1), tr_ts['is_test']
feature_importance = eval.eval.plot_cv_precision_recall(clf=cbc, n_folds=5, n_repeats=1, X=X1, y=y1, stacking=False)
eval.plot_feature_importance(feature_importance.copy(), n_features_to_show=30)