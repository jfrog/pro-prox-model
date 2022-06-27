import numpy as np
import scipy
from scipy import stats
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from utils.general_utils import get_cat_feature_names
from utils.model_extensions_utils import FocalLossObjective
from utils.plot_utils import Evaluation


def cv_evaluation(model, X, y, n_folds=5, n_iter=20, scoring='average_precision', agg_scores='mean',
                  n_features_to_keep=50):
    if model == 'rf':
        params = {'n_estimators': [200, 500, 1000],
                  'max_depth': stats.randint(3, 10),
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_split': stats.randint(3, 10),
                  'min_samples_leaf': stats.randint(1, 5)}
        est = RandomForestClassifier(criterion='entropy', bootstrap=True,
                                     oob_score=True, random_state=2, class_weight='balanced')
    elif model == 'lgb':
        params = {'num_leaves': scipy.stats.randint(6, 50),
                           'min_child_samples': scipy.stats.randint(100, 500),
                           'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                           'learning_rate': scipy.stats.uniform(0.01, 0.3),
                           'subsample': scipy.stats.uniform(loc=0.2, scale=0.8),
                           'colsample_bytree': scipy.stats.uniform(loc=0.4, scale=0.6),
                           'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                           'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
        est = LGBMClassifier(class_weight='balanced', random_state=5, categorical_feature='auto')
    elif model == 'cbc':
        params = {'iterations': [100, 250, 500, 1000],
                  'learning_rate': stats.uniform(0.01, 0.3),
                  'max_depth': stats.randint(3, 10),
                  'l2_leaf_reg': stats.reciprocal(a=1e-2, b=1e1),
                  'border_count': [5, 10, 20, 50, 100, 200],
                  'bootstrap_type': ['Bernoulli', 'Bayesian', 'MVS']}
        est = CatBoostClassifier(cat_features=get_cat_feature_names(X), auto_class_weights="Balanced", random_state=5,
                                 rsm=0.1, verbose=0, loss_function=FocalLossObjective(), eval_metric="Logloss")
    elif model == 'hist':
        params = {'max_iter': [100, 250, 500, 1000],
                  'max_leaf_nodes': stats.randint(2,100),
                  'learning_rate': stats.uniform(0.01, 0.3),
                  'max_depth': stats.randint(3, 10),
                  'min_samples_leaf': stats.randint(1, 30)}
        est = HistGradientBoostingClassifier(categorical_features=get_cat_feature_names(X), verbose=0,
                                             random_state=5, loss="auto", scoring="Logloss")

    if model in ['hist', 'rf']:
        X = X.select_dtypes(include=np.number)

    if model != 'hist':
        feature_importance = Evaluation().plot_cv_precision_recall(clf=est, n_folds=n_folds, n_repeats=1, X=X, y=y)
        feature_importance['average_importance'] = feature_importance[[f'fold_{fold_n + 1}' for fold_n in range(n_folds)]].mean(axis=1)
        features_to_keep = feature_importance.sort_values(by='average_importance', ascending=False).head(n_features_to_keep)['feature']
        X_selected = X[features_to_keep]
    else:
        X_selected = X.copy()



    if model == 'cbc':
        est = CatBoostClassifier(cat_features=get_cat_feature_names(X_selected), auto_class_weights="Balanced", random_state=5,
                                 rsm=0.1, verbose=0, loss_function=FocalLossObjective(), eval_metric="Logloss")

    clf = RandomizedSearchCV(estimator=est, param_distributions=params,
                             scoring=scoring, refit=True, random_state=5, cv=n_folds, n_iter=n_iter,
                             verbose=2, n_jobs=-1)
    cv_scores = cross_val_score(estimator=clf, X=X_selected, y=y, cv=n_folds, scoring=scoring)
    if agg_scores == 'mean':
        agged_scores = np.mean(cv_scores)
    elif agg_scores == 'median':
        agged_scores = np.median(cv_scores)
    clf.fit(X_selected, y)
    if model == 'rf':
        best_clf = RandomForestClassifier(criterion='entropy', bootstrap=True, oob_score=True, random_state=2,
                                          class_weight='balanced', **clf.best_params_)
    elif model == 'lgb':
        best_clf = LGBMClassifier(class_weight='balanced', random_state=5, **clf.best_params_)
    elif model == 'cbc':
        best_clf = CatBoostClassifier(cat_features=get_cat_feature_names(X_selected), auto_class_weights="Balanced", random_state=5,
                                      rsm=0.1, verbose=0, loss_function=FocalLossObjective(), eval_metric="Logloss", **clf.best_params_)
    elif model == 'hist':
        best_clf = HistGradientBoostingClassifier(categorical_features=get_cat_feature_names(X), verbose=0,
                                                  random_state=5, loss="auto", scoring="Logloss", **clf.best_params_)
    best_clf.fit(X_selected, y)
    return agged_scores, best_clf, X_selected.columns

