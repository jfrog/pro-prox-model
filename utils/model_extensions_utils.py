import pandas as pd
import numpy as np
import math
from tqdm import tqdm

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils.general_utils import get_cat_feature_names


# - Focal Loss - a loss for imbalanced classification
class FocalLossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers with only __len__ and __getitem__ defined).
        # weights parameter can be None.
        # Returns list of pairs (der1, der2)
        gamma = 2.
        # alpha = 1.
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        exponents = []
        for index in range(len(approxes)):
            exponents.append(math.exp(approxes[index]))

        result = []
        for index in range(len(targets)):
            p = exponents[index] / (1 + exponents[index])

            if targets[index] > 0.0:
                der1 = -((1 - p) ** (gamma - 1)) * (gamma * math.log(p) * p + p - 1) / p
                der2 = gamma * ((1 - p) ** gamma) * ((gamma * p - 1) * math.log(p) + 2 * (p - 1))
            else:
                der1 = (p ** (gamma - 1)) * (gamma * math.log(1 - p) - p) / (1 - p)
                der2 = p ** (gamma - 2) * (
                        (p * (2 * gamma * (p - 1) - p)) / (p - 1) ** 2 + (gamma - 1) * gamma * math.log(1 - p))

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


# - A custom stacking method, enables to stack models with different set of features.
class CustomStacking:
    def __init__(self, classifiers=[], final_estimator=LogisticRegression(class_weight='balanced'), numeric_inds=[]):
        self.classifiers = classifiers
        self.final_estimator = final_estimator
        self.numeric_inds = numeric_inds

    def fit(self, x_trains, y_train):
        """

        :param x_trains: list of DataFrames of features
        :param y_train: the training set labels
        :return:
        """
        x_train = pd.DataFrame()
        for c in range(len(x_trains)):
            self.classifiers[c].fit(x_trains[c], y_train)
            x_train[str(c)] = self.classifiers[c].predict_proba(x_trains[c])[:, 1]
        self.final_estimator.fit(x_train, y_train)

    def predict_proba(self, x_tests):
        """

        :param x_tests: list of DataFrames of features (similar to x_trains)
        :return: predict the probability to belong for each of the classes
        """
        x_test = pd.DataFrame()
        for c in range(len(x_tests)):
            x_test[str(c)] = self.classifiers[c].predict_proba(x_tests[c])[:, 1]
        return self.final_estimator.predict_proba(x_test)

    def feature_importance(self, x_trains):
        n_features = np.max([x_train.shape[1] for x_train in x_trains])
        weights = self.final_estimator.coef_
        weights /= np.sum(weights)
        imp = np.zeros((n_features, len(self.estimators)))
        for c in range(len(self.estimators)):
            clf_i = self.estimators[c]
            imp_i = clf_i.feature_importances_
            if x_trains[c] < n_features:
                imp[self.numeric_inds, c] = (imp_i - np.min(imp_i)) / (np.max(imp_i) - np.min(imp_i))
            else:
                imp[:, c] = (imp_i - np.min(imp_i)) / (np.max(imp_i) - np.min(imp_i))

        cat_inds = [ind for ind in range(n_features) if ind not in self.numeric_inds]
        for ind in cat_inds:
            imp[ind, :] = imp[ind, :][imp[ind, :] != 0][0]
        return np.matmul(imp, np.transpose(weights))


# - get the sample weights by train & tests similarity
def get_sample_weights(X_train, X_test):
    """

    :param X_train: the training set (a DataFrame)
    :param X_test: the tests set (a DataFrame)
    :return: the weight of each training example by the probability to belong to the tests set,
    the motivation behind it is that an example that similar to tests set will be more use-full for the prediction
    """
    X_train1, X_test1 = X_train.copy(), X_test.copy()
    X_train1['is_test'] = 0
    X_test1['is_test'] = 1
    tr_ts = pd.concat([X_train1, X_test1], axis=0, ignore_index=True)
    X_attack, y_attack = tr_ts.drop('is_test', axis=1), tr_ts['is_test']

    predictions = np.zeros(y_attack.shape)
    cbc = CatBoostClassifier(cat_features=get_cat_feature_names(X_attack), auto_class_weights='Balanced',
                             verbose=0, random_state=5, loss_function=FocalLossObjective(), eval_metric="Logloss",
                             bootstrap_type='Bayesian', rsm=0.1)
    cv = StratifiedKFold(n_splits=5)

    for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X_attack, y_attack))):
        cbc.fit(X_attack.loc[train_idx, :], y_attack.loc[train_idx])
        probs = cbc.predict_proba(X_attack.loc[test_idx, :])[:, 1]
        predictions[test_idx] = probs

    sample_weight = predictions[:X_train1.shape[0]]
    return sample_weight