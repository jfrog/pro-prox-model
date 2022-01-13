import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, f1_score, classification_report, confusion_matrix, \
    accuracy_score, average_precision_score, recall_score, precision_score
from scipy import interp
# from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from utils.fe_utils import add_anomaly_score_feature, add_cluster_feature, get_trial_scoring, \
    get_trial_scoring_clustering


class EDA:
    def __init__(self, font_scale=1.0):
        self.font_scale = font_scale

    def barplot_normalized(self, df, feature_name, class_name, rotate=False, min_obs_to_keep=50):
        """
        :param font_scale: as the name suggests
        :param min_obs_to_keep: min observations of a category to keep it for visualization
        :param df: a data frame
        :param feature_name: col name of a categorical variable
        :param class_name: col name of the class variable
        :param rotate: boolean, indicates whether to present the x ticks in 90 degrees rotation
        :return: none, it display a normalize bar plot for each category of col1 for each output of the class variable
        """
        df_to_plot = df.copy()
        categories_to_drop = [category for category in pd.unique(df_to_plot[feature_name]) if
                              df_to_plot[feature_name].value_counts()[category] < min_obs_to_keep]
        df_to_plot = (df_to_plot.groupby([feature_name])[class_name]
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values(class_name))
        df_to_plot = df_to_plot[~df_to_plot[feature_name].isin(categories_to_drop)]
        mapping = {category: str(category) + ': ' + str(list(df[feature_name].value_counts())[ind]) for ind, category in
                   enumerate(df[feature_name].value_counts().index)}
        df_to_plot = df_to_plot.replace({feature_name: mapping})
        sns.set(font_scale=self.font_scale)
        sns.barplot(x=feature_name, y='percentage', hue=class_name, data=df_to_plot)
        plt.xticks(rotation=90 * rotate)
        plt.show()


class Evaluation:
    def __init__(self, font_scale=1.0):
        self.font_scale = font_scale

    def plot_feature_importance(self, feature_importance, n_features_to_show=-1):
        """

        :param n_features_to_show:
        :param feature_importance: df consist of the feature names in the first columns and the importance
        for each fold
        :return:
        """
        n_folds = feature_importance.shape[1] - 1
        feature_importance['average_importance'] = feature_importance[
            [f'fold_{fold_n + 1}' for fold_n in range(n_folds)]].mean(axis=1)
        if not n_features_to_show == -1:
            plt.figure(figsize=(16, 16))
            sns.set(font_scale=self.font_scale)
            sns.barplot(
                data=feature_importance.sort_values(by='average_importance', ascending=False).head(n_features_to_show),
                x='average_importance', y='feature')
            plt.title('Feature Importances over {} folds'.format(n_folds))
            # plt.savefig(fig_name + '.png', bbox_inches='tight')
            plt.show()

        else:
            plt.figure(figsize=(16, 16))
            sns.set(font_scale=self.font_scale)
            sns.barplot(data=feature_importance.sort_values(by='average_importance', ascending=False),
                        x='average_importance', y='feature')
            plt.title('Feature Importances over {} folds'.format(n_folds))
            # plt.savefig(fig_name + '.png', bbox_inches='tight')
            plt.show()

    def plot_cv_precision_recall(self, clf, n_folds, n_repeats, X, y, random_state=0, stacking=False,
                                 threshold=0.5):
        cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        f1s = []
        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        feature_importance = pd.DataFrame()
        feature_importance['feature'] = X.columns

        plt.figure(figsize=(24, 18))
        i = 0
        for train, test in cv.split(X, y):
            x_train, x_test = X.loc[X.index[train], :], X.loc[X.index[test], :]
            probas_ = clf.fit(x_train, y[y.index[train]]).predict_proba(x_test)
            if not stacking:
                feature_importance[f'fold_{i + 1}'] = clf.feature_importances_
            else:
                weights = clf.final_estimator_.coef_
                weights /= np.sum(weights)
                imp = np.zeros((x_train.shape[1], len(clf.estimators_)))
                for ind, est in enumerate(clf.estimators_):
                    clf_i = clf.estimators_[ind]
                    imp_i = clf_i.feature_importances_
                    imp[:, ind] = (imp_i - np.min(imp_i)) / (np.max(imp_i) - np.min(imp_i))
                avg_imp = np.matmul(imp, np.transpose(weights))
                feature_importance[f'fold_{i + 1}'] = avg_imp
            # y_pred = np.argmax(probas_, axis=1)
            y_pred = np.where(probas_[:, 1] > threshold, 1, 0)
            # Compute PR curve and area the curve
            # precision, recall, thresholds = precision_recall_curve(y[y.index[test]], probas_[:, 1])
            precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
            prs.append(interp(mean_recall, precision, recall))
            # pr_auc = auc(recall, precision)
            pr_auc = average_precision_score(y[test], probas_[:, 1])
            aucs.append(pr_auc)
            f1s.append(f1_score(y_pred=y_pred, y_true=y[test]))
            plt.plot(recall, precision, lw=3, alpha=0.5, label='Fold %d (AUCPR = %0.2f)' % (i + 1, pr_auc))
            i += 1

        plt.hlines(y=np.mean(y), xmin=0.0, xmax=1.0, linestyle='--', lw=3, color='k', label='Luck', alpha=.8)
        mean_precision = np.mean(prs, axis=0)
        # mean_auc = auc(mean_recall, mean_precision)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        plt.plot(mean_precision, mean_recall, color='navy',
                 label=r'Mean (AUCPR = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=4)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontweight="bold", fontsize=30)
        plt.ylabel('Precision', fontweight="bold", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(prop={'size': 20}, loc=0)
        plt.text(x=0.2, y=0.3, s=r'Mean (F1 = %0.3f $\pm$ %0.2f)' % (np.mean(f1s), np.std(f1s)), fontsize=20)
        plt.show()
        return feature_importance

    def plot_confusion_matrix(self, y_test, y_pred):
        sns.set(font_scale=self.font_scale)
        recall = np.round(recall_score(y_true=y_test, y_pred=y_pred), 3)
        precision = np.round(precision_score(y_true=y_test, y_pred=y_pred), 3)
        cf_matrix = confusion_matrix(y_test, y_pred)
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        plt.text(1.7, 2.0, 'Recall: ' + str(recall), style='italic', fontsize=10,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 2})
        plt.text(1.7, 2.1, 'Precision: ' + str(precision), style='italic', fontsize=10,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 2})
        plt.show()

    def plot_precision_recall_test(self, y_true, y_scores, title=''):
        mean_recall = np.linspace(0, 1, 100)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        mean_precision = interp(mean_recall, precision, recall)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(24, 18))
        sns.set(font_scale=self.font_scale)
        plt.hlines(y=np.mean(y_true), xmin=0.0, xmax=1.0, linestyle='--', lw=3, color='k', label='Luck', alpha=.8)
        plt.plot(mean_precision, mean_recall, color='navy',
                 label=r'(AUCPR =' + str(np.round(pr_auc, 2)), lw=4)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title(title, fontdict ={'family': 'serif', 'color': 'darkred', 'size': 50})
        plt.xlabel('Recall', fontweight="bold", fontsize=30)
        plt.ylabel('Precision', fontweight="bold", fontsize=30)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(prop={'size': 20}, loc=0)
        plt.show()


class SHAPAnalysis:
    def __init__(self, font_scale=1.0, save_fig=False):
        self.font_scale = font_scale
        self.save_fig = save_fig

    def exploring_positive_class(self, result_df, output_df, cols, fig_name=''):
        """

        :param result_df: a df contains 3 columns: account, class_pred, and prob
        :param output_df: a df created with create_output_table function
        :param cols: feature columns
        :param save_fig: boolean indicates whether to save the figure or not
        :param fig_name: name of the saved figure
        :return: show/save a figure exploring the features effect on the positive class
        """
        # --- exploring fn
        fn_emails = result_df.loc[(result_df['class'] == 1) & (result_df['class_pred'] == 0), 'email']
        output_df_fn = output_df[output_df['email'].isin(fn_emails)]

        # --- features that decreased the prob for the false-negative requests
        negative_effect = output_df_fn[output_df_fn['shap_importance'] < 0]
        count_features_occ_fn = {feature: np.sum(negative_effect['feature'] == feature) for feature in
                                 pd.unique(negative_effect['feature'])}
        count_features_occ_fn = {feature: negative_effect[negative_effect['feature'] == feature].shape[0] for feature in
                                 cols}

        # --- exploring tp
        tp_emails = result_df.loc[(result_df['class'] == 1) & (result_df['class_pred'] == 1), 'email']
        output_df_tp = output_df[output_df['email'].isin(tp_emails)]

        # --- features that increase the prob for the true-positive requests
        positive_effect = output_df_tp[output_df_tp['shap_importance'] > 0]

        count_features_occ_tp = {feature: positive_effect[positive_effect['feature'] == feature].shape[0] for feature in
                                 cols}

        # --- plot negative effect on fn & positive effect on tp
        positive_df = pd.DataFrame()
        positive_df['feature'] = np.nan
        positive_df['count'] = np.nan
        positive_df['is_tp'] = np.nan

        ind = 0
        for key, value in count_features_occ_fn.items():
            positive_df.loc[ind, 'count'] = value
            positive_df.loc[ind, 'feature'] = key
            positive_df.loc[ind, 'is_tp'] = 0
            ind += 1

        for key, value in count_features_occ_tp.items():
            positive_df.loc[ind, 'count'] = value
            positive_df.loc[ind, 'feature'] = key
            positive_df.loc[ind, 'is_tp'] = 1
            ind += 1

        inds_to_drop = []
        for feature in cols:
            if np.sum(positive_df.loc[positive_df['feature'] == feature, 'count'] == 0) == 2:
                inds_to_drop += list(positive_df[positive_df['feature'] == feature].index)

        positive_df = positive_df.drop(inds_to_drop, axis=0)
        sns.set(font_scale=self.font_scale)
        plt.figure(figsize=(15, 12))
        sns.barplot(x="feature", y="count", hue="is_tp", data=positive_df)
        plt.xticks(rotation=90)
        if not self.save_fig:
            plt.show()
        else:
            plt.savefig(fig_name)
            plt.close()

    def exploring_negative_class(self, result_df, output_df, cols, fig_name=''):
        """

        :param result_df: a df contains 3 columns: account, class_pred, and prob
        :param output_df: a df created with create_output_table function
        :param cols: feature columns
        :param save_fig: boolean indicates whether to save the figure or not
        :param fig_name: name of the saved figure
        :return: show/save a figure exploring the features effect on the negative class
        """
        fp_emails = result_df.loc[(result_df['class'] == 0) & (result_df['class_pred'] == 1), 'email']
        output_df_fp = output_df[output_df['email'].isin(fp_emails)]

        # --- features that increased the prob for the false-positive requests
        positive_effect = output_df_fp[output_df_fp['shap_importance'] > 0]
        count_features_occ_fp = {feature: positive_effect[positive_effect['feature'] == feature].shape[0] for feature in
                                 cols}

        # --- exploring tn
        tn_emails = result_df.loc[(result_df['class'] == 0) & (result_df['class_pred'] == 0), 'email']
        output_df_tn = output_df[output_df['email'].isin(tn_emails)]

        # --- features that decreased the prob for the true-negative requests
        negative_effect = output_df_tn[output_df_tn['shap_importance'] < 0]
        count_features_occ_tn = {feature: negative_effect[negative_effect['feature'] == feature].shape[0] for feature in
                                 cols}

        # --- plot negative effect on tn & positive effect on fp
        negative_df = pd.DataFrame()
        negative_df['feature'] = np.nan
        negative_df['count'] = np.nan
        negative_df['is_tn'] = np.nan

        ind = 0
        for key, value in count_features_occ_fp.items():
            negative_df.loc[ind, 'count'] = value
            negative_df.loc[ind, 'feature'] = key
            negative_df.loc[ind, 'is_tn'] = 0
            ind += 1

        for key, value in count_features_occ_tn.items():
            negative_df.loc[ind, 'count'] = value
            negative_df.loc[ind, 'feature'] = key
            negative_df.loc[ind, 'is_tn'] = 1
            ind += 1

        inds_to_drop = []
        for feature in cols:
            if np.sum(negative_df.loc[negative_df['feature'] == feature, 'count'] == 0) == 2:
                inds_to_drop += list(negative_df[negative_df['feature'] == feature].index)

        negative_df = negative_df.drop(inds_to_drop, axis=0)
        plt.figure(figsize=(15, 12))
        sns.set(font_scale=self.font_scale)
        sns.barplot(x="feature", y="count", hue="is_tn", data=negative_df)
        plt.xticks(rotation=90)
        if not self.save_fig:
            plt.show()
        else:
            plt.savefig(fig_name)
            plt.close()


def train_test_difference(train_df, test_df, threshold=0.1):
    """Use KS to estimate columns where distributions differ a lot from each other"""

    # Find the columns where the distributions are very different
    diff_data = []
    for col in tqdm(train_df.columns):
        statistic, pvalue = ks_2samp(
            train_df[col].values,
            test_df[col].values
        )
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diff_data.append({'feature': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    # Put the differences into a dataframe
    diff_df = pd.DataFrame(diff_data).sort_values(by='statistic', ascending=False)

    # Let us see the distributions of these columns to confirm they are indeed different
    n_cols, n_rows = 7, 10

    _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = [x for l in axes for x in l]

    # Create plots
    for i, (_, row) in enumerate(diff_df.iterrows()):
        if i >= len(axes):
            break
        extreme = np.max(np.abs(train_df[row.feature].tolist() + test_df[row.feature].tolist()))
        train_df.loc[:, row.feature].apply(np.log1p).hist(
            ax=axes[i], alpha=0.5, label='Train', density=True,
            bins=np.arange(-extreme, extreme, 0.25)
        )
        test_df.loc[:, row.feature].apply(np.log1p).hist(
            ax=axes[i], alpha=0.5, label='Test', density=True,
            bins=np.arange(-extreme, extreme, 0.25)
        )
        axes[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
        axes[i].set_xlabel(f'Log({row.feature})')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

    return diff_df


def plot_elbow(data, random_state=0, min_k=2, max_k=5):
    model = KMeans(init='k-means++', max_iter=3000, random_state=random_state)
    visualizer = KElbowVisualizer(model, k=(min_k, max_k))
    visualizer.fit(data)
    visualizer.show()
    return visualizer.elbow_value_


def plot_clusters(df_, features, min_members_in_group=1, is_dbscan=False):
    group_counts = df_['group'].value_counts()
    groups_to_keep = [group for group in group_counts.index if group_counts[group] > min_members_in_group]
    df_to_plot = df_[df_['group'].isin(groups_to_keep)]
    categories = features
    categories = [*categories, categories[0]]
    # TODO: Change lables of calculation for each feature
    # agg_dict = {'Usage': 'median', 'Organization-spread': 'median', 'purchase': 'mean'}
    agg_dict = {col: "median" for col in categories}
    # --- radar plot for each seniority group
    df_agg_by_cluster = df_to_plot.groupby(by='group').agg(agg_dict)
    group_counts = list(df_to_plot['group'].value_counts(sort=False))
    group_vals = [list(df_agg_by_cluster.loc[i]) for i in np.sort(pd.unique(df_to_plot['group']))]
    for i in range(len(group_vals)):
        group_vals[i] = [*group_vals[i], group_vals[i][0]]
    # group_mapping = {group: ind + 1 for ind, group in enumerate(pd.unique(df_to_plot['group']))}
    # df_to_plot = df_to_plot.replace({'group': group_mapping})
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'tan', 'lime', 'teal', 'grey']
    print(df_to_plot['group'].value_counts())
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(group_vals[0]))
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    is_dbscan_indicator = int(is_dbscan)
    for i in np.sort(pd.unique(df_to_plot['group'])):
        plt.plot(label_loc, group_vals[i - is_dbscan_indicator], label='G' + str(i) + ' (' + str(group_counts[i - is_dbscan_indicator]) + ')', color=colors[i - is_dbscan_indicator])
    plt.title('Radar plot', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend(loc='lower right')
    plt.show()