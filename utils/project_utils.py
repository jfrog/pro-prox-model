from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
# import hdbscan
from utils.fe_utils import get_amount_of_distinct_main_urls, get_sha1_id
import pandas as pd
from datetime import date
from utils.plot_utils import plot_elbow, plot_clusters
from pylab import *


def kmeans_and_plot(df_for_cluster):
    elbow_val = plot_elbow(df_for_cluster.copy(), max_k=10)
    model = KMeans(init='k-means++', max_iter=5000, random_state=0, n_clusters=elbow_val)
    df_for_cluster['group'] = pd.DataFrame(model.fit_predict(df_for_cluster))
    df_for_cluster['group'] = df_for_cluster['group'].astype(int)
    features_for_plot = df_for_cluster.columns.drop('group')
    plot_clusters(df_for_cluster, features_for_plot, min_members_in_group=1)
    return model, df_for_cluster


def dbscan_and_plot(df_for_cluster):
    model = DBSCAN(eps=1.9)
    df_for_cluster['group'] = pd.DataFrame(model.fit_predict(df_for_cluster))
    df_for_cluster['group'] = df_for_cluster['group'].astype(int)
    features_for_plot = df_for_cluster.columns.drop('group')
    plot_clusters(df_for_cluster, features_for_plot, min_members_in_group=1, is_dbscan=True)
    return model, df_for_cluster
    # pd.plotting.parallel_coordinates(
    #     df_for_cluster, 'group', color=('red', 'blue', 'green', 'purple', 'grey', 'yellow', '#05f03f')
    # )
    # plt.show()
    # plt.close()
    # fig, ax = plt.subplots(figsize=(12, 12))
    # df_for_cluster['color'] = df_for_cluster['group'].apply(
    #     lambda x: 'r' if x == 0 else 'g' if x == 1 else 'b' if x == 2 else 'y' if x == 3 else 'b')
    # if not feature_1 or not feature_2:
    #     feature_1, feature_2 = features_for_cluster[0], features_for_cluster[1]
    # ax.scatter(df_for_cluster[feature_1], df_for_cluster[feature_2], c=df_for_cluster.color)
    # ax.set_xlabel('versioning_up_count', size=20)
    # ax.set_ylabel('files_per_module', size=20)
    # ax.tick_params(axis='both', colors='black', size=10)
    # plt.title('before outliers removal \n total of ', c='b', size=25)
    # plt.show()


def gmm_and_plot(df_for_cluster):
    elbow_val = plot_elbow(df_for_cluster.copy(), max_k=10)
    model = GaussianMixture(n_components=4, random_state=1)
    df_for_cluster['group'] = pd.DataFrame(model.fit_predict(df_for_cluster))
    df_for_cluster['group'] = df_for_cluster['group'].astype(int)
    features_for_plot = df_for_cluster.columns.drop('group')
    plot_clusters(df_for_cluster, features_for_plot, min_members_in_group=1)
    return model, df_for_cluster


def cluster_and_plot(df_for_cluster, model='KMeans'):
    if model.lower() == 'kmeans':
        elbow_val = plot_elbow(df_for_cluster.copy(), max_k=10)
        model = KMeans(init='k-means++', max_iter=5000, random_state=0, n_clusters=elbow_val + 1)
    elif model.lower() == 'dbscan':
        model = DBSCAN(eps=1.9)
    elif model.lower() == 'gmm':
        model = GaussianMixture(n_components=4, random_state=1)
    df_for_cluster['group'] = pd.DataFrame(model.fit_predict(df_for_cluster))
    df_for_cluster['group'] = df_for_cluster['group'].astype(int)
    features_for_plot = df_for_cluster.columns.drop('group')
    plot_clusters(df_for_cluster, features_for_plot, min_members_in_group=1)
    return model, df_for_cluster


def plot_2d(df_for_plot, feature_1=None, feature_2=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    # df_for_plot['color'] = df_for_plot['group'].apply(
    #     lambda x: 'r' if x == 0 else 'g' if x == 1 else 'b' if x == 2 else 'y' if x == 3 else 'b')
    features_for_plot = list(df_for_plot.columns)
    if not feature_1 or not feature_2:
        feature_1, feature_2 = features_for_plot[0], features_for_plot[1]
    ax.scatter(df_for_plot[feature_1], df_for_plot[feature_2], c='blue')
    ax.set_xlabel(feature_1, size=20)
    ax.set_ylabel(feature_2, size=20)
    ax.tick_params(axis='both', colors='black', size=10)
    plt.title('2D plot', c='b', size=25)
    plt.show()
    # print(df_for_plot.loc[:, ['color', 'num_of_deployments']].groupby('color').median())
