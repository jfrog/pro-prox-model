from abc import ABC, abstractmethod
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from config.insights import ShortInsights, MediumInsights


class Insights(ABC):
    RELATION_PATTERN = r'\((.*?)\)'
    NUMBER_PATTERN = r'([-+]?\d*\.?\d+)'
    SHORT = ShortInsights().to_dict()
    MEDIUM = MediumInsights().to_dict()

    def __init__(self, features_df):
        self.features_df = features_df

    def _select_index(self, feature_col):
        return self.features_df[self.features_df['feature'] \
                                == feature_col].index

    def _binary_feat(self, feature_col, text_before_value, text_after_value, short_insight='', medium_insight='',
                     is_past=False):
        idx = self._select_index(feature_col)
        if len(idx) > 0:
            value = pd.Series(
                np.where(
                    self.features_df.loc[idx, 'feature_value'].str.contains('1.0'),
                    '',
                    " didn't" if is_past else ' not'
                )
            )
            relation = self.features_df.loc[idx, 'feature_value'] \
                .str.extract(self.RELATION_PATTERN)
            insight = text_before_value + value + f' using {text_after_value}, ' + relation
            self.features_df.loc[idx, 'long_insight'] = insight.values
            self.features_df.loc[idx, 'medium_insight'] = medium_insight
            self.features_df.loc[idx, 'short_insight'] = short_insight

    def _categorical_feat(self, feature_col: str, text_before_cat: str, text_after_cat: str,
                          short_insight: str = ''):
        idx = self._select_index(feature_col)
        if len(idx) > 0:
            value = self.features_df.loc[idx, 'feature_value']
            if feature_col in ['Revenue range', 'Number of employees (range)']:
                text_before_cat = pd.Series([text_before_cat.rsplit(' ', 1)[0] if val.startswith('Over') or val.startswith('Under') else text_before_cat
                                             for val in value.values])
                insight = text_before_cat.values + ' ' + value + f' {text_after_cat}'
            else:
                insight = f'{text_before_cat} ' + value \
                          + f' {text_after_cat}'
            self.features_df.loc[idx, 'long_insight'] = insight.values
            self.features_df.loc[idx, 'medium_insight'] = insight.values
            self.features_df.loc[idx, 'short_insight'] = short_insight

    def _numeric_feat(self, feature_col: str, text_before_num: str, text_after_num: str,
                      short_insight: str = '', medium_insight: str = ''):
        idx = self._select_index(feature_col)
        is_growth_feature = 'growth' in feature_col or 'incline' in feature_col
        if len(idx) > 0:
            relation = self.features_df.loc[idx, 'feature_value'] \
                .str.extract(self.RELATION_PATTERN)
            numbers = self.features_df.loc[idx, 'feature_value'] \
                .str.extract(self.NUMBER_PATTERN).astype(float).round(2).astype(str)
            if is_growth_feature:
                numbers = (numbers.astype(float) * 100).astype(str) + '%'
            insight = f'{text_before_num} ' + numbers \
                      + f' {text_after_num}, ' + relation
            # make missing values interpretable
            insight = insight.replace('-1', '0', regex=True)
            self.features_df.loc[idx, 'long_insight'] = insight.values
            self.features_df.loc[idx, 'medium_insight'] = insight.values if feature_col.lower().__contains__('seniority') else medium_insight
            self.features_df.loc[idx, 'short_insight'] = short_insight

    @abstractmethod
    def _mapping(self):
        pass

    def translate_into_insight(self, n_jobs=-1):
        features = self.features_df['feature'].unique()
        Parallel(n_jobs=n_jobs)(delayed(self._mapping().get)(feat)
                                for feat in features)


class InsightsSHUpsell(Insights):

    def __init__(self, features_df):
        super().__init__(features_df)
        self.features_df['long_insight'] = ''
        self.features_df['medium_insight'] = ''
        self.features_df['short_insight'] = ''

    def _n_ent_mentioned_sessions(self):
        self._numeric_feat(feature_col='Number of times enterprise mentioned in sessions',
                           text_before_num='Account has',
                           text_after_num='technical session(s) in which enterprise was mentioned in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'])

    def _n_ha_mentioned_sessions(self):
        self._numeric_feat(feature_col='Number of times high-availability mentioned in sessions',
                           text_before_num='Account has',
                           text_after_num='technical session(s) in which high-availability was mentioned in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'])

    def _n_xray_mentioned_sessions(self):
        self._numeric_feat(feature_col='Number of times xray mentioned in sessions',
                           text_before_num='Account has',
                           text_after_num='technical session(s) in which xray was mentioned in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'])

    def _n_competitor_mentioned_sessions(self):
        self._numeric_feat(feature_col='Number of times competitors mentioned in sessions',
                           text_before_num='Account has',
                           text_after_num='technical session(s) in which competitors was mentioned in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'])

    def _n_users(self):
        self._numeric_feat(feature_col='Number of users',
                           text_before_num="Account's number of users in the JFrog platform is",
                           text_after_num='',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_USERS'])

    def _n_repos(self):
        self._numeric_feat(feature_col='Number of repositories',
                           text_before_num="Account has",
                           text_after_num='repositories in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_REPOSITORIES'])

    def _n_env(self):
        self._numeric_feat(feature_col='Number of environments',
                           text_before_num="Account has",
                           text_after_num='environments in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_ENVIRONMENTS'])

    def _n_permissions(self):
        self._numeric_feat(feature_col='Number of permissions',
                           text_before_num="Account has",
                           text_after_num='permission(s) in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_PERMISSIONS'])

    def _n_internal_groups(self):
        self._numeric_feat(feature_col='Number of internal groups',
                           text_before_num="Account has",
                           text_after_num='internal groups(s) in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_INTERNAL_GROUPS'])

    def _n_artifacts_count(self):
        self._numeric_feat(feature_col='Storage: artifacts count',
                           text_before_num="Account has",
                           text_after_num='artifacts in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_ARTIFACTS'])

    def _total_artifacts_size(self):
        self._numeric_feat(feature_col='Storage: artifacts size',
                           text_before_num="The total size of the artifacts in the account's platform is",
                           text_after_num='GB',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['ARTIFACTS_SIZE'])

    def _total_binaries_size(self):
        self._numeric_feat(feature_col='Storage: binaries size',
                           text_before_num="The total size of the binaries in the account's platform is",
                           text_after_num='GB',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['BINARIES_SIZE'])

    def _n_binaries_count(self):
        self._numeric_feat(feature_col='Storage: binaries count',
                           text_before_num="Account has",
                           text_after_num='binaries in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'])

    def _n_items_count(self):
        self._numeric_feat(feature_col='Storage: items count',
                           text_before_num="Account has",
                           text_after_num='items in JFrog platform',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_ITEMS'])

    def _n_users_growth_monthly(self):
        self._numeric_feat(feature_col='Number of users monthly incline',
                           text_before_num="Account's number of users increased by",
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_USERS_IN_THE_LAST_3_MONTHS'])

    def _n_users_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of users quarterly incline',
                           text_before_num="Account's number of users increased by",
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_USERS_IN_THE_LAST_3_QUARTERS'])

    def _n_permissions_growth_monthly(self):
        self._numeric_feat(feature_col='Number of permissions monthly incline',
                           text_before_num="Account's number of permissions increased by",
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_PERMISSIONS_IN_THE_LAST_3_MONTHS'])

    def _n_permissions_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of permissions quarterly incline',
                           text_before_num="Account's number of permissions increased by",
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_PERMISSIONS_IN_THE_LAST_3_QUARTERS'])

    def _n_internal_groups_growth_monthly(self):
        self._numeric_feat(feature_col='Number of internal groups monthly incline',
                           text_before_num="Account's number of internal groups increased by",
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_INTERNAL_GROUPS_IN_THE_LAST_3_MONTHS'])

    def _n_internal_groups_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of internal groups quarterly incline',
                           text_before_num="Account's number of internal groups increased by",
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_INTERNAL_GROUPS_IN_THE_LAST_3_QUARTERS'])

    def _artifacts_count_growth_monthly(self):
        self._numeric_feat(feature_col='Storage: artifacts count monthly incline',
                           text_before_num='Number of artifacts in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ARTIFACTS_IN_THE_LAST_3_MONTHS'])

    def _artifacts_count_growth_quarterly(self):
        self._numeric_feat(feature_col='Storage: artifacts count quarterly incline',
                           text_before_num='Number of artifacts in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ARTIFACTS_IN_THE_LAST_3_QUARTERS'])

    def _binaries_count_growth_monthly(self):
        self._numeric_feat(feature_col='Storage: binaries count monthly incline',
                           text_before_num='Number of binaries in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_BINARIES_IN_THE_LAST_3_MONTHS'])

    def _binaries_count_growth_quarterly(self):
        self._numeric_feat(feature_col='Storage: binaries count quarterly incline',
                           text_before_num='Number of binaries in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_BINARIES_IN_THE_LAST_3_QUARTERS'])

    def _artifacts_size_growth_monthly(self):
        self._numeric_feat(feature_col='Storage: artifacts size monthly incline',
                           text_before_num='The total size of the artifacts in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_ARTIFACTS_SIZE_IN_THE_LAST_3_MONTHS'])

    def _artifacts_size_growth_quarterly(self):
        self._numeric_feat(feature_col='Storage: artifacts size quarterly incline',
                           text_before_num='The total size of the artifacts in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_ARTIFACTS_SIZE_IN_THE_LAST_3_QUARTERS'])

    def _binaries_size_growth_monthly(self):
        self._numeric_feat(feature_col='Storage: binaries size monthly incline',
                           text_before_num='The total size of the binaries in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_BINARIES_SIZE_IN_THE_LAST_3_MONTHS'])

    def _binaries_size_growth_quarterly(self):
        self._numeric_feat(feature_col='Storage: binaries size quarterly incline',
                           text_before_num='The total size of the binaries in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_BINARIES_SIZE_IN_THE_LAST_3_QUARTERS'])

    def _items_count_growth_monthly(self):
        self._numeric_feat(feature_col='Storage: items count monthly incline',
                           text_before_num='Number of items in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ITEMS_IN_THE_LAST_3_MONTHS'])

    def _items_count_growth_quarterly(self):
        self._numeric_feat(feature_col='Storage: items count quarterly incline',
                           text_before_num='Number of items in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ITEMS_IN_THE_LAST_3_QUARTERS'])

    def _n_docker_repos(self):
        self._numeric_feat(feature_col='Docker repositories',
                           text_before_num='Account has',
                           text_after_num='Docker repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_DOCKER_REPOSITORIES'])

    def _docker_growth_monthly(self):
        self._numeric_feat(feature_col='Docker monthly incline',
                           text_before_num='Docker repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_DOCKER_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _docker_growth_quarterly(self):
        self._numeric_feat(feature_col='Docker quarterly incline',
                           text_before_num='Docker repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_DOCKER_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_maven_repos(self):
        self._numeric_feat(feature_col='Maven repositories',
                           text_before_num='Account has',
                           text_after_num='Maven repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_MAVEN_REPOSITORIES'])

    def _maven_growth_monthly(self):
        self._numeric_feat(feature_col='Maven monthly incline',
                           text_before_num='Maven repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_MAVEN_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _maven_growth_quarterly(self):
        self._numeric_feat(feature_col='Maven quarterly incline',
                           text_before_num='Maven repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_MAVEN_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_generic_repos(self):
        self._numeric_feat(feature_col='Generic repositories',
                           text_before_num='Account has',
                           text_after_num='Generic repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_GENERIC_REPOSITORIES'])

    def _generic_growth_monthly(self):
        self._numeric_feat(feature_col='Generic monthly incline',
                           text_before_num='Generic repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_GENERIC_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _generic_growth_quarterly(self):
        self._numeric_feat(feature_col='Generic quarterly incline',
                           text_before_num='Generic repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_GENERIC_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_nuget_repos(self):
        self._numeric_feat(feature_col='Nuget repositories',
                           text_before_num='Account has',
                           text_after_num='Nuget repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_NUGET_REPOSITORIES'])

    def _nuget_growth_monthly(self):
        self._numeric_feat(feature_col='Nuget monthly incline',
                           text_before_num='Nuget repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_NUGET_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _nuget_growth_quarterly(self):
        self._numeric_feat(feature_col='Nuget quarterly incline',
                           text_before_num='Nuget repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_NUGET_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_pypi_repos(self):
        self._numeric_feat(feature_col='Pypi repositories',
                           text_before_num='Account has',
                           text_after_num='Pypi repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_PYPI_REPOSITORIES'])

    def _pypi_growth_monthly(self):
        self._numeric_feat(feature_col='Pypi monthly incline',
                           text_before_num='Pypi repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_PYPI_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _pypi_growth_quarterly(self):
        self._numeric_feat(feature_col='Pypi quarterly incline',
                           text_before_num='Pypi repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_PYPI_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_npm_repos(self):
        self._numeric_feat(feature_col='Npm repositories',
                           text_before_num='Account has',
                           text_after_num='Npm repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_NPM_REPOSITORIES'])

    def _npm_growth_monthly(self):
        self._numeric_feat(feature_col='Npm monthly incline',
                           text_before_num='Npm repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_NPM_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _npm_growth_quarterly(self):
        self._numeric_feat(feature_col='Npm quarterly incline',
                           text_before_num='Npm repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_NPM_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_gradle_repos(self):
        self._numeric_feat(feature_col='Gradle repositories',
                           text_before_num='Account has',
                           text_after_num='Gradle repositories',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_GRADLE_REPOSITORIES'])

    def _gradle_growth_monthly(self):
        self._numeric_feat(feature_col='Gradle monthly incline',
                           text_before_num='Gradle repositories in JFrog platform grew by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_GRADLE_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _gradle_growth_quarterly(self):
        self._numeric_feat(feature_col='Gradle quarterly incline',
                           text_before_num='Gradle repositories in JFrog platform grew by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_GRADLE_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_distinct_repos(self):
        self._numeric_feat(feature_col='Number of technologies',
                           text_before_num='Account has',
                           text_after_num='repository types',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_TECHNOLOGIES'])

    def _n_distinct_repos_growth_monthly(self):
        self._numeric_feat(feature_col='Number of technologies monthly incline',
                           text_before_num='Account repo types increased by',
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_TECHNOLOGIES_IN_THE_LAST_3_MONTHS'])

    def _n_distinct_repos_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of technologies quarterly incline',
                           text_before_num='Account repo types increased by',
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_TECHNOLOGIES_IN_THE_LAST_3_QUARTERS'])

    def _n_repos_growth_monthly(self):
        self._numeric_feat(feature_col='Number of repositories monthly incline',
                           text_before_num="Account's number of repositories increased by",
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_REPOSITORIES_IN_THE_LAST_3_MONTHS'])

    def _n_repos_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of repositories quarterly incline',
                           text_before_num="Account's number of repositories increased by",
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_REPOSITORIES_IN_THE_LAST_3_QUARTERS'])

    def _n_env_growth_monthly(self):
        self._numeric_feat(feature_col='Number of environments monthly incline',
                           text_before_num="Account's number of environments increased by",
                           text_after_num='in the last 3 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ENVIRONMENTS_IN_THE_LAST_3_MONTHS'])

    def _n_env_growth_quarterly(self):
        self._numeric_feat(feature_col='Number of environments quarterly incline',
                           text_before_num="Account's number of environments increased by",
                           text_after_num='in the last 3 quarters',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['GROWTH_IN_THE_NUMBER_OF_ENVIRONMENTS_IN_THE_LAST_3_QUARTERS'])

    def _industry_group(self):
        self._categorical_feat(feature_col='Industry',
                               text_before_cat='Account is operating on the',
                               text_after_cat='industry',
                               short_insight=self.SHORT['HIGH_SCALE'])

    def _revenue_range(self):
        self._categorical_feat(feature_col='Revenue range',
                               text_before_cat="Account's revenue range is between",
                               text_after_cat='',
                               short_insight=self.SHORT['HIGH_SCALE'])

    def _n_cases_year(self):
        self._numeric_feat(feature_col='Number of cases in the last year',
                           text_before_num='Account had',
                           text_after_num='support case(s) in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'],
                           medium_insight=self.MEDIUM['NUMBER_OF_SUPPORT_CASES_IN_THE_LAST_YEAR'])

    def _n_cases_quarter(self):
        self._numeric_feat(feature_col='Number of cases in the last 3 months',
                           text_before_num='Account had',
                           text_after_num='support case(s) in the last 3 months',
                           short_insight=self.SHORT['SUPPORT'],
                           medium_insight=self.MEDIUM['NUMBER_OF_SUPPORT_CASES_IN_THE_LAST_3_MONTHS'])

    def _n_sessions(self):
        self._numeric_feat(feature_col='Number of sessions last year',
                           text_before_num='Account had',
                           text_after_num='technical session(s) in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'],
                           medium_insight=self.MEDIUM['NUMBER_OF_TECHNICAL_SESSIONS_IN_THE_LAST_YEAR'])

    def _n_xray_sessions(self):
        self._numeric_feat(feature_col='Number of Xray sessions last year',
                           text_before_num='Account had',
                           text_after_num='Xray technical session(s) in the last 12 months',
                           short_insight=self.SHORT['SUPPORT'],
                           medium_insight=self.MEDIUM['NUMBER_OF_XRAY_TECHNICAL_SESSIONS_IN_THE_LAST_YEAR'])

    def _n_trials(self):
        self._numeric_feat(feature_col='Number of trials in the last year',
                           text_before_num='Account started',
                           text_after_num='trials in the last 12 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_TRIALS_IN_THE_LAST_YEAR'])

    def _n_ent_trials(self):
        self._numeric_feat(feature_col='Number of Enterprise trials in the last year',
                           text_before_num='Account started',
                           text_after_num='Enterprise trials in the last 12 months',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['NUMBER_OF_ENT_TRIALS_IN_THE_LAST_YEAR'])

    def _n_contacts(self):
        self._numeric_feat(feature_col='Number of contacts',
                           text_before_num="Account's number of contact(s) is",
                           text_after_num='',
                           short_insight=self.SHORT['SUPPORT'],
                           medium_insight=self.MEDIUM['NUMBER_OF_CONTACT'])

    def _n_engineers_normalized(self):
        self._numeric_feat(feature_col='Proportion of engineers detected by Zoom-info',
                           text_before_num='Account ratio between engineers to total employees is',
                           text_after_num='',
                           short_insight=self.SHORT['HIGH_SCALE'])

    def _n_devops_engineers_normalized(self):
        self._numeric_feat(feature_col='Proportion of devops engineers detected by Zoom-info',
                           text_before_num='Account ratio between devops engineers to total employees is',
                           text_after_num='',
                           short_insight=self.SHORT['HIGH_SCALE'])

    def _n_developers_normalized(self):
        self._numeric_feat(feature_col='Proportion of developers detected by Zoom-info',
                           text_before_num='Account ratio between developers to total employees is',
                           text_after_num='',
                           short_insight=self.SHORT['HIGH_SCALE'])

    def _seniority(self):
        self._numeric_feat(feature_col='Seniority in JFrog',
                           text_before_num="Account's seniority as JFrog customer is",
                           text_after_num='months',
                           short_insight=self.SHORT['HIGH_SCALE'])

    def _company_age(self):
        self._numeric_feat(feature_col='Company age (years)',
                           text_before_num="Account's company age is",
                           text_after_num='years',
                           short_insight=self.SHORT['HIGH_SCALE'])

    def _n_active_contracts(self):
        self._numeric_feat(feature_col='Number of active contracts',
                           text_before_num="Account has",
                           text_after_num='active contracts',
                           short_insight=self.SHORT['HIGH_SCALE'],
                           medium_insight=self.MEDIUM['NUMBER_OF_CONTRACTS'])

    def _total_employees_range(self):
        self._categorical_feat(feature_col='Number of employees (range)',
                               text_before_cat='Account has between',
                               text_after_cat='employees',
                               short_insight=self.SHORT['HIGH_SCALE'])

    def _leading_tech(self):
        # - without relation
        self._categorical_feat(feature_col='Leading technology',
                               text_before_cat='Account most used technology is',
                               text_after_cat='',
                               short_insight=self.SHORT['PLATFORM'])

    def _company_type(self):
        # - without relation
        self._categorical_feat(feature_col='Company type',
                               text_before_cat="The account's company is",
                               text_after_cat='',
                               short_insight=self.SHORT['HIGH_SCALE'])

    def _is_cotermed(self):
        self._binary_feat(feature_col='Is previously co-termed?',
                          text_before_value='A previous contract of the account was',
                          text_after_value='co-termed',
                          short_insight=self.SHORT['SALES'])

    def _avg_resolution_days(self):
        self._numeric_feat(feature_col='Average resolution days for a case in the last year',
                           text_before_num='The average resolution days for a case is',
                           text_after_num='in the last 12 months',
                           short_insight=self.SHORT['SALES'],
                           medium_insight=self.MEDIUM['RESOLVE_DAYS'])

    def __days_from_contact_added(self):
        self._numeric_feat(feature_col='Number of days since last contact added',
                           text_before_num='',
                           text_after_num='days passed since the last contact added',
                           short_insight=self.SHORT['SALES'],
                           medium_insight=self.MEDIUM['DAYS_FROM_LAST_CONTACT'])

    def __days_from_artifacts_size_changed(self):
        self._numeric_feat(feature_col='Number of days since artifacts size changed',
                           text_before_num='',
                           text_after_num='of days passed since artifacts size changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_artifacts_counts_changed(self):
        self._numeric_feat(feature_col='Number of days since artifacts count changed',
                           text_before_num='',
                           text_after_num='of days passed since artifacts count changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_binaries_size_changed(self):
        self._numeric_feat(feature_col='Number of days since binaries size changed',
                           text_before_num='',
                           text_after_num='of days passed since binaries size changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_binaries_counts_changed(self):
        self._numeric_feat(feature_col='Number of days since binaries count changed',
                           text_before_num='',
                           text_after_num='of days passed since binaries count changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_n_users_changed(self):
        self._numeric_feat(feature_col='Number of days since number of users changed',
                           text_before_num='',
                           text_after_num='of days passed since number of users changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_n_permissions_changed(self):
        self._numeric_feat(feature_col='Number of days since number of permissions changed',
                           text_before_num='',
                           text_after_num='of days passed since number of permissions changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def __days_from_internal_groups_changed(self):
        self._numeric_feat(feature_col='Number of days since number of internal groups changed',
                           text_before_num='',
                           text_after_num='of days passed since number of internal groups changed',
                           short_insight=self.SHORT['PLATFORM'],
                           medium_insight=self.MEDIUM['PLATFORM_ACTIVITY'])

    def _mapping(self):
        return {
            'Number of users': self._n_users(),
            'Number of repositories': self._n_repos(),
            'Number of environments': self._n_env(),
            'Number of permissions': self._n_permissions(),
            'Number of internal groups': self._n_internal_groups(),
            'Storage: artifacts count': self._n_artifacts_count(),
            'Storage: artifacts size': self._total_artifacts_size(),
            'Storage: binaries size': self._total_binaries_size(),
            'Storage: binaries count': self._n_binaries_count(),
            'Storage: items count': self._n_items_count(),
            'Number of users monthly incline': self._n_users_growth_monthly(),
            'Number of users quarterly incline': self._n_users_growth_quarterly(),
            'Number of permissions monthly incline': self._n_permissions_growth_monthly(),
            'Number of permissions quarterly incline': self._n_permissions_growth_quarterly(),
            'Number of internal groups monthly incline': self._n_internal_groups_growth_monthly(),
            'Number of internal groups quarterly incline': self._n_internal_groups_growth_quarterly(),
            'Docker repositories': self._n_docker_repos(),
            'Maven repositories': self._n_maven_repos(),
            'Generic repositories': self._n_generic_repos(),
            'Pypi repositories': self._n_pypi_repos(),
            'Nuget repositories': self._n_nuget_repos(),
            'Npm repositories': self._n_npm_repos(),
            'Number of technologies': self._n_distinct_repos(),
            'Average resolution days for a case in the last year': self._avg_resolution_days(),
            'Number of technologies monthly incline': self._n_distinct_repos_growth_monthly(),
            'Number of technologies quarterly incline': self._n_distinct_repos_growth_quarterly(),

            'Industry': self._industry_group(),
            'Company type': self._company_type(),
            'Revenue range': self._revenue_range(),
            'Number of cases in the last year': self._n_cases_year(),
            'Number of cases in the last 3 months': self._n_cases_quarter(),
            'Number of sessions in the last year': self._n_sessions(),
            'Number of Xray sessions in the last year'
            'Number of days since last contact added': self.__days_from_contact_added(),
            'Number of contacts': self._n_contacts(),
            'Proportion of engineers detected by Zoom-info': self._n_engineers_normalized(),
            'Proportion of devops engineers detected by Zoom-info': self._n_devops_engineers_normalized(),
            'Proportion of developers detected by Zoom-info': self._n_developers_normalized(),
            'Seniority in JFrog (months)': self._seniority(),
            'Company age (years)': self._company_age(),
            'Number of active contracts': self._n_active_contracts(),
            'Number of trials in the last year': self._n_trials(),
            'Number of Enterprise trials in the last year': self. _n_ent_trials(),
            'Number of employees (range)': self._total_employees_range(),
            'Leading technology': self._leading_tech(),
            'Is previously co-termed?': self._is_cotermed(),
            'Storage: artifacts count monthly incline': self._artifacts_count_growth_monthly(),
            'Storage: artifacts count quarterly incline': self._artifacts_count_growth_quarterly(),
            'Storage: binaries count monthly incline': self._binaries_count_growth_monthly(),
            'Storage: binaries count quarterly incline': self._binaries_count_growth_quarterly(),
            'Storage: items count monthly incline': self._items_count_growth_monthly(),
            'Storage: items count quarterly incline': self._items_count_growth_quarterly(),
            'Storage: artifacts size monthly incline': self._artifacts_size_growth_monthly(),
            'Storage: artifacts size quarterly incline': self._artifacts_size_growth_quarterly(),
            'Storage: binaries size monthly incline': self._binaries_size_growth_monthly(),
            'Storage: binaries size quarterly incline': self._binaries_size_growth_quarterly(),
            'Number of environments monthly incline': self._n_env_growth_monthly(),
            'Number of environments quarterly incline': self._n_env_growth_quarterly(),
            'Number of repositories monthly incline': self._n_repos_growth_monthly(),
            'Number of repositories quarterly incline': self._n_repos_growth_quarterly(),
            'Docker monthly incline': self._docker_growth_monthly(),
            'Docker quarterly incline': self._docker_growth_quarterly(),
            'Generic monthly incline': self._generic_growth_monthly(),
            'Generic quarterly incline': self._generic_growth_quarterly(),
            'Maven monthly incline': self._maven_growth_monthly(),
            'Maven quarterly incline': self._maven_growth_quarterly(),
            'Npm monthly incline': self._npm_growth_monthly(),
            'Npm quarterly incline': self._npm_growth_quarterly(),
            'Pypi monthly incline': self._pypi_growth_monthly(),
            'Pypi quarterly incline': self._pypi_growth_quarterly(),
            'Nuget monthly incline': self._nuget_growth_monthly(),
            'Nuget quarterly incline': self._nuget_growth_quarterly(),
            'Number of days since artifacts size changed': self.__days_from_artifacts_size_changed(),
            'Number of days since artifacts count changed': self.__days_from_artifacts_counts_changed(),
            'Number of days since binaries count changed': self.__days_from_binaries_counts_changed(),
            'Number of days since binaries size changed': self._binaries_size_growth_monthly(),
            'Number of days since number of users changed': self.__days_from_n_users_changed(),
            'Number of days since number of permissions changed': self.__days_from_n_permissions_changed(),
            'Number of days since number of internal groups changed': self.__days_from_internal_groups_changed(),
            'Number of times enterprise mentioned in sessions': self._n_ent_mentioned_sessions(),
            'Number of times high-availability mentioned in sessions': self._n_ha_mentioned_sessions(),
            'Number of times xray mentioned in sessions': self._n_xray_mentioned_sessions(),
            'Number of times competitors mentioned in sessions': self._n_competitor_mentioned_sessions(),
        }

