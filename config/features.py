from dataclasses import dataclass
from typing import Tuple


@dataclass
class Features:
    REPORT_IGNORE: Tuple = ('n_calls', 'total_seconds_in_calls', 'n_stages_in_calls', 'n_trackers',
                            'n_distinct_trackers', 'n_differentiation_trackers', 'n_features_trackers',
                            'n_subscription_trackers', 'n_distribution_trackers', 'n_advanced_security_trackers',
                            'n_pricing_trackers', 'n_value_prop_trackers', 'n_topics', 'n_distinct_topics',
                            'n_small_talk_topics', 'n_pricing_topics', 'n_call_setup_topics',
                            'n_ha_mentioned_sessions', 'n_ent_mentioned_sessions', 'n_competitor_mentioned_sessions',
                            'n_xray_mentioned_sessions', 'is_cotermed', 'days_since_xray_task', 'n_sent',
                            'days_since_reply', 'days_since_sent', 'leading_tech', 'n_security_contacts',
                            'pricing_views', 'artifactory_views', 'xray_views', 'support_views', 'knowledge_views')
    REPORT_IGNORE_IF_NOT_HIGH: Tuple = ('artifacts_size', 'artifacts_count', 'binaries_size', 'binaries_count',
                                        'items_count', 'number_of_permissions', 'internal_groups', 'number_of_users',
                                        'maven', 'generic', 'docker', 'npm', 'pypi', 'gradle', 'nuget',
                                        'n_ent_trials', 'n_contacts', 'n_trials', 'n_active_contracts',
                                        'n_sessions_last_year', 'n_xray_sessions_last_year', 'n_cases_last_year',
                                        'n_repos', 'n_env', 'n_tech', 'n_env_monthly_growth', 'n_env_quarter_growth',
                                        'n_tech_monthly_growth', 'n_tech_quarter_growth', 'generic_monthly_growth',
                                        'gradle_monthly_growth', 'npm_monthly_growth', 'nuget_monthly_growth',
                                        'maven_monthly_growth', 'docker_monthly_growth', 'pypi_monthly_growth',
                                        'artifacts_count_monthly_growth', 'artifacts_size_monthly_growth',
                                        'binaries_count_monthly_growth', 'binaries_size_monthly_growth',
                                        'items_count_monthly_growth', 'number_of_users_monthly_growth',
                                        'n_repos_monthly_growth', 'number_of_permissions_monthly_growth',
                                        'internal_groups_monthly_growth', 'generic_quarter_growth',
                                        'maven_quarter_growth', 'docker_quarter_growth',
                                        'npm_quarter_growth', 'gradle_quarter_growth',
                                        'nuget_quarter_growth', 'pypi_quarter_growth', 'n_cases_last_3_months',
                                        'artifacts_count_quarter_growth', 'artifacts_size_quarter_growth',
                                        'binaries_count_quarter_growth', 'binaries_size_quarter_growth',
                                        'items_count_quarter_growth', 'number_of_users_quarter_growth',
                                        'n_repos_quarter_growth', 'number_of_permissions_quarter_growth',
                                        'internal_groups_quarter_growth',)
    REPORT_IGNORE_IF_NOT_LOW: Tuple = ('days_from_contact_added', 'avg_resolution_days','days_from_artifacts_size_change',
                                       'days_from_artifacts_count_change', 'days_from_binaries_size_change', 'days_from_binaries_count_change',
                                       'days_from_items_count_change', 'days_from_permissions_change', 'days_from_internal_groups_change',
                                       'days_from_users_change')
