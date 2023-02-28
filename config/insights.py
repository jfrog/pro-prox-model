from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Insights:
    def to_dict(self):
        return asdict(self)


@dataclass
class ShortInsights(Insights):
    PLATFORM: str = "Platform usage patterns"
    SUPPORT: str = "Level of interaction with support"
    SALES: str = "Interactions with sales/support"
    MARKETING: str = "Interactions with JFrog marketing/events"
    WEBSITE: str = "Interactions with JFrog website"
    HIGH_SCALE: str = "High-scale account business profile"


@dataclass
class MediumInsights:
    def __init__(self):
        self._initialize_all_feats()
        self.NUMBER_OF_VISITS_PRIVATE = 'High interest in JFrog web materials/pages; can indicate potential upsell'
        self.NUMBER_OF_VISITS_PUBLIC = 'High interest in JFrog web materials/pages; can indicate potential upsell'
        self.NUMBER_OF_CONTRACTS = 'High number of active contracts with the account; can indicate potential upsell'
        self.NUMBER_OF_CONTACT = 'High number of contacts; can be leveraged to a potential upsell'
        self.DAYS_FROM_LAST_CONTACT = 'New contact for the account has been added recently; can indicate potential upsell opportunity'
        self.PLATFORM_ACTIVITY = 'High activity in JFrog platform detected in the last few days'
        self.RESOLVE_DAYS = 'The account experienced short support cases resolution period in the last year; can indicate they are satisfied from JFrog support'

    def to_dict(self):
        return vars(self)

    def _set_high_value_detection(self, name: str, value: str, is_jfrog_platform: bool,
                                  period: str, indication: str):
        jfrog_platform_text = ' in JFrog platform' if is_jfrog_platform else ''
        setattr(
            self,
            name,
            f"High {value} detected{jfrog_platform_text}{period}; can indicate {indication}",
        )

    def _initialize_all_feats(self):
        self.EMPTY = ""
        values_12m = [
            "number of support cases",
            "number of technical sessions",
            "number of xray technical sessions",
            "number of trials",
            "number of ent trials",
            "number of training sessions"
        ]
        period_12m = " in the last year"
        values_3m = [
            "number of support cases",
            "growth in the number of permissions",
            "growth in the number of users",
            "growth in the number of internal groups",
            "growth in the number of binaries",
            "growth in the binaries size",
            "growth in the number of artifacts",
            "growth in the artifacts size",
            "growth in the number of artifacts",
            "growth in the number of items",
            "growth in the number of Docker repositories",
            "growth in the number of Maven repositories",
            "growth in the number of Generic repositories",
            "growth in the number of Nuget repositories",
            "growth in the number of Pypi repositories",
            "growth in the number of Npm repositories",
            "growth in the number of Gradle repositories",
            "growth in the number of technologies",
            "growth in the number of repositories",
            "growth in the number of environments"
        ]
        period_3m = " in the last 3 months"
        values_3q = [
            "growth in the number of permissions",
            "growth in the number of users",
            "growth in the number of internal groups",
            "growth in the number of binaries",
            "growth in the binaries size",
            "growth in the number of artifacts",
            "growth in the artifacts size",
            "growth in the number of artifacts",
            "growth in the number of items",
            "growth in the number of Docker repositories",
            "growth in the number of Maven repositories",
            "growth in the number of Generic repositories",
            "growth in the number of Nuget repositories",
            "growth in the number of Pypi repositories",
            "growth in the number of Npm repositories",
            "growth in the number of Gradle repositories",
            "growth in the number of technologies",
            "growth in the number of repositories",
            "growth in the number of environments"
        ]
        period_3q = " in the last 3 quarters"

        values_no_period = ["number of permissions",
                            "number of users",
                            "number of internal groups",
                            "number of binaries",
                            "binaries size",
                            "number of artifacts",
                            "artifacts size",
                            "number of artifacts",
                            "number of items",
                            "number of Docker repositories",
                            "number of Maven repositories",
                            "number of Generic repositories",
                            "number of Nuget repositories",
                            "number of Debian repositories",
                            "number of Pypi repositories",
                            "number of Gradle repositories",
                            "number of Npm repositories",
                            "number of technologies",
                            "number of repositories",
                            "number of replications",
                            "number of environments"]
        empty_period = ""

        value_period_to_initiate = zip(
            [values_12m, values_3m, values_3q, values_no_period],
            [period_12m, period_3m, period_3q, empty_period],
        )

        for values_no_period, period in value_period_to_initiate:
            self._initialize_period_feats(values_no_period, period)

    def _initialize_period_feats(self, values: List[str], period: str):
        for value in values:
            name_period_suffix = (
                "_" + self._get_variable_name(period) if period != "" else ""
            )
            name = self._get_variable_name(value) + name_period_suffix
            not_jfrog_platform_ls = ['training', 'sessions', 'trials', 'cases']
            is_not_jfrog_platform = any([x in value for x in not_jfrog_platform_ls])
            high_adoption_ls = ['repositories', 'training', 'permissions', 'internal', 'technologies']
            is_high_adoption = any([x in value for x in high_adoption_ls])
            indication = "high adoption" if is_high_adoption else "potential upsell"
            self._set_high_value_detection(name=name, value=value, period=period,
                                           is_jfrog_platform=not is_not_jfrog_platform,
                                           indication=indication)

    @staticmethod
    def _get_variable_name(value: str) -> str:
        return "_".join(value.split()).upper()


