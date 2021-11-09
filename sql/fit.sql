select accountid account_id,territory,closedate::DATE - INTERVAL '90 day' as relevant_date,
case when new_product not in ('JFrog Pro', 'JFrog Pro - New Pricing') then 1 else 0 end as class into #base_accounts from (
------- base population
------ take won and closed from the opp table
(select distinct accountid, id as opp_id, top_product_id__c, product_type__c, original_product__c, iswon, isclosed, previous_arr__c, arr__c, arr_growth__c, type as opp_type, closedate
from salesforce.opportunity
where type != 'New Business' and iswon='true' and isclosed ='true' and closedate >= '2018-01-01') as opps
join
(select product_id,product_name old_product from dims.dim_products ---- add previous product name
where product_name in ('JFrog Pro - New Pricing', 'JFrog Pro')) as old_products
on opps.original_product__c = old_products.old_product
join
(select product_id, product_name new_product from dims.dim_products ---- add renewal product name
where product_family = 'Artifactory' and type = 'License' and product_team != 'None'
and product_name not in ('Support Services for JFrog Pro Edition') and lower(product_name) not like '%cloud%') as new_products
on opps.top_product_id__c = new_products.product_id
join (
select account_id,territory from dims.dim_accounts
) as da
on da.account_id = opps.accountid
)
order by accountid;

---------------------------------
-- Jira cases
---------------------------------
DROP TABLE IF EXISTS #jira_cases;

SELECT account_id,
       relevant_date,
       count(*) AS n_jira_cases,
       sum(CASE
               WHEN jira_case_resolution_status IN ('Wont Fix', 'Not Fixed', 'Will not implement', 'Unresolved', 'Wont Do') THEN 1
               ELSE 0
           END) AS unresolved_jira_cases INTO #jira_cases
FROM #base_accounts AS a
JOIN salesforce.account AS b ON a.account_id = left(b.accountid_full, LEN (b.accountid_full) -3)
JOIN salesforce.dim_jira_cases AS c ON b.name = c.jira_case_account_name
WHERE datediff('month', jira_created_date:: date, relevant_date) <= 12
  AND relevant_date >= jira_created_date:: date
GROUP BY 1,
         2;

---------------------------------
-- Account activity per package types features - for trends
---------------------------------
DROP TABLE IF EXISTS #repositories;

SELECT account_id,
       period_range,
       relevant_date,
       CLASS,
       count(*) AS n_env,
       sum(maven) AS maven,
       sum(generic) AS generic,
       sum(docker) AS docker,
       sum(npm) AS npm,
       sum(pypi) AS pypi,
       sum(gradle) AS gradle,
       sum(nuget) AS nuget--, sum(YUM) as YUM, sum(Helm) as Helm, sum(Gems) as Gems, sum(Debian) as Debian, sum(Ivy) as Ivy
INTO #repositories
FROM
  (SELECT environment_service_id,
          repo.account_id,
          relevant_date,
          CLASS,
          CASE
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -1) AND date_trunc('day', b.relevant_date) THEN '3 Months' -- Actually 1 month from relevant date
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -4) AND add_months(date_trunc('day', b.relevant_date), -3) THEN '4 Months' -- Actually 1 quarter from relevant date
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -7) AND add_months(date_trunc('day', b.relevant_date), -6) THEN '5 Months' -- Actually 2 quarters from relevant date
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -2) AND add_months(date_trunc('day', b.relevant_date), -1) THEN '6 Months' -- Actually 2 months from relevant date
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -3) AND add_months(date_trunc('day', b.relevant_date), -2) THEN '7 Months' -- Actually 3 months from relevant date
          END AS period_range,
          max(CASE
                  WHEN repo.package_type = 'Maven' THEN avg_repos
                  ELSE 0
              END) AS maven,
          max(CASE
                  WHEN repo.package_type = 'Generic' THEN avg_repos
                  ELSE 0
              END) AS generic,
           max(CASE
                   WHEN repo.package_type = 'Docker' THEN avg_repos
                   ELSE 0
               END) AS docker,
           max(CASE
                   WHEN repo.package_type = 'Npm' THEN avg_repos
                   ELSE 0
               END) AS npm,
           max(CASE
                   WHEN repo.package_type = 'Pypi' THEN avg_repos
                   ELSE 0
               END) AS pypi,
           max(CASE
                   WHEN repo.package_type = 'Gradle' THEN avg_repos
                   ELSE 0
               END) AS gradle,
           max(CASE
                   WHEN repo.package_type = 'NuGet' THEN avg_repos
                   ELSE 0
               END) AS nuget
--        max(case when repo.package_type = 'YUM' then avg_repos else 0 end) as YUM,
--        max(case when repo.package_type = 'Helm' then avg_repos else 0 end) as Helm,
--        max(case when repo.package_type = 'Gems' then avg_repos else 0 end) as Gems,
--        max(case when repo.package_type = 'Debian' then avg_repos else 0 end) as Debian,
--        max(case when repo.package_type = 'Ivy' then avg_repos else 0 end) as Ivy
FROM #base_accounts AS b
   LEFT JOIN artifactory.dwh_service_trends_repo repo ON repo.account_id = b.account_id
   WHERE period_range IS NOT NULL
   GROUP BY 1,
            2,
            3,
            4,
            5)
GROUP BY 1,
         2,
         3,
         4;

---------------------------------
-- Storage
---------------------------------
-- Same as above
DROP TABLE IF EXISTS #storage ;

/* Artifcats */
SELECT account_id,
       relevant_date,
       period_range,
       sum(artifacts_count) AS artifacts_count,
       sum(artifacts_size) AS artifacts_size,
       sum(binaries_count) AS binaries_count,
       sum(binaries_size) AS binaries_size,
       sum(items_count) AS items_count INTO #storage
FROM
  (SELECT a.account_id,
          a.environment_service_id,
          relevant_date,
          CASE
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -1) AND date_trunc('day', b.relevant_date) THEN '3 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -4) AND add_months(date_trunc('day', b.relevant_date), -3) THEN '4 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -7) AND add_months(date_trunc('day', b.relevant_date), -6) THEN '5 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -2) AND add_months(date_trunc('day', b.relevant_date), -1) THEN '6 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -3) AND add_months(date_trunc('day', b.relevant_date), -2) THEN '7 Months'
          END AS period_range,
          avg(avg_artifacts_count) AS artifacts_count,
          avg(avg_artifacts_size) AS artifacts_size,
          avg(avg_binaries_count) AS binaries_count,
          avg(avg_binaries_size) AS binaries_size,
          avg(avg_items_count) AS items_count --
FROM #base_accounts AS b
   LEFT JOIN artifactory.dwh_service_trends_summary_storage AS a ON a.account_id = b.account_id
   WHERE period_range IS NOT NULL
   GROUP BY 1,
            2,
            3,
            4)
GROUP BY 1,
         2,
         3;

---------------------------------
-- days from storage
---------------------------------
DROP TABLE IF EXISTS #storage_over_time;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      art_create_date,
      sum(avg_artifacts_count) AS artifacts_count,
      sum(avg_artifacts_size) AS artifacts_size,
      sum(avg_binaries_count) AS binaries_count,
      sum(avg_binaries_size) AS binaries_size,
      sum(avg_items_count) AS items_count INTO #storage_over_time
FROM #base_accounts AS a
LEFT JOIN artifactory.dwh_service_trends_summary_storage AS b ON a.account_id = b.account_id
WHERE art_create_date <= relevant_date
GROUP BY 1,
        2,
        3,
        4;


DROP TABLE IF EXISTS #days_from_artifacts_count;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_artifacts_count_change INTO #days_from_artifacts_count
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          artifacts_count AS curr_artifacts_count
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            artifacts_count,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE artifacts_count != curr_artifacts_count
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_binaries_count;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_binaries_count_change INTO #days_from_binaries_count
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          binaries_count AS curr_binaries_count
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            binaries_count,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE binaries_count != curr_binaries_count
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_artifacts_size;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_artifacts_size_change INTO #days_from_artifacts_size
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          artifacts_size AS curr_artifacts_size
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            artifacts_size,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE artifacts_size != curr_artifacts_size
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_binaries_size;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_binaries_size_change INTO #days_from_binaries_size
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          binaries_size AS curr_binaries_size
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            binaries_size,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE binaries_size != curr_binaries_size
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_artifacts_count;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_artifacts_count_change INTO #days_from_artifacts_count
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          artifacts_count AS curr_artifacts_count
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            artifacts_count,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE artifacts_count != curr_artifacts_count
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_items_count;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_items_count_change INTO #days_from_items_count
FROM #storage_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          items_count AS curr_items_count
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            items_count,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #storage_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE items_count != curr_items_count
GROUP BY 1,
        2,
        3;

-- Same as above
DROP TABLE IF EXISTS #users ;

/* Users */
SELECT account_id,
       period_range,
       relevant_date,
       sum(number_of_premissions) AS number_of_permissions,
       sum(internal_groups) AS internal_groups,
       sum(number_of_users) AS number_of_users INTO #users
FROM
  (SELECT u.account_id,
          u.environment_service_id,
          relevant_date,
          CASE
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -1) AND date_trunc('day', b.relevant_date) THEN '3 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -4) AND add_months(date_trunc('day', b.relevant_date), -3) THEN '4 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -7) AND add_months(date_trunc('day', b.relevant_date), -6) THEN '5 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -2) AND add_months(date_trunc('day', b.relevant_date), -1) THEN '6 Months'
              WHEN date_trunc('day', art_create_date) BETWEEN add_months(date_trunc('day', b.relevant_date), -3) AND add_months(date_trunc('day', b.relevant_date), -2) THEN '7 Months'
          END AS period_range,
          avg(number_of_permission_targets) AS number_of_premissions,
          avg(internal_groups) AS internal_groups,
          avg(number_of_users) AS number_of_users
   FROM #base_accounts AS b
   LEFT JOIN artifactory.dwh_service_trends_security AS u ON u.account_id = b.account_id
   WHERE period_range IS NOT NULL
   GROUP BY 1,
            2,
            3,
            4)
GROUP BY 1,
         2,
         3;

---------------------------------
-- days from users
---------------------------------
DROP TABLE IF EXISTS #users_over_time;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      art_create_date,
      sum(number_of_permission_targets) AS number_of_permissions,
      sum(internal_groups) AS internal_groups,
      sum(number_of_users) AS number_of_users INTO #users_over_time
FROM #base_accounts AS a
LEFT JOIN artifactory.dwh_service_trends_security AS b ON a.account_id = b.account_id
WHERE art_create_date <= relevant_date
GROUP BY 1,
        2,
        3,
        4;

DROP TABLE IF EXISTS #days_from_users;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_users_change INTO #days_from_users
FROM #users_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          number_of_users AS curr_number_of_users
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            number_of_users,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #users_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE number_of_users != curr_number_of_users
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_permissions;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_permissions_change INTO #days_from_permissions
FROM #users_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          number_of_permissions AS curr_number_of_permissions
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            number_of_permissions,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #users_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE number_of_permissions != curr_number_of_permissions
GROUP BY 1,
        2,
        3;

DROP TABLE IF EXISTS #days_from_internal_groups;

SELECT a.account_id,
      a.relevant_date,
      a.class,
      datediff('day', max(art_create_date), a.relevant_date) AS days_from_internal_groups_change INTO #days_from_internal_groups
FROM #users_over_time AS a
JOIN
  (SELECT account_id,
          relevant_date,
          internal_groups AS curr_internal_groups
  FROM
    (SELECT account_id,
            relevant_date,
            art_create_date,
            internal_groups,
            row_number() OVER (PARTITION BY a.account_id,
                                            a.relevant_date
                                ORDER BY art_create_date DESC) AS rn
      FROM #users_over_time AS a)
  WHERE rn = 1) AS b ON a.account_id = b.account_id
AND a.relevant_date = b.relevant_date
WHERE internal_groups != curr_internal_groups
GROUP BY 1,
        2,
        3;

---------------
-- Support
---------------
DROP TABLE IF EXISTS #support;

SELECT a.account_id,
      relevant_date,
      count(DISTINCT CASE
                          WHEN movement_date BETWEEN dateadd(MONTH, -12, a.relevant_date) AND a.relevant_date THEN cases.case_id
                      END) AS cases_within_last_year,
      count(DISTINCT CASE
                          WHEN movement_date BETWEEN dateadd(MONTH, -3, a.relevant_date) AND a.relevant_date THEN cases.case_id
                      END) AS cases_within_3_last_months INTO #support
FROM qoc.stg_events_measures AS EVENTS
LEFT JOIN salesforce.fact_cases AS cases ON cases.case_id = events.case_id
AND events.parameter_id IN (10)
INNER JOIN #base_accounts AS a ON cases.account_id = a.account_id
WHERE date_trunc('day', movement_date) BETWEEN add_months(date_trunc('month', a.relevant_date), -12) AND a.relevant_date
GROUP BY 1,
        2;

DROP TABLE IF EXISTS #poor_cases;

SELECT a.account_id,
      a.relevant_date,
      sum(CASE
              WHEN support_experience__c IN ('Poor', 'Extremely Poor')
                    OR cse_reason__c != '' THEN 1
              ELSE 0
          END) AS n_poor_cases INTO #poor_cases
FROM #base_accounts AS a
JOIN data_science.satisfaction_case_level AS b ON a.account_id = b.account_id
WHERE case_created_date BETWEEN add_months(relevant_date, -12) AND relevant_date
GROUP BY 1,
        2;

DROP TABLE IF EXISTS #cases_agg;

SELECT a.account_id,
      a.relevant_date,
      sum(CASE
              WHEN lower(subject) like '%xray%'
                    OR lower(subject) like '%scan%'
                    OR lower(subject) like '%security%' THEN 1
              ELSE 0
          END) AS n_security_subject,
      sum(CASE
              WHEN subject like '%HA%'
                    OR lower(subject) like '%high availability%' THEN 1
              ELSE 0
          END) AS n_ha_subject,
      sum(CASE
              WHEN lower(subject) like '%docker%' THEN 1
              ELSE 0
          END) AS n_docker_subject,
      avg(CASE
              WHEN status IN ('Closed', 'Resolved') THEN datediff('day', createddate, coalesce(resolved_time__c, closeddate))
          END) AS avg_resolution_days INTO #cases_agg
FROM #base_accounts AS a
LEFT JOIN salesforce.cases AS b ON a.account_id = b.accountid
WHERE createddate BETWEEN add_months(date_trunc('month', a.relevant_date), -12) AND a.relevant_date
GROUP BY 1,
        2;

-------------------------
-- Technical sessions
-------------------------
DROP TABLE IF EXISTS #technical_sessions;

SELECT s.account_id,
       relevant_date,
       count(*) AS total_sessions_past_year,
       sum(CASE
               WHEN lower(session_name) like '%xray%' THEN 1
               ELSE 0
           END) AS total_xray_sessions_past_year,
       sum(CASE
               WHEN primary_product__c = 'Xray'
                    OR secondary_products__c = 'Xray' THEN 1
               ELSE 0
           END) AS total_xray_sessions_past_year1 INTO #technical_sessions
FROM facts.fact_technical_sessions s
INNER JOIN #base_accounts AS a ON s.account_id = a.account_id
WHERE session_createddate BETWEEN add_months(date_trunc('month', a.relevant_date), -12) AND a.relevant_date
GROUP BY 1,
         2;

-------------------------
-- Training
-------------------------
DROP TABLE IF EXISTS #training;

SELECT account_id,
       relevant_date,
       count(distinct(createddate)) AS n_training INTO #training
FROM
  (SELECT *
   FROM salesforce.training__c
   WHERE recordtypeid='012w0000000R1Yp'
     AND stage__c != 'Unprovided') AS st
RIGHT JOIN #base_accounts AS ar ON st.account__c = ar.account_id
WHERE createddate BETWEEN add_months(date_trunc('month', ar.relevant_date), -24) AND ar.relevant_date
GROUP BY 1,
         2;

-------------------------
-- Trials
-------------------------
DROP TABLE IF EXISTS #trials;

SELECT a.account_id,
       a.relevant_date,
       count(DISTINCT createddate::date) AS n_trials,
       count(DISTINCT CASE
                          WHEN subscription_type__c = 'ENTERPRISE' THEN createddate::date
                      END) AS n_ent_trials INTO #trials
FROM #base_accounts AS a
LEFT JOIN salesforce.trial AS b ON a.account_id = b.account__c
WHERE createddate BETWEEN add_months(date_trunc('month', a.relevant_date), -24) AND a.relevant_date --and status__c not in ('BLACKLISTED', 'Cancelled')
GROUP BY 1,
         2;

-------------
--Contacts
-------------
DROP TABLE IF EXISTS #contacts;

SELECT accountid,
       relevant_date,
       count(distinct(createddate)) AS n_contacts,
       sum(CASE
               WHEN lower(title) like '%security%'
                    OR lower(title) like '%cyber%'
                    OR lower(title) like '%ciso%' THEN 1
               ELSE 0
           END) AS n_security_contacts,
       n_security_contacts::float/n_contacts::float AS security_contacts_prop INTO #contacts
FROM salesforce.contact AS c
RIGHT JOIN #base_accounts AS ar ON c.accountid = ar.account_id
WHERE createddate <= relevant_date
GROUP BY 1,
         2;

DROP TABLE IF EXISTS #days_from_contact;

SELECT account_id,
      relevant_date,
      CLASS,
      datediff('day', max(createddate), relevant_date) AS days_from_contact_added INTO #days_from_contact
FROM salesforce.contact AS c
RIGHT JOIN #base_accounts AS ar ON c.accountid = ar.account_id
WHERE createddate <= relevant_date
GROUP BY 1,
        2,
        3;

-------------
--Contracts
-------------
DROP TABLE IF EXISTS #contracts;

SELECT a.account_id,
       relevant_date,
       class,
       count(distinct(CASE
                          WHEN relevant_date <= enddate THEN startdate
                      END)) AS n_active_contracts,
       max(CASE
               WHEN status = 'Co-termed'
                    AND relevant_date >= enddate THEN 1
               ELSE 0
           END) AS is_cotermed,
       count(distinct(CASE
                          WHEN contract_value__c like '%Enterprise%'
                               AND relevant_date <= enddate THEN startdate
                      END)) AS count_ent,
       count(distinct(CASE
                          WHEN contract_value__c = 'JFrog Pro X'
                          AND relevant_date <= enddate THEN startdate
                      END)) AS count_prox,
       count(distinct(CASE
                          WHEN contract_value__c = 'JFrog Pro'
                          AND relevant_date <= enddate THEN startdate
                      END)) AS count_pro
INTO #contracts
FROM #base_accounts AS a
LEFT JOIN salesforce.contract AS c ON c.accountid = a.account_id
WHERE relevant_date >= date_trunc('month', startdate)
GROUP BY 1,
         2,
         3;

-- select * from #upsell_acounts where account_id='0012000000ol9la'
-- select class, case when count_prox > 0 then 1 else 0 end as morethen0prox, count(*) from #contracts group by 1,2

-------------
--QOE
-------------
DROP TABLE IF EXISTS #qoe;

SELECT d.account_id,
       qoe_score,
       relevant_date INTO #qoe
FROM salesforce.dwh_qoe_scores AS d
RIGHT JOIN #base_accounts AS ar ON d.account_id = ar.account_id
WHERE create_date_monthly = relevant_date
  AND is_fictive = 0;

-------------
--Xray
-------------
-- DROP TABLE IF EXISTS #xray;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       CASE
--           WHEN date_trunc('day',date) BETWEEN add_months(date_trunc('day', ab.relevant_date), -1) AND date_trunc('day', ab.relevant_date) THEN '3 Months'
--           WHEN date_trunc('day',date) BETWEEN add_months(date_trunc('day', ab.relevant_date), -4) AND add_months(date_trunc('day', ab.relevant_date), -3) THEN '4 Months'
--           WHEN date_trunc('day',date) BETWEEN add_months(date_trunc('day', ab.relevant_date), -7) AND add_months(date_trunc('day', ab.relevant_date), -6) THEN '5 Months'
--           WHEN date_trunc('day',date) BETWEEN add_months(date_trunc('day', ab.relevant_date), -2) AND add_months(date_trunc('day', ab.relevant_date), -1) THEN '6 Months'
--           WHEN date_trunc('day',date) BETWEEN add_months(date_trunc('day', ab.relevant_date), -3) AND add_months(date_trunc('day', ab.relevant_date), -2) THEN '7 Months'
--       END AS period_range,
--       avg(indexed_repositories_count) AS indexed_repos INTO #xray
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_indexed_resources AS c ON b._id = c._id
-- WHERE period_range IS NOT NULL
-- GROUP BY 1,
--         2,
--         3;

-- DROP TABLE IF EXISTS #indexed_repos_over_time;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       ab.class, date, sum(indexed_repositories_count) AS indexed_repos INTO #indexed_repos_over_time
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_indexed_resources AS c ON b._id = c._id
-- WHERE date <= relevant_date
-- GROUP BY 1,
--         2,
--         3,
--         4;

-- DROP TABLE IF EXISTS #days_from_indexed_repos;

-- SELECT a.account_id,
--       a.relevant_date,
--       a.class,
--       datediff('day', max(date), a.relevant_date) AS days_from_indexed_repos_change INTO #days_from_indexed_repos
-- FROM #indexed_repos_over_time AS a
-- JOIN
--   (SELECT account_id,
--           relevant_date,
--           indexed_repos AS curr_indexed_repos
--   FROM
--     (SELECT account_id,
--             relevant_date, date, indexed_repos,
--                                   row_number() OVER (PARTITION BY a.account_id,
--                                                                   a.relevant_date
--                                                     ORDER BY date DESC) AS rn
--       FROM #indexed_repos_over_time AS a)
--   WHERE rn = 1) AS b ON a.account_id = b.account_id
-- AND a.relevant_date = b.relevant_date
-- WHERE indexed_repos != curr_indexed_repos
-- GROUP BY 1,
--         2,
--         3;

-- DROP TABLE IF EXISTS #days_from_xray;

-- SELECT a.account_id,
--       relevant_date,
--       CLASS,
--       max(b.date) AS last_xray_date,
--       datediff('day', last_xray_date, relevant_date) AS days_from_last_xray INTO #days_from_xray
-- FROM #base_accounts AS a
-- LEFT JOIN
--   (SELECT account_id,
--           c.date
--   FROM xray.dim_xray_onprem AS a
--   JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
--   JOIN xray.fact_totals AS c ON b._id = c._id) AS b ON a.account_id = b.account_id
-- WHERE date <= relevant_date
-- GROUP BY 1,
--         2,
--         3;

-- DROP TABLE IF EXISTS #xray_policies;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       CASE
--           WHEN avg(assigned_policies_count) > 0 THEN 1
--           ELSE 0
--       END AS is_policies,
--       sum(CASE
--               WHEN policy_type = 'security'
--                     AND assigned_policies_count > 0 THEN 1
--               ELSE 0
--           END) AS total_security_policies INTO #xray_policies
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_policies AS c ON b._id = c._id
-- WHERE datediff('day', c.date, ab.relevant_date) <= 90
--   AND datediff('day', c.date, ab.relevant_date) >= 0
-- GROUP BY 1,
--         2;

-- DROP TABLE IF EXISTS #xray_watches;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       CASE
--           WHEN avg(watches_count) > 0 THEN 1
--           ELSE 0
--       END AS is_watches INTO #xray_watches
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_watches AS c ON b._id = c._id
-- WHERE datediff('day', c.date, ab.relevant_date) <= 90
--   AND datediff('day', c.date, ab.relevant_date) >= 0
-- GROUP BY 1,
--         2;

-- DROP TABLE IF EXISTS #xray_system_massage;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       CASE
--           WHEN sum(CASE
--                         WHEN system_messages_count > 0 THEN 1
--                         ELSE 0
--                     END) > 0 THEN 1
--           ELSE 0
--       END AS is_system_message INTO #xray_system_massage
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_system_health AS c ON b._id = c._id
-- WHERE datediff('day', c.date, ab.relevant_date) <= 90
--   AND datediff('day', c.date, ab.relevant_date) >= 0
-- GROUP BY 1,
--         2;

-- DROP TABLE IF EXISTS #xray_totals;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       avg(total_artifactories_count) AS total_artifactories_count,
--       avg(total_components_count) AS total_components_count,
--       avg(total_root_components_count) AS total_root_components_count,
--       avg(total_indexed_builds_count) AS total_indexed_builds_count,
--       avg(total_policies_count) AS total_policies_count,
--       avg(total_violations_count) AS total_violations_count,
--       avg(total_watches_count) AS total_watches_count INTO #xray_totals
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_totals AS c ON b._id = c._id
-- WHERE datediff('day', c.date, ab.relevant_date) <= 90
--   AND datediff('day', c.date, ab.relevant_date) >= 0
-- GROUP BY 1,
--         2;

-- DROP TABLE IF EXISTS #xray_totals_over_time;

-- SELECT ab.account_id,
--       ab.relevant_date,
--       ab.class, date, sum(total_policies_count) AS total_policies_count,
--                       sum(total_violations_count) AS total_violations_count,
--                       sum(total_watches_count) AS total_watches_count INTO #xray_totals_over_time
-- FROM #base_accounts AS ab
-- LEFT JOIN xray.dim_xray_onprem AS a ON ab.account_id = a.account_id
-- JOIN xray.mongo_attributes AS b ON a.product_id = b.environment_license_id
-- JOIN xray.fact_totals AS c ON b._id = c._id
-- WHERE date <= relevant_date
-- GROUP BY 1,
--         2,
--         3,
--         4;

-- DROP TABLE IF EXISTS #days_from_policies_count;

-- SELECT a.account_id,
--       a.relevant_date,
--       a.class,
--       datediff('day', max(date), a.relevant_date) AS days_from_policies_change INTO #days_from_policies_count
-- FROM #xray_totals_over_time AS a
-- JOIN
--   (SELECT account_id,
--           relevant_date,
--           total_policies_count AS curr_total_policies_count
--   FROM
--     (SELECT account_id,
--             relevant_date, date, total_policies_count,
--                                   row_number() OVER (PARTITION BY a.account_id,
--                                                                   a.relevant_date
--                                                     ORDER BY date DESC) AS rn
--       FROM #xray_totals_over_time AS a)
--   WHERE rn = 1) AS b ON a.account_id = b.account_id
-- AND a.relevant_date = b.relevant_date
-- WHERE total_policies_count != curr_total_policies_count
-- GROUP BY 1,
--         2,
--         3;

-- DROP TABLE IF EXISTS #days_from_violations_count;

-- SELECT a.account_id,
--       a.relevant_date,
--       a.class,
--       datediff('day', max(date), a.relevant_date) AS days_from_violations_change INTO #days_from_violations_count
-- FROM #xray_totals_over_time AS a
-- JOIN
--   (SELECT account_id,
--           relevant_date,
--           total_violations_count AS curr_total_violations_count
--   FROM
--     (SELECT account_id,
--             relevant_date, date, total_violations_count,
--                                   row_number() OVER (PARTITION BY a.account_id,
--                                                                   a.relevant_date
--                                                     ORDER BY date DESC) AS rn
--       FROM #xray_totals_over_time AS a)
--   WHERE rn = 1) AS b ON a.account_id = b.account_id
-- AND a.relevant_date = b.relevant_date
-- WHERE total_violations_count != curr_total_violations_count
-- GROUP BY 1,
--         2,
--         3;

-- DROP TABLE IF EXISTS #days_from_watches_count;

-- SELECT a.account_id,
--       a.relevant_date,
--       a.class,
--       datediff('day', max(date), a.relevant_date) AS days_from_watches_change INTO #days_from_watches_count
-- FROM #xray_totals_over_time AS a
-- JOIN
--   (SELECT account_id,
--           relevant_date,
--           total_watches_count AS curr_total_watches_count
--   FROM
--     (SELECT account_id,
--             relevant_date, date, total_watches_count,
--                                   row_number() OVER (PARTITION BY a.account_id,
--                                                                   a.relevant_date
--                                                     ORDER BY date DESC) AS rn
--       FROM #xray_totals_over_time AS a)
--   WHERE rn = 1) AS b ON a.account_id = b.account_id
-- AND a.relevant_date = b.relevant_date
-- WHERE total_watches_count != curr_total_watches_count
-- GROUP BY 1,
--         2,
--         3;

-------------
--Clear Bit
-------------
DROP TABLE IF EXISTS #cbit;

SELECT da.account_id,
       max(cbit__companycategoryindustrygroup__c) AS industry_group,
       max(cbit__companyfoundedyear__c) AS founded_year,
       max(cbit__companymetricsemployees__c) AS total_employees,
       max(cbit__companymetricsemployeesrange__c) AS total_employees_range,
       COUNT (DISTINCT cbit__email__c) AS total_employees_with_details,
             count(DISTINCT CASE
                                WHEN cbit__employmentsubrole__c IN ('software_engineer', 'web_engineer')
                                     OR cbit__employmenttitle__c IN ('Software Developer', 'Java Developer', 'Software Engineer') THEN cbit__email__c
                            END) AS developers,
             count(DISTINCT CASE
                                WHEN cbit__employmentsubrole__c = 'devops_engineer' THEN cbit__email__c
                            END) AS devops_engineers,
             count(DISTINCT CASE
                                WHEN cbit__employmentsubrole__c like '%engineer%'
                                     OR cbit__employmenttitle__c like '%engineer%' THEN id
                            END) AS engineers INTO #cbit
FROM salesforce.cbit__clearbit__c a
INNER JOIN dims.dim_contacts dc ON dc.cbit__clearbit__c = a.id
INNER JOIN dims.dim_accounts da ON da.account_id = dc.account_id
GROUP BY 1;

-------------------
--Google Analytics
-------------------
DROP TABLE IF EXISTS #key_pages_views;

SELECT account_id,
       relevant_date,
       max(pricing_views) AS pricing_views,
       max(artifactory_views) AS artifactory_views,
       max(xray_views) AS xray_views,
       max(support_views) AS support_views,
       max(knowledge_views) AS knowledge_views INTO #key_pages_views
FROM
  (SELECT a.account_id,
          b.contact_id,
          a.relevant_date,
          sum(CASE
                  WHEN pagepath like '%/pricing%' THEN pageviews
                  ELSE 0
              END) AS pricing_views,
          sum(CASE
                  WHEN pagepath like '%/artifactory%' THEN pageviews
                  ELSE 0
              END) AS artifactory_views,
          sum(CASE
                  WHEN pagepath like '%/xray%' THEN pageviews
                  ELSE 0
              END) AS xray_views,
          sum(CASE
                  WHEN pagepath like '%/support/%' THEN pageviews
                  ELSE 0
              END) AS support_views,
          sum(CASE
                  WHEN pagepath like '%/knowledge%' THEN pageviews
                  ELSE 0
              END) AS knowledge_views
   FROM #base_accounts AS a
   LEFT JOIN dims.dim_contacts AS b ON a.account_id = b.account_id
   JOIN google_analytics.jfrogcom_sessions AS c ON b.ga_d_id__c = c.device_id
   JOIN google_analytics.jfrogcom_pageviews AS d ON d.session_id = c.session_id
   WHERE datediff(DAY, d.date, a.relevant_date) >= 0
     AND datediff(DAY, d.date, a.relevant_date) <= 90
   GROUP BY 1,
            2,
            3)
GROUP BY 1,
         2;

--------------------
--Cloud subscription
--------------------
DROP TABLE IF EXISTS #cloud_sub;

SELECT a.account_id,
       a.relevant_date,
       max(CASE
               WHEN c.owner_email IS NOT NULL THEN 1
               ELSE 0
           END) AS have_cloud_subscription INTO #cloud_sub
FROM #base_accounts AS a
JOIN salesforce.contact AS b ON a.account_id = b.accountid
LEFT JOIN dims.dim_cloud_servers AS c ON b.email = c.owner_email
WHERE aols_creation_date <= relevant_date
  AND server_type != 'Trial'
GROUP BY 1,
         2;

---------------------------------
-- triggers
---------------------------------
DROP TABLE IF EXISTS #triggers_cases;

SELECT a.account_id,
      a.relevant_date,
      count(distinct(CASE
                          WHEN keyword IN ('high availability', 'high-availability', ' ha') THEN created_date
                      END)) AS n_ha_mentioned_cases,
      count(distinct(CASE
                          WHEN keyword IN ('balancer', 'balancing', 'balance') THEN created_date
                      END)) AS n_bal_mentioned_cases,
      count(distinct(CASE
                          WHEN keyword = 'enterprise' THEN created_date
                      END)) AS n_ent_mentioned_cases,
      count(distinct(CASE
                          WHEN keyword = 'multiple' THEN created_date
                      END)) AS n_mul_mentioned_cases,
      count(distinct(CASE
                          WHEN keyword = 'replications' THEN created_date
                      END)) AS n_rep_mentioned_cases,
      count(distinct(CASE
                          WHEN keyword IN ('disaster recovery', 'downtime', ' dr', 'down time', 'bad performance') THEN created_date
                      END)) AS n_bad_mentioned_cases INTO #triggers_cases
FROM data_science.triggered_instances_cases AS tc
JOIN #base_accounts AS a ON tc.account_id = a.account_id
WHERE to_date(created_date, 'MM/DD/YYYY') BETWEEN add_months(relevant_date, -4) AND relevant_date
GROUP BY 1,
        2;

DROP TABLE IF EXISTS #triggers_sessions;

SELECT a.account_id,
      relevant_date,
      count(distinct(CASE
                          WHEN term = 'high availability' THEN instance_date
                      END)) AS n_ha_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'enterprise' THEN instance_date
                      END)) AS n_ent_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'multiple' THEN instance_date
                      END)) AS n_mul_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'balance' THEN instance_date
                      END)) AS n_bal_mentioned_sessions,
      count(distinct(CASE
                          WHEN term IN ('disaster recovery', 'downtime', 'bad performance') THEN instance_date
                      END)) AS n_bad_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'replications' THEN instance_date
                      END)) AS n_rep_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'Budget' THEN instance_date
                      END)) AS n_budget_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'churn' THEN instance_date
                      END)) AS n_churn_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'Competitor' THEN instance_date
                      END)) AS n_competitor_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'compliance' THEN instance_date
                      END)) AS n_compliance_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'CVE' THEN instance_date
                      END)) AS n_cve_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'Cyber' THEN instance_date
                      END)) AS n_cyber_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'downsell' THEN instance_date
                      END)) AS n_downsell_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'Scanning' THEN instance_date
                      END)) AS n_scanning_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'SLA' THEN instance_date
                      END)) AS n_sla_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'support' THEN instance_date
                      END)) AS n_support_mentioned_sessions,
      count(distinct(CASE
                          WHEN term = 'Xray' THEN instance_date
                      END)) AS n_xray_mentioned_sessions INTO #triggers_sessions
FROM #base_accounts AS a
LEFT JOIN data_science.termtriggers_new_version AS b ON a.account_id = b.account_id
WHERE TYPE IN ('session_Background',
              'session_Session Summary')--type in ('task_Comments', 'task_Subject')
AND to_date(instance_date, 'MM/DD/YYYY') BETWEEN add_months(relevant_date, -4) AND relevant_date
GROUP BY 1,
        2;

---------------------------------
-- tasks
---------------------------------
DROP TABLE IF EXISTS #emails;

SELECT a.account_id,
       a.relevant_date,
       count(distinct(CASE
                          WHEN salesloft1__salesloft_type__c = 'Reply' THEN subject
                      END)) AS n_replys,
       count(distinct(CASE
                          WHEN salesloft1__salesloft_type__c = 'Email'
                               OR subject like '%Message Sent%' THEN subject
                      END)) AS n_sent,
       count(distinct(CASE
                          WHEN salesloft1__salesloft_type__c = 'Call' THEN subject
                      END)) AS n_calls,
       sum(CASE
               WHEN lower(subject) like '%xray%' THEN 1
               ELSE 0
           END) AS n_task_xray,
       CASE
           WHEN n_sent > 0 THEN n_replys::float/n_sent::float
           ELSE -1
       END AS replys_to_sent INTO #emails
FROM #base_accounts AS a
JOIN salesforce.task AS b ON a.account_id = b.accountid
WHERE createddate BETWEEN add_months(relevant_date, -4) AND relevant_date
GROUP BY 1,
         2;

DROP TABLE IF EXISTS #days_sicne_reply;

SELECT account_id,
       relevant_date,
       datediff('day', max(CASE
                               WHEN salesloft1__salesloft_type__c = 'Reply' THEN createddate
                           END), relevant_date) AS days_since_reply,
       datediff('day', max(CASE
                               WHEN salesloft1__salesloft_type__c = 'Email'
                                    OR subject like '%Message Sent%' THEN createddate
                           END), relevant_date) AS days_since_sent,
       datediff('day', max(CASE
                               WHEN lower(subject) like '%xray%' THEN createddate
                           END), relevant_date) AS days_since_xray_task INTO #days_sicne_reply
FROM #base_accounts AS a
JOIN salesforce.task AS b ON a.account_id = b.accountid
WHERE createddate <= relevant_date
GROUP BY 1,
         2;

---------------
-- Main
---------------
SELECT DISTINCT a.account_id,
                a.relevant_date,
                a.class,
                CASE
                    WHEN (a.territory = 'None' OR a.territory IS NULL OR a.territory = '') THEN 'unknown'
                    ELSE a.territory
                END AS territory,
                b1.*,
                b2.*,
                b3.*,
                b4.*,
                b5.*,
                u1.*,
                u2.*,
                u3.*,
                u4.*,
                u5.*,
                c1.*,
                c2.*,
                c3.*,
                c4.*,
                c5.*,
                -- xr1.*,
                -- xr2.*,
                -- xr3.*,
                -- xr4.*,
                -- xr5.*,
                coalesce(n_contacts, 0) AS n_contacts,
                cr.*,
                coalesce(n_security_contacts, 0) AS n_security_contacts,
                -- coalesce(months_from_upgrade, -1) AS months_from_upgrade,
                coalesce(total_sessions_past_year, 0) AS n_sessions_last_year,
                coalesce(cases_within_last_year, 0) AS n_cases_last_year,
                coalesce(total_xray_sessions_past_year, 0) AS n_xray_sessions_last_year,
                coalesce(cases_within_3_last_months, 0) AS n_cases_last_3_months,
                coalesce(n_poor_cases, 0) AS n_poor_cases,
                cb.engineers, cb.devops_engineers, cb.developers, cb.total_employees_with_details,
 CASE
     WHEN cb.industry_group = ''
          OR cb.industry_group IS NULL THEN 'unknown'
     ELSE industry_group
 END,
 CASE
     WHEN (total_employees_range = ''
           OR total_employees_range IS NULL) THEN 'unknown'
     ELSE total_employees_range
 END,
 extract(YEAR
         FROM a.relevant_date) - cb.founded_year AS company_age,
 coalesce(tra.n_training, 0) AS n_training,
 coalesce(qoe_score, -1) AS qoe_score,
days_from_contact_added,
coalesce(days_from_artifacts_count_change, -1) AS days_from_artifacts_count_change,
coalesce(days_from_artifacts_size_change, -1) AS days_from_artifacts_size_change,
coalesce(days_from_binaries_count_change, -1) AS days_from_binaries_count_change,
coalesce(days_from_binaries_size_change, -1) AS days_from_binaries_size_change,
coalesce(days_from_items_count_change, -1) AS days_from_items_count_change,
-- coalesce(days_from_last_xray, -1) AS days_from_last_xray,
-- coalesce(total_security_policies, -1) AS total_security_policies,
-- coalesce(total_policies_count, -1) AS total_policies_count,
-- coalesce(total_violations_count, -1) AS total_violations_count,
-- coalesce(total_watches_count, -1) AS total_watches_count,
coalesce(days_from_permissions_change, -1) AS days_from_permissions_change,
coalesce(days_from_internal_groups_change, -1) AS days_from_internal_groups_change,
coalesce(days_from_users_change, -1) AS days_from_users_change,
-- coalesce(days_from_indexed_repos_change, -1) AS days_from_indexed_repos_change,
-- coalesce(days_from_policies_change, -1) AS days_from_policies_change,
-- coalesce(days_from_violations_change, -1) AS days_from_violations_change,
-- coalesce(days_from_watches_change, -1) AS days_from_watches_change,
 coalesce(n_jira_cases, 0) AS n_jira_cases,
 coalesce(unresolved_jira_cases, 0) AS unresolved_jira_cases,
coalesce(avg_resolution_days, -1) AS avg_resolution_days,
 coalesce(pricing_views, -1) AS pricing_views,
 coalesce(artifactory_views, -1) AS artifactory_views,
 coalesce(xray_views, -1) AS xray_views,
 coalesce(support_views, -1) AS support_views_views,
 coalesce(knowledge_views, -1) AS knowledge_views,
--  coalesce(datediff(MONTH, sl.createddate ::date, a.relevant_date), -1) AS seniority,
 coalesce(n_trials, 0) AS n_trials,
 coalesce(n_ent_trials, 0) AS n_ent_trials,
coalesce(n_ha_mentioned_sessions, 0) AS n_ha_mentioned_sessions,
coalesce(n_ent_mentioned_sessions, 0) AS n_ent_mentioned_sessions,
coalesce(n_competitor_mentioned_sessions, 0) AS n_competitor_mentioned_sessions,
coalesce(n_xray_mentioned_sessions, 0) AS n_xray_mentioned_sessions,
coalesce(n_replys, 0) AS n_replys,
coalesce(n_sent, 0) AS n_sent,
coalesce(n_calls, 0) AS n_calls,
coalesce(n_task_xray, 0) AS n_task_xray,
coalesce(replys_to_sent, -1) AS replys_to_sent,
coalesce(days_since_reply, 1000) AS days_since_reply,
coalesce(days_since_sent, 1000) AS days_since_sent
-- coalesce(days_since_xray_task, 1000) AS days_since_xray_task
FROM #base_accounts AS a
LEFT JOIN #storage AS b1 ON a.account_id = b1.account_id
AND a.relevant_date = b1.relevant_date
AND b1.period_range = '3 Months'
LEFT JOIN #storage AS b2 ON a.account_id = b2.account_id
AND a.relevant_date = b2.relevant_date
AND b2.period_range = '4 Months'
LEFT JOIN #storage AS b3 ON a.account_id = b3.account_id
AND a.relevant_date = b3.relevant_date
AND b3.period_range = '5 Months'
LEFT JOIN #storage AS b4 ON a.account_id = b4.account_id
AND a.relevant_date = b4.relevant_date
AND b4.period_range = '6 Months'
LEFT JOIN #storage AS b5 ON a.account_id = b5.account_id
AND a.relevant_date = b5.relevant_date
AND b5.period_range = '7 Months'
LEFT JOIN #users AS u1 ON a.account_id = u1.account_id
AND a.relevant_date = u1.relevant_date
AND u1.period_range = '3 Months'
LEFT JOIN #users AS u2 ON a.account_id = u2.account_id
AND a.relevant_date = u2.relevant_date
AND u2.period_range = '4 Months'
LEFT JOIN #users AS u3 ON a.account_id = u3.account_id
AND a.relevant_date = u3.relevant_date
AND u3.period_range = '5 Months'
LEFT JOIN #users AS u4 ON a.account_id = u4.account_id
AND a.relevant_date = u4.relevant_date
AND u4.period_range = '6 Months'
LEFT JOIN #users AS u5 ON a.account_id = u5.account_id
AND a.relevant_date = u5.relevant_date
AND u5.period_range = '7 Months'
LEFT JOIN #repositories AS c1 ON a.account_id = c1.account_id
AND a.relevant_date = c1.relevant_date
AND c1.period_range = '3 Months'
LEFT JOIN #repositories AS c2 ON a.account_id = c2.account_id
AND a.relevant_date = c2.relevant_date
AND c2.period_range = '4 Months'
LEFT JOIN #repositories AS c3 ON a.account_id = c3.account_id
AND a.relevant_date = c3.relevant_date
AND c3.period_range = '5 Months'
LEFT JOIN #repositories AS c4 ON a.account_id = c4.account_id
AND a.relevant_date = c4.relevant_date
AND c4.period_range = '6 Months'
LEFT JOIN #repositories AS c5 ON a.account_id = c5.account_id
AND a.relevant_date = c5.relevant_date
AND c5.period_range = '7 Months'
LEFT JOIN #cbit AS cb ON cb.account_id = a.account_id
LEFT JOIN #contacts AS ct ON ct.accountid = a.account_id
AND a.relevant_date = ct.relevant_date
LEFT JOIN #training AS tra ON tra.account_id = a.account_id
AND a.relevant_date = tra.relevant_date
LEFT JOIN #qoe AS q ON q.account_id = a.account_id
AND a.relevant_date = q.relevant_date
-- LEFT JOIN #xray AS xr1 ON a.account_id = xr1.account_id
-- AND a.relevant_date = xr1.relevant_date
-- AND xr1.period_range = '3 Months'
-- LEFT JOIN #xray AS xr2 ON a.account_id = xr2.account_id
-- AND a.relevant_date = xr2.relevant_date
-- AND xr2.period_range = '4 Months'
-- LEFT JOIN #xray AS xr3 ON a.account_id = xr3.account_id
-- AND a.relevant_date = xr3.relevant_date
-- AND xr3.period_range = '5 Months'
-- LEFT JOIN #xray AS xr4 ON a.account_id = xr4.account_id
-- AND a.relevant_date = xr4.relevant_date
-- AND xr4.period_range = '6 Months'
-- LEFT JOIN #xray AS xr5 ON a.account_id = xr5.account_id
-- AND a.relevant_date = xr5.relevant_date
-- AND xr5.period_range = '7 Months'
-- LEFT JOIN #days_from_xray AS dx ON a.account_id = dx.account_id
-- AND a.relevant_date = dx.relevant_date
-- LEFT JOIN #xray_policies AS xp ON a.account_id = xp.account_id
-- AND a.relevant_date = xp.relevant_date
-- LEFT JOIN #xray_totals AS xt ON a.account_id = xt.account_id
-- AND a.relevant_date = xt.relevant_date
-- LEFT JOIN #days_from_policies_count AS xrp ON a.account_id = xrp.account_id
-- AND a.relevant_date = xrp.relevant_date
-- LEFT JOIN #days_from_watches_count AS xrw ON a.account_id = xrw.account_id
-- AND a.relevant_date = xrw.relevant_date
-- LEFT JOIN #days_from_violations_count AS xrv ON a.account_id = xrv.account_id
-- AND a.relevant_date = xrv.relevant_date
-- LEFT JOIN #days_from_indexed_repos AS dfir ON a.account_id = dfir.account_id
-- AND a.relevant_date = dfir.relevant_date
LEFT JOIN #support AS s ON a.account_id = s.account_id
AND a.relevant_date = s.relevant_date
LEFT JOIN #technical_sessions AS ts ON a.account_id = ts.account_id
AND a.relevant_date = ts.relevant_date
LEFT JOIN #jira_cases AS jc ON a.account_id = jc.account_id
AND a.relevant_date = jc.relevant_date
LEFT JOIN #cases_agg AS ca ON a.account_id = ca.account_id
AND a.relevant_date = ca.relevant_date
LEFT JOIN #poor_cases AS pc ON a.account_id = pc.account_id
AND a.relevant_date = pc.relevant_date
LEFT JOIN #key_pages_views AS k ON a.account_id = k.account_id
AND a.relevant_date = k.relevant_date
LEFT JOIN dims.dim_accounts AS da ON da.account_id = a.account_id
LEFT JOIN #contracts AS cr ON a.account_id = cr.account_id
AND a.relevant_date = cr.relevant_date
LEFT JOIN #trials AS tr ON a.account_id = tr.account_id
AND a.relevant_date = tr.relevant_date
LEFT JOIN #cloud_sub AS cs ON a.account_id = cs.account_id
AND a.relevant_date = cs.relevant_date
LEFT JOIN #emails AS em ON a.account_id = em.account_id
AND a.relevant_date = em.relevant_date
LEFT JOIN #days_sicne_reply AS dsr ON a.account_id = dsr.account_id
AND a.relevant_date = dsr.relevant_date
LEFT JOIN #triggers_sessions AS tss ON a.account_id = tss.account_id
AND a.relevant_date = tss.relevant_date
-- LEFT JOIN #moved_to_prox AS mfp ON a.account_id = mfp.account_id
-- AND a.relevant_date = mfp.relevant_date
LEFT JOIN #days_from_artifacts_count AS dfac ON a.account_id = dfac.account_id
AND a.relevant_date = dfac.relevant_date
LEFT JOIN #days_from_artifacts_size AS dfas ON a.account_id = dfas.account_id
AND a.relevant_date = dfas.relevant_date
LEFT JOIN #days_from_binaries_count AS dfbc ON a.account_id = dfbc.account_id
AND a.relevant_date = dfbc.relevant_date
LEFT JOIN #days_from_binaries_size AS dfbs ON a.account_id = dfbs.account_id
AND a.relevant_date = dfbs.relevant_date
LEFT JOIN #days_from_items_count AS dfic ON a.account_id = dfic.account_id
AND a.relevant_date = dfic.relevant_date
LEFT JOIN #days_from_permissions AS dfp ON a.account_id = dfp.account_id
AND a.relevant_date = dfp.relevant_date
LEFT JOIN #days_from_internal_groups AS dfig ON a.account_id = dfig.account_id
AND a.relevant_date = dfig.relevant_date
LEFT JOIN #days_from_users AS dfu ON a.account_id = dfu.account_id
AND a.relevant_date = dfu.relevant_date
LEFT JOIN #days_from_contact AS dfc ON a.account_id = dfc.account_id
AND a.relevant_date = dfc.relevant_date
LEFT JOIN salesforce.account AS sl ON a.account_id = left(sl.accountid_full, LEN (sl.accountid_full) -3)
WHERE b1.account_id IS NOT NULL
  AND b2.account_id IS NOT NULL
  AND b3.account_id IS NOT NULL
  AND b4.account_id IS NOT NULL
  AND b5.account_id IS NOT NULL
  AND count_ent = 0
  AND count_prox = 0
ORDER BY 1,
         2