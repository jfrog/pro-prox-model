--------------------
-- Account Details
--------------------
drop table if exists #contracts_accounts_tmp;
select  ca.account_id,
        sales_business_unit,
        ca.sales_team,
        new_prd.product_name as new_product_name,
        prev_prd.product_name as prev_product_name,
        case when net_new_arr > 0 and event_type = 'Customer Success' then 1 else 0 end as pure_upsell_flag,
        event_month,
        da.territory,
        logo.pitchbookid__c as pbid,
        opp_owner_id,
        opp_type,
        max(event_month) over (partition by ca.account_id) as last_event_month,
        min(event_month) over (partition by ca.account_id) as first_event_month
into #contracts_accounts_tmp
  from staging.contract_analysis ca
left join dims.dim_products prev_prd on prev_prd.product_id = ca.product_id
left join dims.dim_products new_prd on new_prd.product_id = ca.product_renewal_id
left join dims.dim_accounts da on da.account_id = ca.account_id
left join salesforce.logo__c as logo on da.logo = logo.name
where isclosed = 'true'
and (iswon = 'true' or iswon is null)
and sales_business_unit <> 'Cloud'
and event_month <= CURRENT_DATE
;

---------------------------------
-- random_relevant_month
---------------------------------
drop table if exists #random_relevant_month;
select account_id, to_Date(extract(MONTH from relevant_month)::varchar ||' '|| extract(YEAR from relevant_month)::varchar, 'mm YYYY') as relevant_month
into #random_relevant_month
from (
select *, row_number() over (partition by account_id order by random()) as rn
from  (select a.account_id, case when da.first_changed_to_customer is null
then (add_months(a.first_event_month,4)) + random() * ((current_date::timestamp - INTERVAL '3 month') -  add_months(a.first_event_month,4))
when add_months(da.first_changed_to_customer,4) < first_event_month then
(first_event_month::timestamp) + random() * ((current_date::timestamp - INTERVAL '3 month') -  first_event_month::timestamp)
when add_months(first_changed_to_customer,4) >= first_event_month then
(add_months(da.first_changed_to_customer,4)) + random() * ((current_date::timestamp - INTERVAL '3 month') -  add_months(da.first_changed_to_customer,4))
end as relevant_month
from #contracts_accounts_tmp as a
left join dims.dim_accounts as da on da.account_id = a.account_id)) as sub
where rn = 1;

--------------------
-- Upsells
--------------------
drop table if exists #upsell_accounts;
select account_id, event_month, case when (previous_product_rank) = 1 then 'JFrog Pro X'
            when (previous_product_rank) = 2 then 'JFrog Pro' else 'Pure Upsell' end as previous_product
into #upsell_accounts
from
(
select account_id,
       min(case when prev_product_name  in ('JFrog Pro X','JFrog Pro Plus') then 1
            when prev_product_name = 'JFrog Pro' then 2 else 3 end) as previous_product_rank,
       max(event_month) as event_month
from #contracts_accounts_tmp
where
(
(new_product_name in ('JFrog Enterprise','JFrog Enterprise+')
and prev_product_name in ('JFrog Pro','JFrog Pro X' ,'JFrog Pro Plus'))
or
(
 pure_upsell_flag = 1 and prev_product_name in ('JFrog Enterprise','JFrog Enterprise+') -- Pure Upsell
)
and prev_product_name != new_product_name)
group by 1
);

---------------------------------
-- account_relevant_date
---------------------------------
drop table if exists  #account_relevant_date;
select a.account_id,
       sales_business_unit,
       territory,
       case when ua.event_month is not null -- Upgraded
            then ADD_MONTHS(DATE_TRUNC('month',ua.event_month),-3)
            else r.relevant_month end as relevant_month,
       case when ua.event_month is not null then 1 else 0 end as upgraded,
       max(pbid) as pbid
into #account_relevant_date
from #contracts_accounts_tmp as a
left join #upsell_accounts as ua on ua.account_id = a.account_id
left join #random_relevant_month as r on r.account_id = a.account_id
and relevant_month <= CURRENT_DATE
group by 1,2,3,4,5;

---------------------------------
-- add another relevant_month for upsell_accounts
---------------------------------
drop table if exists  #account_relevant_date1;
select *
into #account_relevant_date1
from #account_relevant_date union
(select a.account_id, a.sales_business_unit, a.territory, case when add_months(da.first_changed_to_customer, 3) < add_months(relevant_month, -12) then
add_months(relevant_month, -12) else add_months(da.first_changed_to_customer, 3) end as relevant_month, 0 as upgraded, pbid
from #account_relevant_date as a left join dims.dim_accounts as da on da.account_id = a.account_id where upgraded = 1);

---------------------------------
-- time from renewal
---------------------------------
drop table if exists  #time_from_renewal;
select account_id, relevant_month, case when max(tfr) > 0 then min(case when tfr > 0 then tfr end)
else 12 + max(tfr) end as tfr
into #time_from_renewal
 from
(select ar.account_id, ca.event_month, ar.relevant_month, datediff(MONTH, ca.event_month, ar.relevant_month) as tfr
from #account_relevant_date1 as ar
left join #contracts_accounts_tmp as ca
on ca.account_id = ar.account_id)
group by 1,2
;

---------------------------------
-- triggers
---------------------------------
drop table if exists #triggers_cases;
select ar.account_id, relevant_month,
count(distinct(case when trigger_term in ('high availability', 'high-availability',' ha') then event_date end)) as n_ha_mentioned_cases,
count(distinct(case when trigger_term in ('balancer', 'balancing', 'balance') then event_date end)) as n_bal_mentioned_cases,
count(distinct(case when trigger_term = 'enterprise' then event_date end)) as n_ent_mentioned_cases,
count(distinct(case when trigger_term = 'multiple' then event_date end)) as n_mul_mentioned_cases,
count(distinct(case when trigger_term = 'replications' then event_date end)) as n_rep_mentioned_cases,
count(distinct(case when trigger_term in ('disaster recovery', 'downtime', ' dr', 'down time', 'bad performance') then event_date end)) as n_bad_mentioned_cases
into #triggers_cases
from #account_relevant_date1 as ar
left join data_science.upsell_terms_trigger as tc
on tc.account_id = ar.account_id
where source in ('incoming email - Support', 'outgoing email - Support')
and event_date between ADD_MONTHS(relevant_month, -12) and relevant_month
group by 1,2;

--drop table if exists #triggers_sessions;
--select ar.account_id, relevant_month,
--count(distinct(case when trigger_term in ('high availability', 'high-availability',' ha') then event_date end)) as n_ha_mentioned_sessions,
--count(distinct(case when trigger_term in ('balancer', 'balancing', 'balance') then event_date end)) as n_bal_mentioned_sessions,
--count(distinct(case when trigger_term = 'enterprise' then event_date end)) as n_ent_mentioned_sessions,
--count(distinct(case when trigger_term = 'multiple' then event_date end)) as n_mul_mentioned_sessions,
--count(distinct(case when trigger_term = 'replications' then event_date end)) as n_rep_mentioned_sessions,
--count(distinct(case when trigger_term in ('disaster recovery', 'downtime', ' dr', 'down time', 'bad performance') then event_date end)) as n_bad_mentioned_sessions
--into #triggers_sessions
--from data_science.triggered_instances_sessions as tc
--from #account_relevant_date1 as ar
--left join data_science.upsell_terms_trigger as tc
--on left(tc.account_id , len(tc.account_id) -3) = ar.account_id
--and len(tc.account_id) > 0
--on tc.account_id = ar.account_id
--where TO_DATE(created_date,'MM/DD/YYYY') between ADD_MONTHS(relevant_month, -12) and relevant_month
--where source = 'technical session'
--and event_date between ADD_MONTHS(relevant_month, -12) and relevant_month
--group by 1,2;


drop table if exists #triggers_sessions;
select a.account_id, relevant_month,
       count(distinct(case when term = 'high availability' then instance_date end)) as n_ha_mentioned_sessions,
       count(distinct(case when term = 'enterprise' then instance_date end)) as n_ent_mentioned_sessions,
       count(distinct(case when term = 'multiple' then instance_date end)) as n_mul_mentioned_sessions,
       count(distinct(case when term = 'balance' then instance_date end)) as n_bal_mentioned_sessions,
       count(distinct(case when term in ('disaster recovery', 'downtime', 'bad performance') then instance_date end)) as n_bad_mentioned_sessions,
       count(distinct(case when term = 'replications' then instance_date end)) as n_rep_mentioned_sessions
into #triggers_sessions
from #account_relevant_date1 as a
left join data_science.termtriggers_sessions_allfields_fixed as b
on a.account_id = b.account_id
where type in ('session_Background', 'session_Session Summary')
and TO_DATE(instance_date,'MM/DD/YYYY') between ADD_MONTHS(relevant_month, -12) and relevant_month
group by 1,2;


---------------------------------
-- derby
---------------------------------
drop table if exists #derby;
select ar.account_id, relevant_month, max(case when db_type = 'derby' then 1 else 0 end) as is_derby
into #derby
from artifactory.dwh_service_trends_summary_storage as s
left join #account_relevant_date1 as ar
on s.account_id = ar.account_id
where art_create_date between ADD_MONTHS(DATE_TRUNC('day', ar.relevant_month),-1) and DATE_TRUNC('day', ar.relevant_month)
group by 1,2;

---------------------------------
-- Docker / Generic / Helm
---------------------------------
drop table if exists  #repositories;
select  account_id,
        period_range,
        relevant_month,
        --max(left(version,1)) as version,
        count(*) as n_env,
        sum(Maven) as Maven,
        sum(Generic) as Generic,
        sum(BuildInfo) as BuildInfo,
        sum(Docker) as Docker,
        sum(Npm) as Npm,
        sum(Pypi) as Pypi,
        sum(Gradle) as Gradle,
        sum(NuGet) as NuGet,
        sum(YUM) as YUM,
        sum(Helm) as Helm,
        sum(Gems) as Gems,
        sum(Debian) as Debian,
        sum(Ivy) as Ivy,
        sum(SBT) as SBT,
        sum(Conan) as Conan,
        sum(Bower) as Bower,
        sum(Go) as Go,
        sum(Chef) as Chef,
        sum(GitLfs) as GitLfs,
        sum(Composer) as Composer,
        sum(Puppet) as Puppet,
        sum(Conda) as Conda,
        sum(Vagrant) as Vagrant,
        sum(CocoaPods) as CocoaPods,
        sum(CRAN) as CRAN,
        sum(Opkg) as Opkg,
        sum(P2) as P2,
        sum(VCS) as VCS,
        sum(Alpine) as Alpine
into #repositories
from
(
select  environment_service_id,
        repo.account_id,
        relevant_month,
        --max(case when left(version,1) != '$' then left(version,1)::float end) as version,
        case when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) and DATE_TRUNC('day',b.relevant_month) then '3 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) then '4 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-3) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) then '5 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',current_date),-1) and DATE_TRUNC('day',current_date) then 'Current'
             end as period_range,
        max(case when repo.package_type = 'Maven' then avg_repos else 0 end) as Maven,
        max(case when repo.package_type = 'Generic' then avg_repos else 0 end) as Generic,
        max(case when repo.package_type = 'BuildInfo' then avg_repos else 0 end) as BuildInfo,
        max(case when repo.package_type = 'Docker' then avg_repos else 0 end) as Docker,
        max(case when repo.package_type = 'Npm' then avg_repos else 0 end) as Npm,
        max(case when repo.package_type = 'Pypi' then avg_repos else 0 end) as Pypi,
        max(case when repo.package_type = 'Gradle' then avg_repos else 0 end) as Gradle,
        max(case when repo.package_type = 'NuGet' then avg_repos else 0 end) as NuGet,
        max(case when repo.package_type = 'YUM' then avg_repos else 0 end) as YUM,
        max(case when repo.package_type = 'Helm' then avg_repos else 0 end) as Helm,
        max(case when repo.package_type = 'Gems' then avg_repos else 0 end) as Gems,
        max(case when repo.package_type = 'Debian' then avg_repos else 0 end) as Debian,
        max(case when repo.package_type = 'Ivy' then avg_repos else 0 end) as Ivy,
        max(case when repo.package_type = 'SBT' then avg_repos else 0 end) as SBT,
        max(case when repo.package_type = 'Conan' then avg_repos else 0 end) as Conan,
        max(case when repo.package_type = 'Bower' then avg_repos else 0 end) as Bower,
        max(case when repo.package_type = 'Go' then avg_repos else 0 end) as Go,
        max(case when repo.package_type = 'Chef' then avg_repos else 0 end) as Chef,
        max(case when repo.package_type = 'GitLfs' then avg_repos else 0 end) as GitLfs,
        max(case when repo.package_type = 'Composer' then avg_repos else 0 end) as Composer,
        max(case when repo.package_type = 'Puppet' then avg_repos else 0 end) as Puppet,
        max(case when repo.package_type = 'Conda' then avg_repos else 0 end) as Conda,
        max(case when repo.package_type = 'Vagrant' then avg_repos else 0 end) as Vagrant,
        max(case when repo.package_type = 'CocoaPods' then avg_repos else 0 end) as CocoaPods,
        max(case when repo.package_type = 'CRAN' then avg_repos else 0 end) as CRAN,
        max(case when repo.package_type = 'Opkg' then avg_repos else 0 end) as Opkg,
        max(case when repo.package_type = 'P2' then avg_repos else 0 end) as P2,
        max(case when repo.package_type = 'VCS' then avg_repos else 0 end) as VCS,
        max(case when repo.package_type = 'Alpine' then avg_repos else 0 end) as Alpine
from artifactory.dwh_service_trends_repo repo
left join #account_relevant_date1 b ON repo.account_id = b.account_id
where period_range is not null --- and repo_type = 'Local Repositories'
group by 1,2,3,4)
group by 1,2,3
;

---------------------------------
-- Storage
---------------------------------
-- Same as above
drop table if exists  #storage ;
/* Artifcats */
select  account_id,
        relevant_month,
        period_range,
        max(version) as version,
        sum(artifacts_count) as artifacts_count,
        sum(artifacts_size) as artifacts_size,
        sum(binaries_count) as binaries_count,
        sum(binaries_size) as binaries_size,
        sum(items_count) as items_count
into #storage
from
(
select a.account_id,
       a.environment_service_id,
       relevant_month,
       case when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) and DATE_TRUNC('day',b.relevant_month) then '3 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) then '4 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-3) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) then '5 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',current_date),-1) and DATE_TRUNC('day',current_date) then 'Current'
             end as period_range,
       max(left(version,1)) as version,
       avg(avg_artifacts_count) as artifacts_count,
       avg(avg_artifacts_size) as artifacts_size,
       avg(avg_binaries_count) as binaries_count,
       avg(avg_binaries_size) as binaries_size,
       avg(avg_items_count) as items_count
--
from artifactory.dwh_service_trends_summary_storage a
INNER JOIN #account_relevant_date1 b ON a.account_id = b.account_id
where period_range is not null
group by 1,2,3,4
)
group by 1,2,3;

-- Same as above
drop table if exists  #users ;
/* Artifcats */
select  account_id,
        period_range,
        relevant_month,
        sum(number_of_premissions) as number_of_premissions,
        sum(internal_groups) as internal_groups,
        sum(number_of_users) as number_of_users
into #users
from
(
select u.account_id,
       u.environment_service_id,
       relevant_month,
       case when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) and DATE_TRUNC('day',b.relevant_month) then '3 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) then '4 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-3) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) then '5 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',current_date),-1) and DATE_TRUNC('day',current_date) then 'Current'
             end as period_range,
       avg(number_of_permission_targets) as number_of_premissions,
       avg(internal_groups) as internal_groups,
       avg(number_of_users) as number_of_users
--
from artifactory.dwh_service_trends_security as u
INNER JOIN #account_relevant_date1 b ON u.account_id = b.account_id
where period_range is not null
group by 1,2,3,4
)
group by 1,2,3;

-- Same as above
drop table if exists  #replications ;
/* Artifcats */
select  account_id,
        period_range,
        relevant_month,
        sum(pull_replications) as pull_replications,
        sum(push_replications) as push_replications,
        sum(event_replications) as event_replications
into #replications
from
(
select r.account_id,
       r.environment_service_id,
       relevant_month,
       case when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) and DATE_TRUNC('day',b.relevant_month) then '3 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-1) then '4 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-3) and ADD_MONTHS(DATE_TRUNC('day',b.relevant_month),-2) then '5 Months'
             when DATE_TRUNC('day',art_create_date) between  ADD_MONTHS(DATE_TRUNC('day',current_date),-1) and DATE_TRUNC('day',current_date) then 'Current'
             end as period_range,
       avg(case when pull_replication = 't' then avg_repos else 0 end) as pull_replications,
       avg(case when push_replication = 'true' then avg_repos else 0 end) as push_replications,
       avg(case when event_replication = 't' then avg_repos else 0 end) as event_replications
--
from artifactory.dwh_service_trends_repo_include_replication as r
INNER JOIN #account_relevant_date1 b ON r.account_id = b.account_id
where period_range is not null
group by 1,2,3,4
)
group by 1,2,3;

--------------------
-- PB
--------------------
drop table if exists #pb;
select distinct c.pbid,
                i.industry_sector_desc,
                i.industry_group_desc,
                i.industry_desc,
                a.employee_count,
                case when c.pbid is not null and a.stock_ticker is null then 'private'
                     when a.stock_ticker is not null then 'public' end as public_or_private,
                a.total_money_raised_amount
into #pb
from pitchbook.config_pbid c
LEFT JOIN pitchbook.attributes a on c.pbid = a.company_id
LEFT JOIN pitchbook.industry i on c.pbid = i.company_id and i.isprimary = 'True'
where  c.pbid is not null;

--------------------
-- ARR
--------------------
drop table if exists #arr ;
select opp.account_id,
       sum(opp.arr_growth__c) as Expected_ARR_Growth,
       sum(previous_arr__c) as Related_Contracts_ARR,
       sum(arr__c) as Expected_ARR
into #arr
from salesforce.fact_opportunities opp
where opp.contract_expiry_date_with_coterm >= ADD_MONTHS(DATE_TRUNC('month',CURRENT_DATE),-13)
and  isclosed = 'false'
group by 1
;

---------------
-- Support
---------------
drop table if exists #support;
select  a.account_id, relevant_month,
        count(distinct case when movement_date between dateadd(month,-12,a.relevant_month) and a.relevant_month then cases.case_id end) as cases_within_last_year,
        count(distinct case when movement_date between dateadd(month,-3,a.relevant_month) and a.relevant_month then cases.case_id end) as cases_within_3_last_months
into #support
from qoc.stg_events_measures as events
left join salesforce.fact_cases as cases on cases.case_id = events.case_id and events.parameter_id in (10)
INNER JOIN #account_relevant_date1 a ON cases.account_id = a.account_id
where DATE_TRUNC('day',movement_date) between ADD_MONTHS(DATE_TRUNC('month',a.relevant_month),-12) and a.relevant_month
group by 1,2
;

-------------------------
-- Technical sessions
-------------------------
drop table if exists #technical_sessions;
select s.account_id, relevant_month, count(*) as total_sessions_past_year
into #technical_sessions
from facts.fact_technical_sessions s
INNER JOIN #account_relevant_date1 a ON s.account_id = a.account_id
where session_createddate between ADD_MONTHS(DATE_TRUNC('month',a.relevant_month),-12) and a.relevant_month
group by 1,2;

-------------------------
-- Trials
-------------------------
-- Investigate the data
drop table if exists #trials;
select accountid as account_id,relevant_month, count(*) as total_trials
into #trials
from (
select whoid,accountid,id,trial_product,t.email,case when c.territory__c = 'Americas' then 'NA' else c.territory__c end as territory,
cast(convert_timezone('PDT',createddate) as date) as  trial_us_date,
row_number() over (partition by whoid,trial_product,cast(date_trunc('month',CreatedDate) as date) order by createddate asc)as rn_trial_in_month
from facts.fact_trials  t
left join dims.dim_contacts c on t.email = c.email
)relevant_trials
INNER JOIN #account_relevant_date1 a ON relevant_trials.accountid = a.account_id
where rn_trial_in_month=1
and trial_us_date between ADD_MONTHS(DATE_TRUNC('month',a.relevant_month),-12) and a.relevant_month
group by 1,2
;

-------------------------
-- Training
-------------------------
drop table if exists #training;
select account_id, relevant_month, count(distinct(createddate)) as n_training
into #training
from
(select * from salesforce.training__c where recordtypeid='012w0000000R1Yp') as st right join
#account_relevant_date1 as ar
on st.account__c = ar.account_id
where createddate <= relevant_month
group by 1,2
;

-------------------------
-- Top Subscription
-------------------------
drop table if exists #top_subscription;
select logo, number_of_accounts,
       case  when min(top_sub) = 1 then 'JFrog Enterprise+'
             when min(top_sub) = 2 then 'JFrog Enterprise'
             when min(top_sub) = 3 then 'JFrog Pro X'
             when min(top_sub) = 4 then 'JFrog Pro' end as top_sub
into #top_subscription
from
(select logo, da.account_id, lo.number_of_accounts__c as number_of_accounts, top_subscription,
case  when top_subscription = 'JFrog Enterprise+' then 1
                                              when top_subscription = 'JFrog Enterprise' then 2
                                              when top_subscription = 'JFrog Pro X' then 3
                                              when top_subscription = 'JFrog Pro' then 4 end as top_sub
from dims.dim_accounts da
inner join salesforce.logo__c lo on da.logo = lo.name)
group by  logo, number_of_accounts;

-------------
--Contacts
-------------
drop table if exists #contacts;
select accountid, relevant_month, count(distinct(createddate)) as n_contacts
into #contacts
from salesforce.contact as c right join #account_relevant_date1 as ar
on c.accountid = ar.account_id
where createddate <= relevant_month
group by 1,2
;

-------------
--Contracts
-------------
drop table if exists #contracts;
select accountid,relevant_month, count(distinct(startdate)) as n_contracts,
count(distinct(case when relevant_month <= enddate then startdate end)) as n_active_contracts,
max(case when status = 'Co-termed' and relevant_month >= enddate then 1 else 0 end) as is_cotermed,
count(distinct(case when contract_value__c in ('JFrog Enterprise','JFrog Enterprise+') then startdate end)) as count_ent,
count(distinct(case when contract_value__c  = 'JFrog Pro' then startdate end)) as count_pro
into #contracts
from salesforce.contract as c right join #account_relevant_date1 as ar
on c.accountid = ar.account_id
where relevant_month >= DATE_TRUNC('month', startdate)
group by 1,2;

-------------------
--Number of servers
-------------------
drop table if exists #servers;
select accountid,relevant_month, max(servers) as max_servers, min(servers) as min_servers, max_servers - min_servers as servers_diff,
max(artifactory_servers) as max_artifactory_servers, min(artifactory_servers) as min_artifactory_servers, max_artifactory_servers - min_artifactory_servers as artifactory_servers_diff
into #servers
from
(select accountid, coalesce(startdate,'2012-01-01') as license_start_date,
case when status = 'Co-termed' then coalesce(co_termed_date__c,enddate)
when (status = 'Expired' or status = 'Activated') then coalesce(enddate,current_Date)
when status = 'Canceled' then coalesce(cancellation_date__c,enddate)
else coalesce(enddate,current_Date) end as actual_end_date,
coalesce(no_of_servers__c,num_of_artifactory_servers__c) as servers, coalesce(num_of_artifactory_servers__c,no_of_servers__c) as artifactory_servers
from salesforce.contract
where status in ('Expired','Co-termed', 'Canceled', 'Activated')
order by license_start_Date, accountid) as ns right join #account_relevant_date1 as ar
on ns.accountid = ar.account_id
where relevant_month between license_start_date and actual_end_date
group by 1,2;

--------------------
--Fail opportunities
--------------------
drop table if exists #opportunities;
select accountid, relevant_month, count(distinct(createddate)) as n_fail_opp
into #opportunities
from (select distinct * from salesforce.opportunity as o
where iswon = 'false' and lower(name) not like '%cloud%'
and lower(name) like '%upsell%' and o.type != 'New Business') as op right join #account_relevant_date1 as ar
on op.accountid = ar.account_id
where relevant_month >= createddate
group by 1,2;

-------------
--QOE
-------------
drop table if exists #qoe;
select d.account_id, qoe_score,relevant_month
into #qoe
from salesforce.dwh_qoe_scores as d right join #account_relevant_date1 as ar
on d.account_id = ar.account_id
where create_date_monthly = relevant_month
and is_fictive = 0;

----------------
--visits_private
----------------
drop table if exists #visits_private;
select ar.account_id, relevant_month, count(distinct(visit_date)) as count_visits_private
into #visits_private
from
(select account_id,(TIMESTAMP WITH TIME ZONE 'epoch' + CAST(LEFT (CAST(s.time_stamp AS text),10) AS BIGINT)*INTERVAL '1 second') visit_date
from dims.dim_contacts  c
join google_analytics.myjfrog_sessions s
on c.ga_cid = s.google_client_id
where account_id is not null and visit_date is not null) as v right join #account_relevant_date1 as ar
on v.account_id = ar.account_id
where relevant_month >= visit_date
group by 1,2;

----------------
--visits_public
----------------
drop table if exists #visits_public;
select ar.account_id, relevant_month, count(distinct(visit_date)) as count_visits_public
into #visits_public
from
(select account_id,(TIMESTAMP WITH TIME ZONE 'epoch' + CAST(LEFT (CAST(s.time_stamp AS text),10) AS BIGINT)*INTERVAL '1 second') visit_date
from dims.dim_contacts  c
join google_analytics.jfrogcom_sessions s
on c.ga_cid = s.google_client_id
where account_id is not null and visit_date is not null) as v right join #account_relevant_date1 as ar
on v.account_id = ar.account_id
where relevant_month >= visit_date
group by 1,2;

-------------
--Clear Bit
-------------
drop table if exists #cbit;
select  da.account_id,
        max(cbit__companycategorysubindustry__c) as category_sub_industry,
        max(cbit__companycategoryindustry__c) as category_industry,
        max(cbit__companycategoryindustrygroup__c) as industry_group,
        max(cbit__companymetricsestimatedannualrevenue__c) as estimated_annual_revenue,
        max(cbit__companymetricsannualrevenue__c) as annual_revenue,
        max(cbit__companymetricsmarketcap__c) as market_cap,
        max(cbit__companygeostreetname__c) as company_street,
        max(cbit__companygeostate__c) as company_state,
        max(cbit__companygeocountry__c) as company_country,
        max(cbit__companygeocity__c) as company_city,
        max(a.cbit__shortcompanyname__c) as cbit_company_name,
        max(cbit__companylinkedinhandle__c) as linkedin_handle,
        max(cbit__companymetricsraised__c) as fund_raised,
        max(cbit__companyfoundedyear__c) as founded_year,
        max(cbit__companymetricsemployees__c) as total_employees,
        max(cbit__companymetricsemployeesrange__c) as total_employees_range,
        count (distinct cbit__email__c) as total_employees_with_details,
        count(distinct case when cbit__employmentsubrole__c in ('software_engineer','web_engineer') or cbit__employmenttitle__c in ('Software Developer','Java Developer','Software Engineer') then cbit__email__c end) as developers,
        count(distinct case when cbit__employmentsubrole__c = 'devops_engineer' then cbit__email__c end) as devops_engineers,
        count(distinct case when cbit__employmentsubrole__c like '%engineer%' or cbit__employmenttitle__c  like '%engineer%' then id end) as engineers
into #cbit
from salesforce.cbit__clearbit__c a
inner join dims.dim_contacts dc on dc.cbit__clearbit__c = a.id
inner join dims.dim_accounts da on da.account_id = dc.account_id
group by 1;





---------------
-- Main
---------------
select distinct
        a.account_id,
        case when a.territory = 'None' then 'unknown' else a.territory end as territory,
        ct.n_contacts,
        cr.n_active_contracts,
        coalesce(vpr.count_visits_private,0) as count_visits_private,
        coalesce(vpu.count_visits_public,0) as count_visits_public,
        coalesce(op.n_fail_opp, 0) as n_fail_opp,
        coalesce(cr.is_cotermed,0) as is_cotermed,
        de.is_derby,
        coalesce(tra.n_training,0) as n_training,

        --triggers--
        coalesce(tc.n_ha_mentioned_cases,0) as n_ha_mentioned_cases,
        coalesce(tc.n_bal_mentioned_cases,0) as n_bal_mentioned_cases,
        coalesce(tc.n_ent_mentioned_cases,0) as n_ent_mentioned_cases,
        coalesce(tc.n_mul_mentioned_cases,0) as n_mul_mentioned_cases,
        coalesce(tc.n_rep_mentioned_cases,0) as n_rep_mentioned_cases,
        coalesce(tc.n_bad_mentioned_cases,0) as n_bad_mentioned_cases,

        coalesce(ts.n_ha_mentioned_sessions,0) as n_ha_mentioned_sessions,
        --coalesce(ts.n_bal_mentioned_sessions,0) as n_bal_mentioned_sessions,
        coalesce(ts.n_ent_mentioned_sessions,0) as n_ent_mentioned_sessions,
        coalesce(ts.n_mul_mentioned_sessions,0) as n_mul_mentioned_sessions,
        --coalesce(ts.n_rep_mentioned_sessions,0) as n_rep_mentioned_sessions,
        --coalesce(ts.n_bad_mentioned_sessions,0) as n_bad_mentioned_sessions,

        u1.number_of_premissions as number_of_permissions_3_months_back,
        u2.number_of_premissions as number_of_permissions_4_months_back,
        u3.number_of_premissions as number_of_permissions_5_months_back,

        coalesce(u1.internal_groups,0) as internal_groups_3_months_back,
        coalesce(u2.internal_groups,0) as internal_groups_4_months_back,
        coalesce(u3.internal_groups,0) as internal_groups_5_months_back,

        coalesce(u1.number_of_users,0) as number_of_users_3_months_back,
        coalesce(u2.number_of_users,0) as number_of_users_4_months_back,
        coalesce(u3.number_of_users,0) as number_of_users_5_months_back,

        b1.artifacts_count as artifacts_count_3_months_back,
        b1.artifacts_size as artifacts_size_3_months_back,
        b1.binaries_count as binaries_count_3_months_back,
        b1.binaries_size as binaries_size_3_months_back,
        b1.items_count as items_count_3_months_back,
        r1.pull_replications as pull_replications_3_months_back,
        r1.push_replications as push_replications_3_months_back,
        r1.event_replications as event_replications_3_months_back,

        b2.artifacts_count as artifacts_count_4_months_back,
        b2.artifacts_size as artifacts_size_4_months_back,
        b2.binaries_count as binaries_count_4_months_back,
        b2.binaries_size as binaries_size_4_months_back,
        b2.items_count as items_count_4_months_back,
        r2.pull_replications as pull_replications_4_months_back,
        r2.push_replications as push_replications_4_months_back,
        r2.event_replications as event_replications_4_months_back,

        b3.artifacts_count as artifacts_count_5_months_back,
        b3.artifacts_size as artifacts_size_5_months_back,
        b3.binaries_count as binaries_count_5_months_back,
        b3.binaries_size as binaries_size_5_months_back,
        b3.items_count as items_count_5_months_back,
        r3.pull_replications as pull_replications_5_months_back,
        r3.push_replications as push_replications_5_months_back,
        r3.event_replications as event_replications_5_months_back,

        --da.industry,
        datediff(MONTH, da.first_changed_to_customer, a.relevant_month) as seniority,
        ---- 3 Months Back
        c1.n_env as n_env_3_months_back,
        c1.Maven as Maven_3_months_back,
        c1.Generic as Generic_3_months_back,
        c1.BuildInfo as BuildInfo_3_months_back,
        c1.Docker as Docker_3_months_back,
        c1.Npm as Npm_3_months_back,
        c1.Pypi as Pypi_3_months_back,
        c1.Gradle as Gradle_3_months_back,
        c1.NuGet as NuGet_3_months_back,
        c1.YUM as YUM_3_months_back,
        c1.Helm as Helm_3_months_back,
        c1.Gems as Gems_3_months_back,
        c1.Debian as Debian_3_months_back,
        c1.Ivy as Ivy_3_months_back,
        c1.SBT as SBT_3_months_back,
        c1.Conan as Conan_3_months_back,
        c1.Bower as Bower_3_months_back,
        c1.Go as Go_3_months_back,
        c1.Chef as Chef_3_months_back,
        c1.GitLfs as GitLfs_3_months_back,
        c1.Composer as Composer_3_months_back,
        c1.Puppet as Puppet_3_months_back,
        c1.Conda as Conda_3_months_back,
        c1.Vagrant as Vagrant_3_months_back,
        c1.CocoaPods as CocoaPods_3_months_back,
        c1.CRAN as CRAN_3_months_back,
        c1.Opkg as Opkg_3_months_back,
        c1.P2 as P2_3_months_back,
        c1.VCS as VCS_3_months_back,
        c1.Alpine as Alpine_3_months_back,
        ----4 Months
        c2.n_env as n_env_4_months_back,
        c2.Maven as Maven_4_months_back,
        c2.Generic as Generic_4_months_back,
        c2.BuildInfo as BuildInfo_4_months_back,
        c2.Docker as Docker_4_months_back,
        c2.Npm as Npm_4_months_back,
        c2.Pypi as Pypi_4_months_back,
        c2.Gradle as Gradle_4_months_back,
        c2.NuGet as NuGet_4_months_back,
        c2.YUM as YUM_4_months_back,
        c2.Helm as Helm_4_months_back,
        c2.Gems as Gems_4_months_back,
        c2.Debian as Debian_4_months_back,
        c2.Ivy as Ivy_4_months_back,
        c2.SBT as SBT_4_months_back,
        c2.Conan as Conan_4_months_back,
        c2.Bower as Bower_4_months_back,
        c2.Go as Go_4_months_back,
        c2.Chef as Chef_4_months_back,
        c2.GitLfs as GitLfs_4_months_back,
        c2.Composer as Composer_4_months_back,
        c2.Puppet as Puppet_4_months_back,
        c2.Conda as Conda_4_months_back,
        c2.Vagrant as Vagrant_4_months_back,
        c2.CocoaPods as CocoaPods_4_months_back,
        c2.CRAN as CRAN_4_months_back,
        c2.Opkg as Opkg_4_months_back,
        c2.P2 as P2_4_months_back,
        c2.VCS as VCS_4_months_back,
        c2.Alpine as Alpine_4_months_back,
        ---- 5 Months Back
        c3.n_env as n_env_5_months_back,
        c3.Maven as Maven_5_months_back,
        c3.Generic as Generic_5_months_back,
        c3.BuildInfo as BuildInfo_5_months_back,
        c3.Docker as Docker_5_months_back,
        c3.Npm as Npm_5_months_back,
        c3.Pypi as Pypi_5_months_back,
        c3.Gradle as Gradle_5_months_back,
        c3.NuGet as NuGet_5_months_back,
        c3.YUM as YUM_5_months_back,
        c3.Helm as Helm_5_months_back,
        c3.Gems as Gems_5_months_back,
        c3.Debian as Debian_5_months_back,
        c3.Ivy as Ivy_5_months_back,
        c3.SBT as SBT_5_months_back,
        c3.Conan as Conan_5_months_back,
        c3.Bower as Bower_5_months_back,
        c3.Go as Go_5_months_back,
        c3.Chef as Chef_5_months_back,
        c3.GitLfs as GitLfs_5_months_back,
        c3.Composer as Composer_5_months_back,
        c3.Puppet as Puppet_5_months_back,
        c3.Conda as Conda_5_months_back,
        c3.Vagrant as Vagrant_5_months_back,
        c3.CocoaPods as CocoaPods_5_months_back,
        c3.CRAN as CRAN_5_months_back,
        c3.Opkg as Opkg_5_months_back,
        c3.P2 as P2_5_months_back,
        c3.VCS as VCS_5_months_back,
        c3.Alpine as Alpine_5_months_back,
       coalesce(cases_within_last_year, 0) as cases_within_last_year,
       coalesce(cases_within_3_last_months,0) as cases_within_3_last_months,
       coalesce(total_sessions_past_year,0) as total_sessions_past_year,
       coalesce(tr.total_trials,0) as total_trials_last_year,
       a.relevant_month,
       --- cb features
       cb.engineers, cb.devops_engineers, cb.developers, cb.total_employees_with_details,
       case when (total_employees_range = '' or total_employees_range is null) then 'unknown' else total_employees_range end,
       case when cb.industry_group = '' or cb.industry_group is null then 'unknown' else cb.industry_group  end as industry,
       extract(YEAR from a.relevant_month) - cb.founded_year as company_age,
       a.upgraded as class,
       coalesce(n_ha_mentioned_cases + n_ha_mentioned_sessions + n_ent_mentioned_cases + n_ent_mentioned_sessions,0) as total_ha_or_ent,
       case when ua.previous_product is not null then ua.previous_product else top_subscription end as product1,
       case when product1 = 'JFrog Pro Plus' then 'JFrog Pro' else product1 end as product,
       case when product = 'JFrog Pro X' and qoe_score = 10 then 4
       when product = 'JFrog Pro X' and qoe_score < 10 and qoe_score >= 7 then 3
       when product = 'JFrog Pro X' and qoe_score < 7 then 2
       else 1 end as qoe_score_cat,
       coalesce(case when cr.count_pro > 0 and product = 'JFrog Pro X' then 1 else 0 end,0) as upgraded_to_prox,
       case when (pb.public_or_private = '' or pb.public_or_private is null) then 'unknown' else pb.public_or_private end,
       case when artifacts_count_3_months_back is null or artifacts_count_4_months_back is null or artifacts_count_5_months_back is null
          then 0 else 1 end as call_home_flag

from #account_relevant_date1 as a
left join #storage as b1 on a.account_id = b1.account_id and a.relevant_month = b1.relevant_month and b1.period_range = '3 Months'
left join #storage as b2 on a.account_id = b2.account_id and a.relevant_month = b2.relevant_month and b2.period_range = '4 Months'
left join #storage as b3 on a.account_id = b3.account_id and a.relevant_month = b3.relevant_month and b3.period_range = '5 Months'
left join #users as u1 on a.account_id = u1.account_id and a.relevant_month = u1.relevant_month and u1.period_range = '3 Months'
left join #users as u2 on a.account_id = u2.account_id and a.relevant_month = u2.relevant_month and u2.period_range = '4 Months'
left join #users as u3 on a.account_id = u3.account_id and a.relevant_month = u3.relevant_month and u3.period_range = '5 Months'
left join #repositories as c1 on a.account_id = c1.account_id and a.relevant_month = c1.relevant_month and  c1.period_range = '3 Months'
left join #repositories as c2 on a.account_id = c2.account_id and a.relevant_month = c2.relevant_month and c2.period_range = '4 Months'
left join #repositories as c3 on a.account_id = c3.account_id and a.relevant_month = c3.relevant_month and c3.period_range = '5 Months'
left join #replications as r1 on a.account_id = r1.account_id and a.relevant_month = r1.relevant_month and r1.period_range = '3 Months'
left join #replications as r2 on a.account_id = r2.account_id and a.relevant_month = r2.relevant_month and r2.period_range = '4 Months'
left join #replications as r3 on a.account_id = r3.account_id and a.relevant_month = r3.relevant_month and r3.period_range = '5 Months'
left join dims.dim_accounts as da on da.account_id = a.account_id
left join dims.dim_users as du ON da.user_owner_id = du.id
left join #pb as pb on pb.pbid = a.pbid
left join #support as s on s.account_id = a.account_id and a.relevant_month = s.relevant_month
left join #technical_sessions as ses  on ses.account_id = a.account_id and a.relevant_month = ses.relevant_month
left join #top_subscription as logo on logo.logo = da.logo
left join #trials as tr on a.account_id = tr.account_id and a.relevant_month = tr.relevant_month
left join #upsell_accounts as ua on ua.account_id = da.account_id
left join #cbit as cb on cb.account_id = a.account_id
left join #contacts as ct on ct.accountid = a.account_id and a.relevant_month = ct.relevant_month
left join #contracts as cr on cr.accountid = a.account_id and a.relevant_month = cr.relevant_month
left join #opportunities as op on op.accountid = a.account_id and a.relevant_month = op.relevant_month
left join #qoe as q on q.account_id = a.account_id and a.relevant_month = q.relevant_month
left join #time_from_renewal as tf on tf.account_id = a.account_id and a.relevant_month = tf.relevant_month
left join #triggers_sessions as ts on ts.account_id = a.account_id and a.relevant_month = ts.relevant_month
left join #triggers_cases as tc on tc.account_id = a.account_id and a.relevant_month = tc.relevant_month
left join #derby as de on de.account_id = a.account_id and a.relevant_month = de.relevant_month
left join #training as tra on tra.account_id = a.account_id and a.relevant_month = tra.relevant_month
left join #servers as sv on sv.accountid = a.account_id and a.relevant_month = sv.relevant_month
left join #visits_public as vpu on vpu.account_id = a.account_id and a.relevant_month = vpu.relevant_month
left join #visits_private as vpr on vpr.account_id = a.account_id and a.relevant_month = vpr.relevant_month
where a.sales_business_unit <> 'Cloud'
--and artifacts_count_3_months_back is not null
--and artifacts_count_4_months_back is not null
--and artifacts_count_5_months_back is not null
--and da.first_changed_to_customer is not null
and da.top_subscription is not null
and da.top_subscription != ''
--and sales_business_unit != 'China'
and not (class = 0 and product like '%Enterprise%')
and ((company_age is null) or (company_age >=0))
and (seniority > 0 or seniority is null)
and (ua.previous_product is null or ua.previous_product != 'Pure Upsell')
and lower(top_subscription) not like '%mp%'
and lower(top_subscription) not like '%cloud%'
and lower(top_subscription) not like '%bucket%'
and count_ent = 0;
