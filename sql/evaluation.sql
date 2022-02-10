select rating, territory, month,
       avg(upsell)::float as CR,
       count(*) as count,
       sum(upsell) as n_upsells
from
(select scores.*, cast(date_trunc('month', ops.createddate) as date) as month, case when contract_type__c = '' then type else contract_type__c end as status,
        case when territory__c = 'Americas' then 'Americas' else 'EMEA/APAC' end as territory,
        case when (status = 'JFrog Pro X' or product_type__c = 'JFrog Pro X' ) then 1.000 else 0.000 end as upsell
from
((select a.account_id, b.rating,
         min(insert_date) as first_scoring_date
from
(select account_id, case when (max(prob)>0.5 and max(prob)<0.625) then 'medium'
                         when max(prob)>=0.625 then 'high'
                         else 'low' end as rating
from data_science.pro_to_pro_x_prediction
group by 1) as a
join (select distinct account_id, insert_date, case when prob>0.5 and prob<0.625 then 'medium'
                                                    when prob>=0.625 then 'high'
                                                    else 'low' end as rating
      from data_science.pro_to_pro_x_prediction) as b
on a.account_id = b.account_id and a.rating = b.rating
group by 1,2) as scores
left join salesforce.account
on account.id = scores.account_id
left join
(select accountid, createddate, product_type__c
 from salesforce.opportunity) as ops
on (scores.account_id = ops.accountid and scores.first_scoring_date < ops.createddate)))
where month >= '2021-02-01'
group by 1,2,3