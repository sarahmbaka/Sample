import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import os
import numpy as np
import math

#model_directory = "../models/"

# csv_name = input("Enter name of training CSV in data folder (e.g. \"training_flat_file_v2.csv\"): ")
all_features_name = "survey_metropol_without_scores_train.csv"


#valid_path = os.path.join(os.getcwd(),"../data/flatfile/", valid_name)
all_features_path = os.path.join(os.getcwd(),"../data/flatfile/", all_features_name)

#init h2o cluster
h2o.init()
#clean slate, in case cluster was already running
h2o.remove_all()      

# Load data into H2O
#valid = h2o.import_file(valid_path)
df_all_features = h2o.import_file(all_features_path)


df_all_features.name = 'all_features'

_df = df_all_features

#check the number of columns are list them
print("Detected columns: {}".format(len(_df.columns)))
print(_df.columns)

# response_var = input("Enter response variable: ")
response_var = "status_by_margins_30"
#response_var = "status_by_margins_60"

   
# excluded_colstr = input("Enter other columns to exclude eg. \"col1,col2,col3\"" : ")   #response_var will already be excuded
excluded_colstr = "derived_actual_status,total_arrears,margin,average_overdue_bal.group,cooperative_member,order_value,order_value.group,age_of_loan"  #used for business analysis
excluded_colstr += ",status_by_margins_60,derived_status_30,farmer_national_id,order_id" # alternative responser var

##### uncomment the lines below if you want to include CRB(metropol) data
# excluded_colstr += ",credit_score.group,with_credit_score,delinquency_code,is_guarantor,income_estimated_amount.group"
# excluded_colstr += ",column_count.group,acct_ids_of31accts.group,acct_bals_of31accts.group,days_arrears_of31accts.group"
# excluded_colstr += ",amounts_of31accts.group,overdue_balsof31accts.group,count_active.group,count_closed.group"
# excluded_colstr += ",count_fully_settled.group,portion_active.group"
# excluded_colstr += ",portion_closed.group,portion_fully_settled.group,average_account_bal.group"
# excluded_colstr += ",average_overdue_bal.group,average_original_amount.group"

#### Uncomment the block below to include TRANSUNION data
# excluded_colstr += ",no_of_physicaladdresses.group,no_of_postal_addresses.group,no_of_phone_contacts.group"
# excluded_colstr += ",no_of_accounts.group,open_account.group,performing.group,non_performing.group,closed.group"
# excluded_colstr += ",current_in_arrears.group,arrears_0days.group,arrears_30days.group,arrears_60days.group"
# excluded_colstr += ",arrears_90days.group,arrears_greater_90days.group,principal_amt.group"
# excluded_colstr += ",current_balance_amt.group,max_arrears.group,no_of_enquiries.group,collaterals_given.group"
# excluded_colstr += ",credit_applications_done.group,enquiries_30days.group,enquiries_60days.group"
# excluded_colstr += ",enquiries_greater_90days.group,mobile_total.group,mobile_performing.group"
# excluded_colstr += ",mobile_nonperforming.group,mobile_closed.group,mobile_current_in_arrears.group"
# excluded_colstr += ",mobile_currentbalance_amt.group,mobile_principal_amt.group,pastdue_amt.group"
# excluded_colstr += ",scheduled_paymeny_amt.group,enquiries_90days.group"

#uncoment to include VANDERSAT data
# excluded_colstr += ",W1,W2,W3,W4,W5,W6,W7,W8,W9,W10,G1,G2,G3,G4,F1"
# excluded_colstr += ",F2,F3,F4,M1,M2,M3,M4,M5"

#uncomment to include CONSERWATER data
# excluded_colstr += ",soil_moisture_mean,soil_moisture_deviation,soil_moisture_5thpercentile,soil_moisture_25thpercentile"
# excluded_colstr += ",soil_moisture_75thpercentile,soil_moisture_95thpercentile,soil_moisture_95thpercentile_5thpercentile_diff"
# excluded_colstr += ",soil_moisture_75thpercentile_25thpercentile_diff,cumdiff_lowSM,number_days_lowSM,germination_SM_mean"
# excluded_colstr += ",germination_SM_deviation,germination_SM_count,germination_low_SM,flowering_SM_mean,flowering_SM_deviation"
# excluded_colstr += ",flowering_low_SM_count,flowering_low_SM_75thpercentile_count,maturity_SM_mean,maturity_SM_75thpercentile"
# excluded_colstr += ",maturity_SM_95thpercentile,maturity_high_SM_count,maturity_high_SM_95thpercentile_count,nitrogen_mean"
# excluded_colstr += ",nitrogen_deviation,nitrogen_5thpercentile,nitrogen_25thpercentile,nitrogen_75thpercentile"
# excluded_colstr += ",nitrogen_95thpercentile,nitrogen_95thpercentile_5thpercentile_diff"
# excluded_colstr += ",nitrogen_75thpercentile_25thpercentile_diff"
# excluded_colstr += ",phosphorous_mean,phosphorous_deviation,phosphorous_5thpercentile,phosphorous_25thpercentile"
# excluded_colstr += ",phosphorous_75thpercentile,phosphorous_95thpercentile,phosphorous_95thpercentile_5thpercentile_diff"
# excluded_colstr += ",phosphorous_75thpercentile_25thpercentile_diff"

# potential reponse variables
excluded_cols = excluded_colstr.split(',')

_df[response_var] = _df[response_var].asfactor()
_df['order_value.group'] = _df['order_value.group'].asfactor()

y = response_var

#feature interactions
interacting_variables1 = ["income_estimated_amount.group","column_count.group"]
interacting_variables2 = ["portion_active.group","other_non-farm_income.binary"]
interacting_variables3 = ["amounts_of31accts.group","order_value.group"]
interacting_variables4 = ["portion_closed.group","amounts_of31accts.group"]
interacting_variables5 = ['income_estimated_amount.group','order_value.group']
interacting_variables6 = ["loans_type.inputs_mech",'cooperative_member']
interacting_variables7 = ['cooperative_member','average_overdue_bal.group']
interacting_variables8 = ['average_overdue_bal.group', 'order_value.group']
interacting_variables9 = ['average_original_amount.group',"order_value.group"]
interacting_variables10 = ['income_estimated_amount.group','yield_kg.group']

train_cols1 = _df.interaction(factors=interacting_variables1,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols2 = _df.interaction(factors=interacting_variables2,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols3 = _df.interaction(factors=interacting_variables3,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
          min_occurrence=100)

train_cols4 = _df.interaction(factors=interacting_variables4,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols5 = _df.interaction(factors=interacting_variables5,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols6 = _df.interaction(factors=interacting_variables6,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)


train_cols7 = _df.interaction(factors=interacting_variables7,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols8 = _df.interaction(factors=interacting_variables8,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols9 = _df.interaction(factors=interacting_variables9,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols10 = _df.interaction(factors=interacting_variables10,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)


_df = _df.cbind(train_cols1)
#_df = _df.cbind(train_cols2) 
#_df = _df.cbind(train_cols3)
#_df = _df.cbind(train_cols4)
#_df = _df.cbind(train_cols5)
_df = _df.cbind(train_cols6)  
#_df = _df.cbind(train_cols7)  
_df = _df.cbind(train_cols8)
# _df = _df.cbind(train_cols9)
_df = _df.cbind(train_cols10)  


#check the number of columns are list them
print("Detected columns: {}".format(len(_df.columns)))
print(_df.columns)

x = _df.columns


#remove response variable from column list
x.remove(y)

if excluded_colstr:
    #remove unwanted cols
    for col in excluded_cols:
        if col in x:
            x.remove(col.strip())
            #df2.drop(col.strip(), axis =1)
           
##GBM Model

# GBM hyperparameters
# create hyperameter and search criteria lists (ranges are inclusive..exclusive))
hyper_params_tune = {#'learn_rate': [i * 0.01 for i in range(1, 5)],  #updated
                     'max_depth': list(range(2,11)),
                     #'sample_rate': [i * 0.1 for i in range(8, 10)], #updated
                     #'col_sample_rate': [i * 0.1 for i in range(5, 10)]
                     }

gbm_grid = H2OGradientBoostingEstimator(distribution='bernoulli',
                                       seed = 1234,
                                       ntrees = 10000,
                                       #learn_rate = 0.01,
                                      # sample_rate = 0.8,
                                       col_sample_rate = 0.8,
                                       nfolds =5,
                                       stopping_metric = 'auc',
                                       stopping_rounds = 5,
                                       score_tree_interval = 10)

#Build grid search with previously made GBM and hyper parameters
grid = H2OGridSearch(gbm_grid,hyper_params_tune,
                          grid_id = 'depth_grid',
                         search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 50})


#Train grid search
grid.train(x=x,
           y=y,
           training_frame = _df
           #,validation_frame = valid
           )

sorted_final_grid = grid.get_grid(sort_by='auc', decreasing=True)
print(sorted_final_grid)

#select the best model
mymodel = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])

feature_importance = pd.DataFrame(mymodel.varimp(use_pandas=True))

#model metrics
print("Final model validation stats:")
print('R2', mymodel.r2(valid = True))
print('RMSE', mymodel.rmse(valid = True))


#get coefficient table
#h2o.coef(mymodel)

#mymodel$coefficients_table
model_path = os.path.join(os.getcwd(),"../models/")

# save the model
model_saved = h2o.save_model(model=mymodel, path=model_path, force=True)

# save feature_importance
feature_importance.to_csv(os.path.join(os.getcwd(),model_path,"internal_only_2020.03.13_feature_importance_without_scores_withoutorder_value4.csv"))


print()
print("model saved to {}".format(model_saved))

