import h2o
import os
import time

h2o.init() #init h2o cluster

#load saved model

all_features_model_path = os.path.join(os.getcwd(),"../models/", "depth_grid_model_6")


all_features_model = h2o.load_model(all_features_model_path)


#GBM
print("Final model validation stats:")
print('R2', all_features_model.r2(valid = False))
print('RMSE', all_features_model.rmse(valid = False))
#print('MAE', all_features_model.mae(valid = True))


all_features = "survey_metropol_without_scores_test.csv"


all_features_path = os.path.join(os.getcwd(),"../data/flatfile/", all_features)



# Load data into H2O
df_all_features = h2o.import_file(all_features_path)


#feature interactions
interacting_variables1 = ["income_estimated_amount.group","column_count.group"]
interacting_variables2 = ["portion_active.group","other_non-farm_income.binary"]
interacting_variables3 = ["amounts_of31accts.group","order_value.group"]
interacting_variables4 = ["portion_closed.group","amounts_of31accts.group"]
interacting_variables5 = ["income_estimated_amount.group","order_value.group"]
interacting_variables6 = ["loans_type.inputs_mech",'cooperative_member']
interacting_variables7 = ['cooperative_member','average_overdue_bal.group']
interacting_variables8 = ['average_overdue_bal.group', 'order_value.group']
interacting_variables9 = ['average_original_amount.group',"order_value.group"]
interacting_variables10 = ['income_estimated_amount.group','yield_kg.group']

train_cols1 = df_all_features.interaction(factors=interacting_variables1,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols2 = df_all_features.interaction(factors=interacting_variables2,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols3 = df_all_features.interaction(factors=interacting_variables3,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
          min_occurrence=100)

train_cols4 = df_all_features.interaction(factors=interacting_variables4,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols5 = df_all_features.interaction(factors=interacting_variables5,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols6 = df_all_features.interaction(factors=interacting_variables6,    #Generate pairwise columns
           pairwise=False,
           max_factors=1000,
           min_occurrence=100)


train_cols7 = df_all_features.interaction(factors=interacting_variables7,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols8 = df_all_features.interaction(factors=interacting_variables8,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols9 = df_all_features.interaction(factors=interacting_variables9,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)

train_cols10 = df_all_features.interaction(factors=interacting_variables10,    #Generate pairwise columns
          pairwise=False,
           max_factors=1000,
           min_occurrence=100)


df_all_features = df_all_features.cbind(train_cols1)
#df_all_features = df_all_features.cbind(train_cols2) 
#df_all_features = df_all_features.cbind(train_cols3)
#df_all_features = df_all_features.cbind(train_cols4)
#df_all_features = df_all_features.cbind(train_cols5)
df_all_features = df_all_features.cbind(train_cols6)  
# df_all_features = df_all_features.cbind(train_cols7)  
df_all_features = df_all_features.cbind(train_cols8)
# df_all_features = df_all_features.cbind(train_cols9)
df_all_features = df_all_features.cbind(train_cols10)  


print("Detected columns: {}".format(len(df_all_features.columns)))

print(df_all_features.columns)

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
best_gbm_perf2 = all_features_model.model_performance(df_all_features)

print(best_gbm_perf2.auc())

#produce predictions for arrears
all_features_status_preds = all_features_model.predict(df_all_features)



df_all_features_result = df_all_features.cbind(all_features_status_preds)


#output_name = "predictions_{}_{}".format(int(time.time()),csv_name)
all_features = "predictions_internal_only_2020.03.13_without_scores_without_order_value4.csv"

all_features_predictions_path = os.path.join(os.getcwd(),"../predictions/", all_features)



h2o.export_file(df_all_features_result, path = all_features_predictions_path, parts=1, force=True)

print("predictions exported to {}".format(all_features_predictions_path))



