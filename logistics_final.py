# -*- coding:gbk -*-

# Author:ZJH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from com_util import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

df_train = pd.read_csv("all_oil_train.csv")
test = pd.read_csv("all_oil_test.csv")
#check the numbers of samples and features
print("The train data size  is : {} ".format(df_train.shape))
print("The test data size  is : {} ".format(test.shape))

# check the missing data
miss_train = missing_data(df_train)
#  delete the missing values in the training dataset
df_train = df_train.drop(df_train[df_train.low_temp.isnull()].index)
df_train = df_train.drop(df_train[df_train.value.isnull()].index)

#  delete  missing_values in oil price
df_train = df_train.drop(df_train.loc[df_train.oil == "None"].index)
df_train = df_train.drop(df_train.loc[df_train.oil == "-"].index)
df_train.oil = df_train.oil.astype("float")

### fill missing carlen use mode
test.carLen = test.carLen.fillna(test.carLen.mode()[0])

# fill oil price
test.loc[test.oil == "-", "oil"] = None
test.oil = test.oil.astype("float")
test.oil = test.oil.fillna(test.oil.mode()[0])

# df_train.drop("date",axis=1, inplace=True)
test.drop(["shen", "city", "id"], axis=1, inplace=True)
print("The train data size  is : {} ".format(df_train.shape))
print("The test data size  is : {} ".format(test.shape))
test_ = test

df_train.to_csv("train_0920.csv", index=False)
test.to_csv("test_0920.csv", index=False)

###########################################################################
#  delete the outliers
df_train = df_train.drop(df_train[df_train.price > 30000].index)
# log the  target price
df_train.price = np.log1p(df_train.price)
# delete the outliers in values
df_train = df_train.drop(df_train[df_train.value > 0.125 *(10 ** 8)].index)
# test = test.drop(test[test.value > 0.125 *(10 ** 8)].index)
df_train.value = np.log1p(df_train.value)
test.value = np.log1p(test.value)


train = df_train
# concat the train and test
train = pd.concat([train, test])

##########################################################################
# Feature Engineering
features = ["class", "oneClass",  "orderType", "tranType"]
for col in features:
    train[col] = train[col].astype("str")

# compute the frequency
distance = train.distance.value_counts()
carType = train.carType.value_counts()
carLen = train.carLen.value_counts()
value = train.value.value_counts()
oneClass = train.oneClass.value_counts()
class_ = train["class"].value_counts()
len_train = len(train)
train["distance_count"] = train.distance.apply(lambda x: distance[x] / len_train )
train["carLen_count"] = train.carLen.apply(lambda x: carLen[x] / len_train)
train["carType_count"] = train.carType.apply(lambda x: carType[x] / len_train)
train["value_count"] = train.value.apply(lambda x: value[x] / len_train)
train["oneClass_count"] = train.oneClass.apply(lambda x: oneClass[x] / len_train)
train["class_count"] = train["class"].apply(lambda x: class_[x] / len_train)


# the median  of longstart and latitude in class and oneclass 
train = merge_median(train, ["class"], "longStart", "longstart_median_class")
train = merge_median(train, ["class"], "latiStart", "latistart_median_class")
train = merge_median(train, ["class"], "longEnd", "longEnd_median_class")
train = merge_median(train, ["class"], "latiEnd", "latiEnd_median_class")

train = merge_median(train, ["oneClass"], "longStart", "longstart_median_oneclass")
train = merge_median(train, ["oneClass"], "latiStart", "latistart_median_oneclass")
train = merge_median(train, ["oneClass"], "longEnd", "longEnd_median_oneclass")
train = merge_median(train, ["oneClass"], "latiEnd", "latiEnd_median_oneclass")

train = merge_median(train, ["class", "oneClass"], "longStart", "longstart_median_classone")
train = merge_median(train, ["class", "oneClass"], "latiStart", "latistart_median_classone")
train = merge_median(train, ["class", "oneClass"], "longEnd", "longEnd_median_classone")
train = merge_median(train, ["class", "oneClass"], "latiEnd", "latiEnd_median_classone")

# the mean of value in class and oneclass

train = merge_mean(train, ["class"], "value", "value_mean_class")
train = merge_mean(train, ["oneClass"], "value", "value_mean_oneclass")
train = merge_median(train, ["class", "oneClass"], "value", "value_mean_classone")

# the std of value in class and oneclass

train = merge_std(train, ["class"], "value", "value_std_class")
train = merge_std(train, ["oneClass"], "value", "value_std_oneclass")
train = merge_std(train, ["class", "oneClass"], "value", "value_std_classone")

# the min and max in class and oneclass
train = merge_max(train, ["class"], "value", "value_max_class")
train = merge_max(train, ["oneClass"], "value", "value_max_oneclass")
train = merge_max(train, ["class", "oneClass"], "value", "value_max_classone")

train = merge_min(train, ["class"], "value", "value_min_class")
train = merge_min(train, ["oneClass"], "value", "value_min_oneclass")
train = merge_min(train, ["class", "oneClass"], "value", "value_min_classone")

# the mean, max, min, std of distance in class and oneclass

train = merge_min(train, ["class"], "distance", "distance_min_class")
train = merge_min(train, ["oneClass"], "distance", "distance_min_oneclass")
train = merge_min(train, ["class", "oneClass"], "distance", "distance_classone")

train = merge_max(train, ["class"], "distance", "distance_max_class")
train = merge_max(train, ["oneClass"], "distance", "distance_max_oneclass")
train = merge_max(train, ["class", "oneClass"], "distance", "distance_max_classone")

train = merge_mean(train, ["class"], "distance", "distance_mean_class")
train = merge_mean(train, ["oneClass"], "distance", "distance_mean_oneclass")
train = merge_median(train, ["class", "oneClass"], "distance", "distance_mean_classone")

train = merge_std(train, ["class"], "distance", "distance_std_class")
train = merge_std(train, ["oneClass"], "distance", "distance_std_oneclass")
train = merge_std(train, ["class", "oneClass"], "distance", "distance_std_classone")

# use cartype and carlen to feature engineer in features
features = ["carType", "carLen"]
for col in features:
    train[col] = train[col].astype("str")

train = merge_std(train, ["carType"], "distance", "distance_std_carType")
train = merge_std(train, ["carLen"], "value", "value_std_carType")

train = merge_min(train, ["carType"], "distance", "distance_min_carType")
train = merge_min(train, ["carLen"], "value", "value_min_carType")

train = merge_max(train, ["carType"], "distance", "distance_max_carType")
train = merge_max(train, ["carLen"], "value", "value_max_carType")
train = merge_mean(train, ["carType"], "distance", "distance_mean_carType")
train = merge_mean(train, ["carLen"], "value", "value_mean_carType")

train = merge_std(train, ["carType", "carLen"], "value", "value_std_car")
train = merge_max(train, ["carType", "carLen"], "value", "value_max_car")
train = merge_min(train, ["carType", "carLen"], "value", "value_min_car")
train = merge_mean(train, ["carType", "carLen"], "value", "value_mean_car")

train = merge_std(train, ["carType", "carLen"], "distance", "distance_std_car")
train = merge_max(train, ["carType", "carLen"], "distance", "distance_max_car")
train = merge_min(train, ["carType", "carLen"], "distance", "distance_min_car")
train = merge_mean(train, ["carType", "carLen"], "distance", "distance_mean_car")

# compine distance with value
train["dist_value"] = train.distance * train.value

# use location to cluster to feature engineering
clf = KMeans(n_clusters=27,random_state=1)
data=train[["latiStart","longStart"]].values
clf.fit(data)
train["where_1"] = pd.Series(clf.labels_)

clf = KMeans(n_clusters=27,random_state=1)
data=train[["latiEnd","longEnd"]].values
clf.fit(data)
train["where_2"] = pd.Series(clf.labels_)

# the mean,max,min,std in value and distance
train = merge_std(train, ["where_1", "where_2"], "distance", "distance_std_where")
train = merge_max(train, ["where_1", "where_2"], "distance", "distance_max_where")
train = merge_min(train, ["where_1", "where_2"], "distance", "distance_min_where")
train = merge_mean(train, ["where_1", "where_2"], "distance", "distance_mean_where")

train = merge_std(train, ["where_1", "where_2"], "value", "value_std_where")
train = merge_max(train, ["where_1", "where_2"], "value", "value_max_where")
train = merge_min(train, ["where_1", "where_2"], "value", "value_min_where")
train = merge_mean(train, ["where_1", "where_2"], "value", "value_mean_where")
train.drop(["where_1", "where_2"], axis=1, inplace=True)

# one_hot encoding
features = ["oneClass",  "orderType", "tranType"]
for col in features:
    train[col] = train[col].astype("str")
suburb_dummies = pd.get_dummies(train[features], drop_first=True)
train = train.drop(features, axis=1).join(suburb_dummies)

# high_temp and low_temp in this day
train.high_temp = train.high_temp.replace('--', train.high_temp.mode()[0])
test.high_temp = test.high_temp.replace('--', test.high_temp.mode()[0])
train.low_temp = train.low_temp.replace('--', train.low_temp.mode()[0])
test.low_temp = train.low_temp.replace('--', test.low_temp.mode()[0])
test.low_temp = test.low_temp.fillna(0)

train.low_temp = train.low_temp.astype(float)
train.high_temp = train.high_temp.astype(float)
test.high_temp = test.high_temp.astype(float)
test.low_temp = test.low_temp.astype(float)
# Mapping Temp
train.loc[train.low_temp <= -15.833, 'low_temp']  = 0
train.loc[(train.low_temp > -15.833) & (train.low_temp <= -6.667), 'low_temp'] = 1
train.loc[(train.low_temp > -6.667) & (train.low_temp <= 2.5), 'low_temp'] = 2
train.loc[(train.low_temp > 2.5) & (train.low_temp <= 11.667), 'low_temp'] = 3
train.loc[(train.low_temp > 11.667) & (train.low_temp <= 20.833), 'low_temp'] = 4
train.loc[train.low_temp > 20.833, 'low_temp'] = 5
train.loc[train.high_temp <= -5, 'high_temp']  = 0
train.loc[(train.high_temp > -5) & (train.high_temp <= 4), 'high_temp'] = 1
train.loc[(train.high_temp > 4) & (train.high_temp <= 13), 'high_temp'] = 2
train.loc[(train.high_temp > 13) & (train.high_temp <= 22), 'high_temp'] = 3
train.loc[(train.high_temp > 22) & (train.high_temp <= 31), 'high_temp'] = 4
train.loc[train.high_temp > 31, 'high_temp'] = 5
# Mapping weather condition 
weather_replace = {
     '多云': 0, '少云': 0, '阴' : 0, '晴':0,'局部多云': 0,'--':0, '-':0,'阴天':0,# good
    '小雨': 1, '阵雨': 1, '零散阵雨': 1, '雷阵雨': 1,'中雨' : 1,  '小到中雨':1,'小雨-中雨':1,'雨':1,'刮风':1,
    '中到大雨': 2, '大雨' : 2, '大到暴雨':2, '暴雨': 2, '冻雨':2,'雪':2,
    '雨夹雪':3, '小雪':3, '中雪':3, '阵雪': 3, '小到中雪':3,'小雪-中雪':3,'中雪-大雪':3,
    '雾':4, '霾':4, '浮尘':4, '大雪':4, '暴雪':4, '扬沙':4, '中到大雪':4,'大雪-暴雪':4
}

train["Simple_weather"] = train.weather.replace(
    weather_replace
)

train = train.drop("weather", axis=1)
# Mapping direction
direct_mapping = {'南风':0,  '东南风':1,  '东风':0,  '北风':0, '东北风':1, '西北风':1, '西风':0, '无持续风向':2, '西南风':1, '西北偏西风':3, '东北偏北风':3,
                 '西南偏西风':3, '西北偏北风':3, '东南偏南风':3, '微风':2, '东北偏东风':3, '西南偏南风':3, '东南偏东风':3,'--':3,  '7月10日':3, '东北偏东风':3}
train.wind = train.wind.map(direct_mapping).astype(int)

force_map = {
    '≤3级':0,  '3-4级':0,  '4-5级':1,  '<3级':0,  '4～5级':1,  '3～4级':0,  '5-6级':1,  '6-7级':2, '6～7级':2, '5～6级':1,  '4' : 0, '3' : 0 , '2' : 0, '--':0,
'1':0,  '5':1,  '6':2, '1月2日':0
}
train.force = train.force.map(force_map)
features = ["Simple_weather", "force", "wind", "high_temp", "low_temp"]
for col in features:
    train[col] = train[col].astype("str")
suburb_dummies = pd.get_dummies(train[features], drop_first=True)
train = train.drop(features,axis=1).join(suburb_dummies)

le = LabelEncoder()
features = ["carType", "carLen", "class"]
for col in features:
    train[col] = train[col].astype("str")
    train[col] = le.fit_transform(train[col])

miss = missing_data(train)
miss_feats = miss[(miss.Total >= 1) & (miss.Total < 61)].index.tolist()
for col in miss_feats:
    train[col] = train[col].fillna(train[col].mean())

numeric_features = train.dtypes[train.dtypes != "object"].index.tolist()
numeric_features = [col for col in numeric_features if col != "price"]
# Check the skew of all numerical features
skewed_feats = train[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box and Cox transform".format(skewness.shape[0]))
lam = 0.15
skewed_features = skewness.index
for feat in skewed_features:
    train[feat] = boxcox1p(train[feat], lam)
scaler = StandardScaler()
scaler.fit(train[numeric_features])
scaled = scaler.transform(train[numeric_features])

for i, col in enumerate(numeric_features):
    train[col] = scaled[:, i]
## split the training dataset and test set
all_train=train[train['price'].notnull()]
all_test=train[train['price'].isnull()]

print("\nThe train data size after feature engineering is : {} ".format(all_train.shape)) 
print("The test data size after feature engineering is : {} ".format(all_test.shape))

target = all_train.price.values
feats = [col for col in all_train.columns if (col != "date") ]
train = all_train[feats]
test = all_test[feats]

feats = [col for col in train.columns if (col != "price") ]
train = train[feats]
test = test[feats]
all_train.to_csv("all_train_feature.csv", index=False)
all_test.to_csv("all_test_feature.csv", index=False)               
##############################################################################
#############################Building the Model###############################


# Validation fuction
n_folds = 5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    mse = -cross_val_score(model, train, target, scoring="neg_mean_squared_error", cv=kf)
    return np.sqrt(mse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# Averge Model
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    # we define clones of the original models to fit the data 
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned based model
        for model in self.models_:
            model.fit(X, y)
        return self
    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ]
            
        )
        return np.mean(predictions, axis=1)
# Stacking Model
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

score = rmse_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

with open("score.txt", "a") as f:
    f.write("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmse_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
with open("score.txt", "a") as f:
    f.write("\nENet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(GBoost)

with open("score.txt", "a") as f:
    f.write("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmse_cv(model_xgb)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


with open("score.txt", "a") as f:
    f.write("\nmodel_xgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmse_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

with open("score.txt", "a") as f:
    f.write("\nmodel_lgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

averaged_models = AveragingModels(models=(GBoost, model_lgb, model_xgb))
score = rmse_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

with open("score.txt", "a") as f:
    f.write("\nAveraged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, model_xgb),
                                                 meta_model = lasso)
stacked_averaged_models.fit(train.values, target)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
rmse_stacked = mean_squared_error(target, stacked_train_pred)
print("Stacked RMSE:", np.sqrt(rmse_stacked))
with open("score.txt", "a") as f:
    f.write("\nStacked RMSE: {:.4f}\n".format(np.sqrt(rmse_stacked)))

model_xgb.fit(train.values, target)
xgb_train_pred = model_xgb.predict(train.values)
xgb_pred = np.expm1(model_xgb.predict(test.values))
rmse_xgb = np.sqrt(mean_squared_error(target, xgb_train_pred))
print("Xgboost RMSE", rmse_xgb)


with open("score.txt", "a") as f:
    f.write("\nXgboost RMSE: {:.4f}\n".format(rmse_xgb))

model_lgb.fit(train.values, target)
lgb_train_pred = model_lgb.predict(train.values)
lgb_pred = np.expm1(model_lgb.predict(test.values))
rmse_lgb = np.sqrt(mean_squared_error(target, lgb_train_pred))
print("LightGBM RMSE", rmse_lgb)

with open("score.txt", "a") as f:
    f.write("\nLightGBM RMSE: {:.4f}\n".format(rmse_lgb))

averaged_models.fit(train.values, target)
averaged_models_train_pred = averaged_models.predict(train.values)
averaged_models_pred = np.expm1(averaged_models.predict(test.values))
rmse_aver = np.sqrt(mean_squared_error(target, averaged_models_train_pred))
print("Average RMSE:", rmse_aver)

with open("score.txt", "a") as f:
    f.write("\nAverage RMSE: {:.4f}\n".format(rmse_aver))

'''MSE on the entire Train data when averaging'''

print('MSE score on train data:')
print("Stack Model RMSE", np.sqrt(mean_squared_error(target,stacked_train_pred*0.8 +
               xgb_train_pred*0.12 + lgb_train_pred*0.02 +  averaged_models_train_pred * 0.06)))


with open("score.txt", "a") as f:
    f.write("\nStack Model RMSE: {:.4f}\n".format(np.sqrt(mean_squared_error(target, stacked_train_pred*0.8 +
               xgb_train_pred*0.12 + lgb_train_pred*0.02 +  averaged_models_train_pred * 0.06))))


ensemble = stacked_pred*0.80 + xgb_pred*0.12 + lgb_pred*0.02 + averaged_models_pred * 0.06

sub = test_
sub['price'] = ensemble
sub.value = np.expm1(sub.value)
sub.to_csv('submission_0920.csv',index=False)