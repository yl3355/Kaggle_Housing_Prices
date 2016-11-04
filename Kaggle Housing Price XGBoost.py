###Kaggle Housing Price XGBoost
####Using /anaconda/lib/python2.7/site-packages
#Read Files
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
###set working directory
os.chdir("/Users/youzhuliu/Desktop/kaggle/Kaggle_Housing_Prices/data")
###read files
test=pd.read_csv('test.csv')
df=pd.read_csv('train.csv')
all_names = df.columns.values

cat = ['MSSubClass', 'MSZoning','Street',
        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 
        'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
        'BsmtFinType1', 'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir',
        'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',
        'GarageFinish','GarageQual', 'GarageCond', 'PavedDrive', 
        'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
        "BsmtFullBath","HalfBath", "Fireplaces", "KitchenAbvGr", 
        "BsmtHalfBath", "TotRmsAbvGrd", "FullBath", "BedroomAbvGr","GarageCars"]

time = ['YearBuilt','YearRemodAdd', 'MoSold','YrSold','GarageYrBlt']

cont = set(all_names)-set(cat)-set(time)-set(['Id'])

#check missing values: only three variables missing:
pd.set_option('display.max_rows', 100)
df.count()

#PoolQC, MiscFeature, Alley, Fence, FireplaceQu - missing a lot: drop them
df.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], 1, inplace=True)
test.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], 1, inplace=True)
###deal with missing value in train dataset
df.replace('?', np.nan).dropna().shape

train_nomissing = df.replace('?', np.nan).dropna()
test_nomissing = test.replace('?', np.nan).dropna()
###encoding categorial column
combined_set = pd.concat([train_nomissing, test_nomissing], axis = 0)
combined_set.info()

for feature in combined_set.columns: # Loop through all columns in the dataframe
    if combined_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes # Replace strings with an integer

combined_set.info()

final_train = combined_set[:train_nomissing.shape[0]]
final_test = combined_set[train_nomissing.shape[0]:]

y_train = final_train.pop('SalePrice')
y_test = final_test.pop('SalePrice')

"""
This part needs more research
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'reg:linear'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 100, n_jobs = -1) 

optimized_GBM.fit(final_train, y_train)
"""

"""
model = xgb.XGBClassifier()
model.fit(final_train, y_train)     
print(model)
"""

xgtrain = xgb.DMatrix(final_train, y_train)

xgboost_params = { 
  "objective": "reg:linear",
  "booster": "gbtree",
  #"min_child_weight": 240,
  "subsample": 0.75,
  "colsample_bytree": 0.68,
  "max_depth": 7
  }

boost_round = 20

clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)

xgtest = xgb.DMatrix(final_test)
test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(final_train, y_train)
