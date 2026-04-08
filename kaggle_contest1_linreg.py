
# IMPORTANT:
# file structure should be as follows:
# main.py (this file)
# datasets/prices_test.csv
# datasets/prices_train.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import preprocessing


train_df = pd.read_csv("datasets/prices_train.csv").drop("Unnamed: 0", axis=1) #.fillna(0)
test_df = pd.read_csv("datasets/prices_test.csv").drop("Unnamed: 0", axis=1) #.fillna(0)

new_cols_test = {col: col.strip().lower().replace(" ", "_") for col in test_df.columns}
new_cols_train = {col: col.strip().lower().replace(" ", "_") for col in train_df.columns}

train_df = train_df.rename(columns=new_cols_train)
test_df = test_df.rename(columns=new_cols_test)

# ======== EDA goes brrrr ========
# print(train_df.isna().sum())
# print(test_df.isna().sum())

# the hell is going on with the date???
# print( train_df[train_df["x1_transaction_date"] == 2012.833] )

#plt.scatter(x=train_df["x3_distance_to_the_nearest_mrt_station"], y=train_df["y_house_price_of_unit_area"])
#plt.savefig("temp_plot.png")

#plt.scatter(x=train_df["x2_house_age"], y=train_df["y_house_price_of_unit_area"])
#plt.savefig("temp_plot.png")

#plt.hist(x=np.log1p(train_df["x2_house_age"]), bins=50)
#plt.savefig("hist_plot.png")

#plt.hist(x=np.log1p(train_df["y_house_price_of_unit_area"]), bins=50)
#plt.savefig("hist_plot.png")
# ================================

EARTH_RAD = 6371
RANDOM_SEED = 123

# handling outliers
IQR = train_df["y_house_price_of_unit_area"].quantile(0.75) -  train_df["y_house_price_of_unit_area"].quantile(0.25)
lower = train_df["y_house_price_of_unit_area"].quantile(0.25) - IQR * 1.5
upper = train_df["y_house_price_of_unit_area"].quantile(0.75) + IQR * 1.5

train_df = train_df[ 
  (train_df["y_house_price_of_unit_area"] > lower) &
  (train_df["y_house_price_of_unit_area"] < upper)
]


def df_modification(df):
  df = df.copy()

  # transforming raw cooords into the distance from the house to the city center (city center = Teipei central metr station)
  lat_center, lon_center = 25.048753, 121.514228  # according to Yandex, these are the coords of teh central Teipei metro staion 
  df["lat_rad"] = np.radians(df["x5_latitude"])
  df["lon_rad"] = np.radians(df["x6_longitude"])
  df["center_lat_rad"] = np.radians(lat_center)
  df["center_lon_rad"] = np.radians(lon_center)

  dlat = df["lat_rad"] - df["center_lat_rad"]
  dlon = df["lon_rad"] - df["center_lon_rad"]
  
  # some fancy formula to caluclate the distance in km having raw coords 
  d = 2 * np.arcsin(np.sqrt(
    np.sin( (df["lat_rad"] - df["center_lat_rad"]) / 2 )**2 + np.cos(df["lat_rad"]) * np.cos(df["center_lat_rad"]) * np.sin( (df["lon_rad"] - df["center_lon_rad"]) / 2 )**2
  ))
  df["distance_to_center"] = EARTH_RAD * d
  
  # using ln() here (as scatterploting the columns "x3_distance..." and "y_price..." has shown that the data folows some sort of a log-normal distrivution)
  df["x3_distance_log"] = np.log1p(df["x3_distance_to_the_nearest_mrt_station"])
  df["x2_age_log"] = np.log1p(df["x2_house_age"])

  # combining disatnce and hous age (so older and farther located houses are penalized more than newer and closer located ones)
  df["distance_age"] = df["x3_distance_to_the_nearest_mrt_station"] * df["x2_house_age"]

  # concetnration of stores regarding the distance form center (+1 to avid division by zero )
  df["store_density"] = df["x4_number_of_convenience_stores"] / (df["x3_distance_to_the_nearest_mrt_station"] + 1)

  #df["year_built"] = df["x1_transaction_date"].apply( lambda row: int(str(row)[:4]) ) - df["x2_house_age"].apply( lambda row: int(row) if not math.isnan(float(row)) else 0 )
  #print(df["year_built"])
  
  # clening up 
  df = df.drop(columns=[
    "lat_rad", 
    "lon_rad", 
    "center_lat_rad", 
    "center_lon_rad",
    "x3_distance_to_the_nearest_mrt_station",
    "x2_house_age",
  ])
  
  return df


train_df = df_modification(train_df)
test_df = df_modification(test_df)

print(train_df.columns)


# imputation + training 
imputer = IterativeImputer(estimator=Ridge(), initial_strategy="median", random_state=RANDOM_SEED)
scaler = StandardScaler()
polynom = PolynomialFeatures(degree=2, include_bias=False)
ridge = Ridge(alpha=25)

model = make_pipeline(imputer, scaler, polynom, ridge)

X_train = train_df.drop(columns=["y_house_price_of_unit_area"]).copy()
y_train = np.log1p(train_df["y_house_price_of_unit_area"].copy())

X_test = test_df.copy()

# 5 fold cross-validaation 
def calc_mse(y_true, y_pred):
  # exp required as previous;y logarithm has been applieed
  y_true_orig = np.expm1(y_true)
  y_pred_orig = np.expm1(y_pred)
  return mean_squared_error(y_true_orig, y_pred_orig)

scorer = make_scorer(calc_mse, greater_is_better=False)  # small erro = good
scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
print(f"cross-validation MSE: {-scores.mean():.6f} (+/- {scores.std():.6f})")


model.fit(X_train, y_train)
predictions = np.expm1(model.predict(X_test))


pd.DataFrame(
  {
    "index": [ i for i in range(len(predictions)) ],
    "Y house price of unit area": predictions,
  }
).to_csv("predicts.csv", index=False)



print(len(list(predictions)))
