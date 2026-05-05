
# IMPORTANT:
# file structure should be as follows:
# main.py (this file)
# datasets/test_salary.csv
# datasets/train_salary.csv

# ALSO: all the decisions taken here are described in commecnted parts of code

# NOTE: there's a misleading example in Kaggle submition description: it asks to load a csv file with "ID" column ("ID" in capital letters), while in reality Kaggle throws an error and demands column to be named "id" (in small letters). also, as far as i understand, the contest expects contestants to predict salary, not "probabilities" as stated in the description (what's even "probabilities" here..?)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")


SEED = 123

train_df = pd.read_csv("datasets/train_salary.csv").drop(columns=["Unnamed: 0"], inplace=False)
test_df = pd.read_csv("datasets/test_salary.csv")

#train_df["ind"] = [ i for i in range(train_df.shape[0]) ]
#test_df["ind"] = [ i for i in range(test_df.shape[0]) ]


# =================== EDA ===================

'''
print(train_df.shape)
print(test_df.shape)
print()

print(train_df.columns)
print()
print(test_df.columns)
print()


print(train_df.dtypes)
print(test_df.dtypes)
print()

print(train_df[["salary_mean_net"]].describe())


print(train_df.isna().sum())
print()
print(test_df.isna().sum())

print(train_df[train_df["salary_mean_net"] <= 0].shape)
print()


print(train_df["schedule_name"].value_counts())
print()
print(train_df["experience_name"].value_counts())

plt.hist(train_df["salary_mean_net"])
plt.savefig("plot1.png")

l1 = len([i for i in train_df["key_skills_name"] if "," in i])
l2 = len([i for i in train_df["key_skills_name"] if "\n" in i])
print(l1, l2, len(list(train_df["key_skills_name"])))

print()
print("remote")
print(train_df[train_df["schedule_name"] == "Удаленная работа"].describe())
print("вахта")
print(train_df[train_df["schedule_name"] == "Вахтовый метод"].describe())
print("full day")
print(train_df[train_df["schedule_name"] == "Полный день"].describe())
print("shift")
print(train_df[train_df["schedule_name"] == "Сменный график"].describe())
print("felxible")
print(train_df[train_df["schedule_name"] == "Гибкий график"].describe())


'''

# ===========================================



# some preprocessing + feature engineering 
def df_modification(df, is_train=True, target_col="salary_mean_net"):
    df = df.copy()
    y = None

    if not is_train:
        df = df[ ["id"] + [i for i in df.columns if i != "id"] ]

    if is_train:
        # applying logarith as target has a long right tail (run code adn see plot1.png)
        y = np.log1p(df[target_col].clip(lower=0))
        df.drop(columns=[target_col], inplace=True)

    # cleaning a-la NaNs and such stuff
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(["Не указано", "не указано", "[]", "nan", "None"], np.nan)


    # feature engineering
    df["desc_len"] = df["lemmaized_wo_stopwords_raw_description"].fillna("").str.len()  # let's make an assumption that longer the description adn vacancy name are - the better the jo b is 
    df["name_len"] = df["name"].fillna("").str.len()  # same here

    df["skill_count"] = df["key_skills_name"].fillna("[]").str.count(",") + 1  # num of skills (if just one skill provided (the vast majotiry of rows) => +1 covers that)
    df.loc[df["key_skills_name"].fillna("[]") == "[]", "skill_count"] = 0
    df["has_skills"] = (df["skill_count"] > 0).astype(int)

    # assuming that employees working by вахта method earn more that those working on a remote job (and other types of jobs )
    df["is_remote"] = (df["schedule_name"] == "Удаленная работа").astype(int)
    df["is_vakhta"] = (df["schedule_name"] == "Вахтовый метод").astype(int)
    df["is_shift"] = (df["schedule_name"] == "Сменный график").astype(int)
    df["is_fullday"] = (df["schedule_name"] == "Полный день").astype(int)
    df["is_flexible"] = (df["schedule_name"] == "Гибкий график").astype(int)

    # dropping raw text aand just redunsdant columsn (like ID)
    drop_cols = ["id", 
                 "employer_name", 
                 "raw_description", 
                 "raw_branded_description",
                 "lemmaized_wo_stopwords_raw_description",
                 "lemmaized_wo_stopwords_raw_branded_description",
                 "key_skills_name", 
                 "languages_name"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # filling remaining missing values (if any left)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("nan")
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(0)

    # transfroming categorical cols
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "employer_id" in df.columns:
        df["employer_id"] = df["employer_id"].astype(str)  # converting to str as employer id as num makes no sense and confuses the tree models
        if "employer_id" not in cat_cols:
            cat_cols.append("employer_id")

    # convertign to pandas "category" dtype (for faster operations and safe xg boost)
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df, y, cat_cols
    

# ensemble: cat boost + xg boost (5 fold cross-val)
def run_cv_ensemble(X_tr, y_tr, X_te, cat_cols, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # preparinh for cross-val 
    pred_train_cat = np.zeros(len(X_tr))  # "predicted" for train (predicted := cross-val applied)
    pred_train_xgb = np.zeros(len(X_tr))  # same here (xgb)
    
    pred_test_cat = np.zeros(len(X_te))  # predicted for test via catboost
    pred_test_xgb = np.zeros(len(X_te))  # via xgb

    cb_params = {
        "iterations": 600, 
        "learning_rate": 0.05, 
        "depth": 6, 
        "l2_leaf_reg": 5.0,
        "loss_function": "MAE", 
        "eval_metric": "MAPE", 
        "cat_features": cat_cols,
        "verbose": False, 
        "random_seed": SEED
    }

    xgb_params = {
        "n_estimators": 350, 
        "learning_rate": 0.05, 
        "max_depth": 6,
        "subsample": 0.85, 
        "colsample_bytree": 0.85, 
        "reg_alpha": 0.2,
        "reg_lambda": 1.5, 
        "objective": "reg:squarederror", 
        "eval_metric": "mape",
        "random_state": SEED, 
        "n_jobs": -1, 
        "verbosity": 0
    }

    # fold = tuple index; tuple = (indecies of train rows, indicies of validation rows)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_tr_f, X_val_f = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_tr_f, y_val_f = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        
        X_tr_xgb = X_tr_f.copy()
        X_val_xgb = X_val_f.copy()
        X_te_xgb  = X_te.copy()

        # extracting categories as numbrs (xg boost was previously throwing errors becuase of cyrillic characters) + xgb requires manual encoding
        for col in cat_cols:
            X_tr_xgb[col] = X_tr_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
            X_te_xgb[col]  = X_te_xgb[col].cat.codes
        
        # cat boost (auto encoding)
        cb_model = cb.CatBoostRegressor(**cb_params)
        cb_model.fit(X_tr_f, y_tr_f, eval_set=(X_val_f, y_val_f), early_stopping_rounds=50)
        pred_train_cat[val_idx] = cb_model.predict(X_val_f)
        
        pred_test_cat += cb_model.predict(X_te) / n_splits

        
        # xg boost
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr_xgb, y_tr_f, eval_set=[(X_val_xgb, y_val_f)], verbose=False)
        pred_train_xgb[val_idx] = xgb_model.predict(X_val_xgb)
        
        pred_test_xgb += xgb_model.predict(X_te_xgb) / n_splits

        # tracking MAPE after applying exp (as target has been logarithmed)
        y_orig = np.expm1(y_val_f).clip(lower=1.0)
        pred_cat_orig = np.maximum(np.expm1(pred_train_cat[val_idx]), 1.0)
        val_mape = mean_absolute_percentage_error(y_orig, pred_cat_orig)
        print(f"fold num: {fold + 1 } -> MAPE: {val_mape:.4f}")

    # findign the optimal combination: w * catboost + (1 - w) * xgboost
    best_weight_catb, best_mape = 0.5, float("inf")
    for w in np.arange(0.0, 1.05, 0.05):
        blend_log = w * pred_train_cat + (1 - w) * pred_train_xgb
        blend_orig = np.maximum(np.expm1(blend_log), 1.0)
        mape = mean_absolute_percentage_error(np.expm1(y_tr).clip(lower=1.0), blend_orig)
        if mape < best_mape:
            best_mape, best_weight_catb = mape, w

    print()
    print(f"best found catboost weight: {best_weight_catb:.2f} -> cross-val MAPE: {best_mape:.4f}")

    final_pred = best_weight_catb * pred_test_cat + (1 - best_weight_catb) * pred_test_xgb
    
    return final_pred



X_train, y_train, cat_cols = df_modification(train_df, is_train=True)
X_test, _, _ = df_modification(test_df, is_train=False)

X_test = X_test[X_train.columns]

log_predicts = run_cv_ensemble(X_train, y_train, X_test, cat_cols)


predicts = np.expm1(log_predicts)
print([i for i in predicts if i <= 0])  # turns out, no nulls
predicts = np.maximum(predicts, 1.0)

predictions = pd.DataFrame({
    "id": test_df["id"],
    "salary_mean_net": predicts
})

predictions.to_csv("predicts_3.csv", index=False)



print(predictions.head())



