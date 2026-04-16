
# IMPORTANT:
# file structure should be as follows:
# main.py (this file)
# datasets/test.csv
# datasets/train.csv

# ALSO: all the decisions taken here are described in commecnted parts of code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")


new_cols_test = {col: col.strip().lower().replace(" ", "_") for col in test_df.columns}
new_cols_train = {col: col.strip().lower().replace(" ", "_") for col in train_df.columns}

train_df = train_df.rename(columns=new_cols_train)
test_df = test_df.rename(columns=new_cols_test)


"""
NEW COLUMN NAMES:
"name", "gender", "age", "city", "working_professional_or_student",
   "profession", "academic_pressure", "work_pressure", "cgpa",
   "study_satisfaction", "job_satisfaction", "sleep_duration",
   "dietary_habits", "degree", "have_you_ever_had_suicidal_thoughts_?",
   "work/study_hours", "financial_stress",
   "family_history_of_mental_illness", "depression", "id"
"""

# =================== EDA ===================
print(train_df.shape) 
#print(train_df.dtypes)
print(test_df.shape)
#print(test_df.dtypes)
"""
print()
print(train_df.isna().sum())
print()
print(test_df.isna().sum())
print()
"""

# IMPORTANT !!!!11!!1!1!!
# some coluns related to either academic performance or job-related stuff have null values BUT it doesn"t mean the"r missing!!! this happens as no students work and no working individuals study => students cannot have "work pressure", etc. and working folks cannot have "cgpa", etc. however, some working individuals are indeed missinf a job (even though they're "working"), that;s 129 rows in total. for proofs, uncomment and run the coed below
'''
print(
  train_df[train_df["working_professional_or_student"] == "Student"].shape[0],  # 382
  train_df[train_df["working_professional_or_student"] == "Working Professional"].shape[0],  # 1509
)
print()

print(train_df[train_df["working_professional_or_student"] == "Student"].isna().sum())  # 382 values missing at each job-related column, just as expected
print()
print(train_df[train_df["working_professional_or_student"] == "Working Professional"].isna().sum())  # 1509 values missing at each study-re;eated column, just as expected
print()
print(train_df[(train_df["working_professional_or_student"] == "Student") & (~train_df["job_satisfaction"].isna())])  # double-checking: retruns empty df as, again, studeitns do not work!
'''

# also, since there are too many cities and professions, neither one hot encodre, nor label encder are suitable here => encoding via frequency
print(train_df["city"].nunique())  # 30
print(train_df["profession"].nunique())  # 35
print()

# testing whther any new categorical values appeeared in test that are not presented in train
print(
    set(list(test_df["city"])).issubset(set(list(train_df["city"]))),
    set(list(test_df["profession"])).issubset(set(list(train_df["profession"]))),
    set(list(test_df["sleep_duration"])).issubset(set(list(train_df["sleep_duration"]))),
    set(list(test_df["dietary_habits"])).issubset(set(list(train_df["dietary_habits"]))),
    set(list(test_df["gender"])).issubset(set(list(train_df["gender"]))),
    set(list(test_df["have_you_ever_had_suicidal_thoughts_?"])).issubset(set(list(train_df["have_you_ever_had_suicidal_thoughts_?"]))),
    set(list(test_df["family_history_of_mental_illness"])).issubset(set(list(train_df["family_history_of_mental_illness"]))),
    set(list(test_df["degree"])).issubset(set(list(train_df["degree"]))),
)  # all presented => no new values

# ===========================================


def df_modification():
    train = train_df.copy()
    test = test_df.copy()

    train.drop(columns=["id", "name"], inplace=True, errors="ignore")
    test.drop(columns=["id", "name"], inplace=True, errors="ignore")

    # fillig numerical nan's with -999 (to prevent confusion with other values since "-999" is far outside the scope of positive integers presented in the columns)
    num_cols = ["age", "academic_pressure", "work_pressure", "cgpa", "study_satisfaction", "job_satisfaction", "work/study_hours", "financial_stress"]
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(-999)
        test[col] = pd.to_numeric(test[col], errors="coerce").fillna(-999)

    # filling categorical
    cat_cols = ["sleep_duration", "dietary_habits", "gender", "profession", "degree", "city", "have_you_ever_had_suicidal_thoughts_?", "family_history_of_mental_illness", "working_professional_or_student"]
    for col in cat_cols:
        train[col] = train[col].fillna("nan").astype(str)
        test[col] = test[col].fillna("nan").astype(str)

    y_train = train["depression"]
    train.drop(columns="depression", inplace=True, errors="ignore")

    # encdoding of categorical features is orchestrated via both frequency encoding (frequency is used for both city and profession cols; for details see EDA) and manual encoding (the rest of the columns). both cross-val and kaggle stats confirmed that OHE and LE show worse prediciton results than manual encoding
    # frequency encoding:
    for col in ["city", "profession"]:
        freq_enc = train[col].value_counts(normalize=True).to_dict()
        train[col] = train[col].map(freq_enc).fillna(0.0)
        test[col] = test[col].map(freq_enc).fillna(0.0)

    # manual encdoing (with dictiory)
    enc_dict = {
        "sleep_duration": {
            "Less than 5 hours": 0, 
            "5-6 hours": 1, 
            "7-8 hours": 2, 
            "More than 8 hours": 3
        },
        "dietary_habits": {
            "Unhealthy": 0, 
            "Moderate": 1, 
            "Healthy": 2
        },
        "gender": {
            "Male": 0, 
            "Female": 1
        },
        "have_you_ever_had_suicidal_thoughts_?": {
            "Yes": 1,
            "No": 0
        },
        "family_history_of_mental_illness": {
            "Yes": 1, 
            "No": 0
        },
        "degree": {
            "B.Com": 0, 
            "B.Ed": 1, 
            "BCA": 2, 
            "Class 12": 3, 
            "MCA": 4
        },
        "working_professional_or_student": {
            "Student": 0,
            "Working Professional": 1
        }
    }
    for col, encoding in enc_dict.items():
        train[col] = train[col].map(encoding).fillna(-1)
        test[col] = test[col].map(encoding).fillna(-1)
    
    return train, test, y_train


X_train, X_test, y_train = df_modification()


model = RandomForestClassifier(
    n_estimators=250,
    max_depth=9,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 5 fold cross-val
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba_cv = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


# searching for the most optimal threshhold
best_f1, best_thresh = 0, 0.5
for thresh in np.linspace(0.05, 0.5, 50):
    preds = (y_proba_cv >= thresh).astype(int)
    f1 = f1_score(y_train, preds)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

print()
print(f"optimal thresh found: {best_thresh:.4f}")
print(f"cross-val f1_socre: {best_f1:.4f}")


# trainf and predictiong
model.fit(X_train, y_train)
y_proba_test = model.predict_proba(X_test)[:, 1]
predictions = (y_proba_test >= best_thresh).astype(int)

pd.DataFrame({
    "id": test_df["id"],
    "depression": predictions
}).to_csv("predicts_2.csv", index=False)


print()
print("most importtant features")
feat_imp = pd.Series(
    model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print(feat_imp)#.head(5))







