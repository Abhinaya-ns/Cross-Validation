import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# 1. Load data and drop rows with missing values
data_set = pd.read_csv(r'C:\Users\abhi\Downloads\MELBOURNE_HOUSE_PRICES_LESS.csv.zip')
sample = data_set.sample(n=500, random_state=42)   # sample 1,000 rows
print(sample.columns)
sample = data_set.dropna(subset=data_set.columns[[2, 8, 10, 11, 4]])

# 2. Extract features (X) and target (y)
x = sample.iloc[:, [2, 8, 10, 11]].values
y = sample.iloc[:, 4].values
print("step1")
# 3. Setup KFold cross validation
fold = KFold(n_splits=4, shuffle=True, random_state=42)

def get_score_from_model(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    print("step2")
    return model.score(xtest, ytest)

lr_scores = []
rf_scores = []
svr_scores = []
print("step3")
# 4. Correct indexing loop
for train_index, test_index in fold.split(x, y):
    xtrain, xtest = x[train_index], x[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    print("step4")
    lr_scores.append(get_score_from_model(LinearRegression(), xtrain, xtest, ytrain, ytest))
    rf_scores.append(get_score_from_model(RandomForestRegressor(), xtrain, xtest, ytrain, ytest))
    svr_scores.append(get_score_from_model(SVR(), xtrain, xtest, ytrain, ytest))
print("step5")
print("Linear Regression Scores:", lr_scores)
print("Random Forest Scores:", rf_scores)
print("SVR Scores:", svr_scores)
