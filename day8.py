import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
print(df.head)
print("--------------------------------------------------------------")
print(df.info)
print("--------------------------------------------------------------")
print(df.isnull().sum())

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df.drop('medv', axis=1)
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE ",mean_absolute_error(y_test,predictions))
print("R2 Score ",r2_score(y_test,predictions))

importances = model.feature_importances_
for name, score in zip(X.columns, importances):
    print(name, score)