import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df = df.drop(['Name','Ticket','PassengerId'], axis=1)

#1.fill missing age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

#2.drop cabin entirely, too sparse to use
df = df.drop('Cabin', axis=1)

#3.fill missing Embarked with the most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#4.convert sex to numbers (0/1)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

#5.Convert embarked to numbers using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'])

print(df.isnull().sum())
print(df.head())

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: ",accuracy)

importances = model.feature_importances_
for name, score in zip(X.columns, importances):
    print(name, score)
